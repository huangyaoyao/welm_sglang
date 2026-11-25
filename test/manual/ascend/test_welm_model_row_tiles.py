"""Per-row BF16 projections, base embeddings and OE accumulation on a single A5."""
import os
import tempfile
import unittest
from unittest.mock import patch


class WelmModelRowsTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            import torch_npu
        except ModuleNotFoundError as exc:
            if exc.name not in ("torch", "torch_npu"):
                raise
            self.skipTest(str(exc))
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires a single A5")
        self.torch = torch
        self.enterContext(torch.inference_mode())

    def test_main_attention_output_projection(self):
        from test_welm_page_gate_layer import WelmPageGateLayerTest

        class RowLayer(WelmPageGateLayerTest):
            def _check_graph(inner, plan, replace_attention):
                names = [name for scope in plan.plans for name in scope.kernel_names]
                inner.assertEqual(names.count("sigmoid_mul_kernel_npu"), 1, names)
                inner.assertEqual(names.count("_welm_row_projection"), 1, names)
                inner.assertIn("_welm_narrow_projection_kernel", names)
                scope, = [s for s in plan.plans if "_merge_page_attention" in s.kernel_names]
                merge = scope.kernel_names.index("_merge_page_attention")
                gate = scope.kernel_names.index("sigmoid_mul_kernel_npu")
                inner.assertIn((merge, gate), scope.tile_dependencies)
                if batch > 1:
                    o = scope.kernel_names.index("_welm_row_projection")
                    inner.assertEqual(scope.tile_dependency_ratios.get((gate, o)), (2, 1))

        test = RowLayer()
        self.addCleanup(test.doCleanups)
        test.setUp()
        # A deployment may still carry the old fusion switch. Its independent
        # twelve-head gate must feed the per-row output projection in every bucket.
        with patch.dict(os.environ, SGLANG_NPU_WELMV4_MK_MODEL_ROW_TILES="1",
                        SGLANG_NPU_WELMV4_MK_FUSE_PAGE_GATE="1"):
            for batch in (1, 2, 4, 8):
                with self.subTest(batch=batch):
                    test._check_layer(batch, -1)

    def test_mirror_projection(self):
        from megakernel._fixed import capture_fixed
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import decode_tile_scope
        from sglang.srt.models.welmv4 import Qwen2MoeAttention
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from types import SimpleNamespace

        torch = self.torch
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE, spec_info=None)
        for n in (6144, 8192):
            weight = (torch.randn((n, 2048), dtype=torch.float32) / 64).bfloat16().npu()
            layer = SimpleNamespace(supports_mk_page_gate=True, qkv_proj_weight=weight, qkv_proj_bias=None)
            for b in (1, 2, 4, 8):
                with self.subTest(batch=b, columns=n), patch.dict(os.environ,
                        SGLANG_NPU_WELMV4_MK_MODEL_ROW_TILES="1"), decode_tile_scope(), tempfile.TemporaryDirectory() as build:
                    x = torch.zeros((b, 2048), dtype=torch.bfloat16, device="npu")
                    def forward():
                        return Qwen2MoeAttention._project_mirror_weight(layer, x, batch)
                    plan = capture_fixed(forward, build, tile_chains=True)
                    try:
                        self.assertIn("_welm_row_projection", [k for s in plan.plans for k in s.kernel_names])
                        for _ in range(2):
                            x.copy_(torch.randn(x.shape, dtype=torch.bfloat16))
                            ref = torch.nn.functional.linear(x, weight)
                            torch.testing.assert_close(plan.replay(), ref, atol=1/128, rtol=1/64)
                    finally:
                        plan.reset()

    def test_embedding_and_projection_rows(self):
        import torch.nn.functional as F
        from megakernel._fixed import capture_fixed
        from sglang.srt.hardware_backend.npu.graph_runner.welm_row_projection import (
            try_base_embedding, try_oe_embedding, try_row_projection,
        )
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import decode_tile_scope
        from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

        torch = self.torch
        torch.manual_seed(9104)
        embedding = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
        torch.nn.Module.__init__(embedding)
        embedding.tp_size = 1
        embedding.num_embeddings = 64
        embedding.quant_method = UnquantizedEmbeddingMethod()
        embedding.weight = torch.nn.Parameter(torch.randn((64, 2048), dtype=torch.bfloat16).npu(), False)
        weight = (torch.randn((2048, 2048), dtype=torch.float32) * 0.01).bfloat16().npu()
        o_weight = (torch.randn((2048, 6144), dtype=torch.float32) * 0.01).bfloat16().npu()
        for batch in (1, 2, 4, 8):
            with self.subTest(batch=batch), patch.dict(os.environ,
                    SGLANG_NPU_WELMV4_MK_MODEL_ROW_TILES="1", MK_NATIVE_TILE_DEPENDENCIES="1"
            ), decode_tile_scope(), tempfile.TemporaryDirectory() as build:
                ids = torch.arange(batch, dtype=torch.int64, device="npu")
                concat = torch.ones((batch, 2048), dtype=torch.bfloat16, device="npu")
                attention = torch.ones((batch, 6144), dtype=torch.bfloat16, device="npu")
                def forward():
                    base = try_base_embedding(ids, embedding)
                    self.assertIsNotNone(base)
                    oe = try_oe_embedding(base, concat, weight)
                    o = try_row_projection(attention, o_weight)
                    self.assertIsNotNone(oe)
                    self.assertIsNotNone(o)
                    return oe, o

                plan = capture_fixed(forward, build, tile_chains=True)
                try:
                    names = [name for scope in plan.plans for name in scope.kernel_names]
                    self.assertEqual(names.count("_welm_row_projection"), 2, names)
                    self.assertEqual(names.count("_welm_base_lookup"), 1, names)
                    self.assertEqual(names.count("_welm_embedding_mean"), 1, names)
                    scopes = [s for s in plan.plans if "_welm_embedding_mean" in s.kernel_names]
                    scope, = scopes
                    target = scope.kernel_names.index("_welm_embedding_mean")
                    parents = {scope.kernel_names[i] for i in scope.dependencies[target]}
                    self.assertTrue({"_welm_base_lookup", "_welm_row_projection"} <= parents, parents)
                    if batch > 1:
                        tiled = {scope.kernel_names[a] for a, b in scope.tile_dependencies if b == target}
                        self.assertTrue({"_welm_base_lookup", "_welm_row_projection"} <= tiled, tiled)
                    for step in (1, 2):
                        ids.copy_((torch.arange(batch) * (step + 3) + 19 * step) % 64)
                        concat.copy_(torch.randn(concat.shape, dtype=torch.bfloat16))
                        attention.copy_(torch.randn(attention.shape, dtype=torch.bfloat16))
                        ref_base = F.embedding(ids, embedding.weight)
                        ref_oe = (ref_base + F.linear(concat, weight)) / 2
                        ref_o = F.linear(attention, o_weight)
                        for item in plan.outputs:
                            item.fill_(float("nan"))
                        result = plan.replay()
                        for actual, expected in zip(result, (ref_oe, ref_o)):
                            torch.testing.assert_close(actual, expected, atol=1/128, rtol=1/64)
                finally:
                    plan.reset()


if __name__ == "__main__":
    unittest.main()
