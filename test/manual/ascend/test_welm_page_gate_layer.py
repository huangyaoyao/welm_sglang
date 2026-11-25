"""A5 integration regression for one ordinary WeLM target attention layer."""
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


GATE_FLAG = "SGLANG_NPU_WELMV4_MK_PAGE_GATE_TILES"
LEGACY_FUSE_GATE_FLAG = "SGLANG_NPU_WELMV4_MK_FUSE_PAGE_GATE"


class WelmPageGateLayerTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            import torch_npu
        except ModuleNotFoundError as error:
            if error.name not in {"torch", "torch_npu"}:
                raise
            self.skipTest(f"A5 test requires {error.name}")
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires an Ascend A5 NPU")
        self.torch = torch
        self.torch_npu = torch_npu
        self.rng = torch.Generator().manual_seed(3904)
        self.enterContext(patch.dict(os.environ, {
            "ASCEND_USE_FIA": "1",
            "SGLANG_NPU_USE_SCATTER_PA_KV_CACHE": "1",
            "WELM_NPU_USE_FLASH_ATTN": "1",
            "SGLANG_NPU_WELMV4_MK_PAGE_ATTENTION": "1",
            GATE_FLAG: "1",
            LEGACY_FUSE_GATE_FLAG: "0",
            "SGLANG_NPU_WELMV4_MK_BF16_QKV": "1",
            "SGLANG_NPU_WELMV4_FUSED_QKV": "0",
            "SGLANG_NPU_USE_MULTI_STREAM": "0",
            "MK_NATIVE_TILE_DEPENDENCIES": "1",
        }))
        from sglang.srt.distributed import parallel_state
        from sglang.srt.runtime_context import get_context, get_parallel

        self.enterContext(get_context().override_server_args(
            device="npu", tp_size=1, attention_backend="ascend", model_execution_backend="megakernel-native",
        ))
        self.enterContext(get_parallel().override(tp_rank=0, tp_size=1, attn_tp_rank=0, attn_tp_size=1))
        # Only topology is supplied. The real gate, QKV and output GEMMs still run.
        self.enterContext(patch.object(parallel_state, "_TP", SimpleNamespace(world_size=1, rank_in_group=0)))
        self.enterContext(torch.inference_mode())
        self.addCleanup(torch.set_default_dtype, torch.get_default_dtype())
        torch.set_default_dtype(torch.bfloat16)

    def _random(self, shape):
        torch = self.torch
        return (torch.randint(-3, 4, shape, generator=self.rng).float() / 128).to(torch.bfloat16)

    def _pool(self, pages):
        from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool

        return NPUMHATokenToKVPool(
            size=(pages - 1) * 64, page_size=64, dtype=self.torch.bfloat16,
            head_num=2, head_dim=256, layer_num=2, device="npu",
            enable_memory_saver=False, enable_alt_stream=False,
        )

    def _check_pool(self, pool, pages, formats):
        shape = (2, pages, 64, 2, 256)
        backing_tensors = pool.get_hicache_transfer_buffers()
        for backing, get_layer in zip(backing_tensors, (pool.get_key_buffer, pool.get_value_buffer)):
            self.assertEqual(tuple(backing.shape), shape)
            self.assertTrue(backing.is_contiguous())
            self.assertIn(self.torch_npu.get_npu_format(backing), formats)
            for index in range(2):
                layer = get_layer(index)
                offset = index * pages * 64 * 2 * 256
                self.assertEqual(tuple(layer.shape), (pages * 64, 1, 2, 256))
                self.assertTrue(layer.is_contiguous())
                self.assertIn(self.torch_npu.get_npu_format(layer), formats)
                self.assertEqual(layer.untyped_storage().data_ptr(), backing.untyped_storage().data_ptr())
                self.assertEqual(layer.storage_offset(), backing.storage_offset() + offset)
                self.assertEqual(layer.data_ptr(), backing.data_ptr() + offset * backing.element_size())

    def test_pool_layout_requires_opt_in(self):
        torch = self.torch
        pages = 4
        default = torch.zeros((2, pages, 64, 2, 256), device="npu", dtype=torch.bfloat16)
        ordinary = (self.torch_npu.get_npu_format(default),)
        for flag in (None, "0"):
            with self.subTest(flag=flag), patch.dict(os.environ):
                if flag is None:
                    del os.environ[GATE_FLAG]
                else:
                    os.environ[GATE_FLAG] = flag
                self._check_pool(self._pool(pages), pages, ordinary)
        self._check_pool(self._pool(pages), pages, (0, 2))

    def _layer(self, window):
        from sglang.srt.models.welmv4 import Qwen2MoeAttention

        layer = Qwen2MoeAttention(
            hidden_size=2048, num_heads=24, num_kv_heads=2, head_dim=256,
            layer_id=1, layer_idx=1, k_norm=True, qkv_bias=False, qk_rope_head_dim=64,
            max_position_embeddings=640, sliding_window_size=window,
            enable_attn_sink_layerwise=[True, True], alt_stream=None, is_nextn=False,
        ).eval().npu()
        for name, param in layer.named_parameters():
            if "norm" in name:
                param.fill_(1)
            else:
                param.copy_(self._random(param.shape))
        return layer

    def _check_graph(self, plan, replace_attention):
        nodes = {}
        for scope_id, scope in enumerate(plan.plans):
            for node, name in enumerate(scope.kernel_names):
                nodes.setdefault(name, []).append((scope_id, node))
        (merge_scope, merge), = nodes["_merge_page_attention"]
        if replace_attention:
            # The hook returns a new tensor. That tensor must receive its own
            # sigmoid multiply after the hook, with ordinary dependency analysis.
            self.assertEqual(len(nodes.get("sigmoid_mul_kernel_npu", [])), 1)
            return
        (gate_scope, gate), = nodes["sigmoid_mul_kernel_npu"]
        self.assertEqual(merge_scope, gate_scope, nodes)
        scope = plan.plans[gate_scope]
        self.assertIn(merge, scope.dependencies[gate])
        edge = (merge, gate)
        fine = os.environ["MK_NATIVE_TILE_DEPENDENCIES"] != "0"
        self.assertEqual(edge in scope.tile_dependencies, fine)
        self.assertEqual(scope.tile_dependency_ratios.get(edge), (1, 3) if fine else None)

    def _check_layer(self, batch_size, window, replace_attention=False):
        from megakernel._fixed import capture_fixed
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import (
            decode_tile_scope, install_welm_tile_layouts,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
        from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

        torch = self.torch
        install_welm_tile_layouts()
        layer = self._layer(window)
        if replace_attention:
            # The small random layer otherwise produces sigmoid values close to
            # 0.5. Exercise changes in the input gate on every replay.
            layer.gate_proj.weight.mul_(128)
        if replace_attention:
            # Adding a bias before the gate differs from adding it after the
            # gate. Compare against the same hook running on unfused eager PAGE.
            hook = layer.attn.register_forward_hook(lambda module, args, output: output + 1)
            self.addCleanup(hook.remove)
        columns = 9 if window == 511 else 3
        pages = 1 + columns * batch_size
        pool = self._pool(pages)
        self._check_pool(pool, pages, (0, 2))
        cache = (pool.get_key_buffer(1), pool.get_value_buffer(1))
        seeds = [self._random(tensor.shape).npu() for tensor in cache]

        # Drive the production decode entry point with one worker's static metadata.
        table = torch.arange(1, pages, dtype=torch.int32).reshape(batch_size, columns)
        lengths = torch.zeros(batch_size, dtype=torch.int32, device="npu")
        slots = torch.zeros_like(lengths)
        positions = torch.zeros(batch_size, dtype=torch.int64, device="npu")
        hidden = torch.empty((batch_size, 2048), device="npu", dtype=torch.bfloat16)
        backend = AscendAttnBackend.__new__(AscendAttnBackend)
        backend.token_to_kv_pool = pool
        backend.use_mk_paged_kv = False
        backend.use_mk_page_attention = True
        backend.use_mla = False
        backend.use_welm_flash_attn = True
        backend.capture_welm_decode_metadata = True
        backend.graph_mode = True
        backend.enable_torch_compile = False
        backend.page_size = 64
        backend.forward_metadata = SimpleNamespace(
            block_tables=table.npu(), welm_flash_seqused_kv=lengths, welm_decode_metadata_in_graph=True,
        )
        batch = ForwardBatch.__new__(ForwardBatch)
        batch.forward_mode = ForwardMode.DECODE
        batch.batch_size = batch_size
        batch.out_cache_loc = slots
        batch.spec_info = None
        batch.welm_prefill_graph = None
        batch.welm_prefill_graph_phase = None
        batch.enable_kv_mirror = False

        def set_inputs(step):
            length_pair = (513, 511) if window == 511 else (65, 33)
            used = torch.tensor([length_pair[row % 2] + step for row in range(batch_size)], dtype=torch.int32)
            position = used - 1
            positions.copy_(position)
            lengths.copy_(used)
            slots.copy_(table[torch.arange(batch_size), position.long() // 64] * 64 + position % 64)
            hidden.copy_(self._random(hidden.shape))
            for view, seed in zip(cache, seeds):
                view.copy_(seed)

        def forward():
            return layer(positions, hidden, batch)

        # WelmNativeBackend.capture_one uses single-PK and retains the
        # traced tensor owners for these four single-device coarse buckets.
        single_pk = os.environ["MK_NATIVE_TILE_DEPENDENCIES"] == "0"
        with forward_context(ForwardContext(backend)), decode_tile_scope(single_pk=single_pk), tempfile.TemporaryDirectory() as build:
            set_inputs(0)
            # Keep the hook-output case free of additional observers.
            if replace_attention:
                forward()
            else:
                order = []
                with layer.gate_proj.register_forward_pre_hook(lambda *_: order.append("gate")), \
                        layer.attn.register_forward_pre_hook(lambda *_: order.append("attention")):
                    forward()
                self.assertEqual(order, ["gate", "attention"])
            plan = capture_fixed(forward, build, tile_chains=True, retain_intermediates=single_pk)
            try:
                self._check_graph(plan, replace_attention)
                for step in range(2):
                    set_inputs(step)
                    # The original order computes the gate after attention and uses the ordinary gate grid.
                    with patch.dict(os.environ, {GATE_FLAG: "0", LEGACY_FUSE_GATE_FLAG: "0"}), decode_tile_scope(single_pk=False):
                        expected = forward().cpu()
                    expected_cache = [tensor.cpu() for tensor in cache]
                    for view, seed in zip(cache, seeds):
                        view.copy_(seed)
                    actual = plan.replay().cpu()
                    torch.testing.assert_close(actual.float(), expected.float(), atol=1 / 128, rtol=1 / 64)
                    if replace_attention:
                        relative_rms = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
                        self.assertLessEqual(float(relative_rms), 0.02)
                    for view, oracle in zip(cache, expected_cache):
                        self.assertTrue(torch.equal(view.cpu(), oracle))
                    for first in (pool.get_key_buffer(0), pool.get_value_buffer(0)):
                        self.assertEqual(first.cpu().count_nonzero().item(), 0)
            finally:
                plan.reset()

    def test_target_decode_attention(self):
        for size, window in [(1, -1), (2, -1), (4, -1), (8, -1), (2, 511)]:
            with self.subTest(batch=size, window=window):
                self._check_layer(size, window)

    @patch.dict(os.environ, {LEGACY_FUSE_GATE_FLAG: "1"})
    def test_legacy_fusion_flag_keeps_singleton_coarse_gate(self):
        with patch.dict(os.environ, MK_NATIVE_TILE_DEPENDENCIES="0"):
            self._check_layer(2, 511)

    @patch.dict(os.environ, {LEGACY_FUSE_GATE_FLAG: "1"})
    def test_attention_output_hook_preserves_unfused_gate(self):
        self._check_layer(2, -1, replace_attention=True)


if __name__ == "__main__":
    unittest.main()
