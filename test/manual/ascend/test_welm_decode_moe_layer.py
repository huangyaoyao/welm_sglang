"""Exercise the complete ordinary WeLM MoE decode branch with one real BF16 layer."""
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class DecodeMoeLayerTest(unittest.TestCase):
    def test_routed_and_shared_on_model_forward(self):
        self._check_layer(fixed=False)

    def test_full_ffn_fixed_replay_with_distinct_cube_widths(self):
        self._check_layer(fixed=True)

    def _check_layer(self, *, fixed):
        try:
            import torch
            import torch_npu
        except ModuleNotFoundError as exc:
            if exc.name not in {"torch", "torch_npu"}:
                raise
            self.skipTest(str(exc))
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires A5")

        from sglang.srt.distributed import parallel_state
        from sglang.srt.eplb import expert_location_dispatch
        from sglang.srt.hardware_backend.npu.graph_runner import welm_ffn_tiles as ffn
        from sglang.srt.hardware_backend.npu.graph_runner import welm_narrow_projection as narrow
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import decode_tile_scope
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models.welmv4 import Qwen2MoeSparseMoeBlock
        from sglang.srt.runtime_context import get_context, get_parallel

        self.enterContext(torch.inference_mode())
        self.enterContext(get_context().override_server_args(
            device="npu", tp_size=1, model_execution_backend="megakernel-native",
        ))
        self.enterContext(get_parallel().override(tp_rank=0, tp_size=1, moe_tp_size=1,
                                                  moe_tp_rank=0, moe_ep_rank=0, moe_ep_size=1))
        self.enterContext(patch.object(parallel_state, "_TP", SimpleNamespace(world_size=1, rank_in_group=0)))
        # The single-rank launcher publishes identity expert placement.
        self.enterContext(patch.object(expert_location_dispatch, "get_global_expert_location_metadata",
                                      return_value=SimpleNamespace()))
        self.enterContext(patch.dict(os.environ, SGLANG_NPU_WELMV4_MK_FFN_TILES="1",
                                    WELM_NPU_MOE_USE_ND="1", SGLANG_NPU_USE_MULTI_STREAM="0",
                                    SGLANG_NPU_WELMV4_MK_MODEL_ROW_TILES="0"))
        self.addCleanup(torch.set_default_dtype, torch.get_default_dtype())
        torch.set_default_dtype(torch.bfloat16)
        config = SimpleNamespace(
            num_hidden_layers=1, num_experts=512, num_experts_per_tok=10, moe_intermediate_size=512,
            hidden_size=2048, shared_expert_intermediate_size=512, hidden_act="silu",
            router_score_func="sigmoid", moe_routing_type="expert_bias", norm_topk_prob=False,
            has_shared_expert_gate=False,
        )
        rng = torch.Generator().manual_seed(8351)

        def random(shape):
            return (torch.randint(-3, 4, shape, generator=rng, dtype=torch.int8)
                    .to(torch.bfloat16).div_(32).npu())

        with torch.device("npu"):
            model = Qwen2MoeSparseMoeBlock(0, config)
        for p in model.parameters():
            if p.ndim == 3:
                bank = random((8, *p.shape[1:]))
                for start in range(0, len(p), 8):
                    p[start:start+8].copy_(bank * (1 + (start // 8) % 4) / 4)
            else:
                p.copy_(random(p.shape))
        model.experts.quant_method.process_weights_after_loading(model.experts)
        model.prepare_npu_router_compute_weight_t(torch.bfloat16)
        self.assertTrue(model.use_mk_ffn_tiles)
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE, enable_kv_mirror=False, welm_prefill_graph=None)

        if fixed:
            from megakernel._fixed import capture_fixed

            for rows in (1, 2, 4, 8):
                with self.subTest(rows=rows), tempfile.TemporaryDirectory() as build:
                    x = random((rows, 2048))
                    with decode_tile_scope(single_pk=False), patch.dict(os.environ, MK_NATIVE_TILE_DEPENDENCIES="1"):
                        plan = capture_fixed(lambda: model(x, None, batch), build, tile_chains=True)
                    try:
                        found_shared = found_experts = found_router = False
                        for scope in plan.plans:
                            names = list(scope.kernel_names)
                            edges = set(scope.tile_dependencies)
                            found_router |= "_welm_narrow_projection_kernel" in names
                            for pos in range(len(names) - 2):
                                triple = names[pos:pos+3]
                                if triple == ["_welm_row_projection", "_welm_ffn_swiglu", "_welm_row_projection"]:
                                    found_shared = True
                                    if rows > 1:
                                        self.assertTrue({(pos, pos+1), (pos+1, pos+2)} <= edges,
                                                        (rows, triple, edges))
                                if triple == ["_welm_expert_slot_mm", "_welm_ffn_swiglu", "_welm_expert_slot_mm"]:
                                    found_experts = True
                                    self.assertTrue({(pos, pos+1), (pos+1, pos+2)} <= edges,
                                                    (rows, triple, edges))
                        self.assertEqual((found_shared, found_experts, found_router), (True, True, True))
                        for step in range(2):
                            x.copy_(random(x.shape))
                            if step:
                                x[-1].zero_()
                            expected = model(x, None, batch).cpu()
                            plan.outputs.fill_(float("nan"))
                            actual = plan.replay().cpu()
                            torch.testing.assert_close(actual.float(), expected.float(), atol=1/128, rtol=1/64)
                    finally:
                        plan.reset()
            return

        for rows in (1, 2, 4, 8):
            for tied in (False, True):
                with self.subTest(rows=rows, tie=tied):
                    x = random((rows, 2048))
                    if tied:
                        x[-1].zero_()
                        model.expert_bias.zero_()
                    else:
                        model.expert_bias.copy_(random(model.expert_bias.shape))
                    baseline = []
                    original = model.topk.forward_npu

                    def collect(*args, **kwargs):
                        out = original(*args, **kwargs)
                        baseline.append((out.router_logits.cpu(), out.topk_ids.cpu(), out.topk_weights.cpu()))
                        return out

                    calls = []
                    def watch(fn):
                        def record(*args):
                            out = fn(*args)
                            calls.append((fn.__name__, out is not None))
                            return out
                        return record

                    with patch.object(model.topk, "forward_npu", side_effect=collect):
                        expected = model(x, None, batch).cpu()
                        kernel = narrow._welm_narrow_projection_kernel
                        grids = []
                        class RouterKernel:
                            def __getitem__(self, grid):
                                grids.append(grid)
                                return kernel[grid]
                        with (decode_tile_scope(single_pk=False),
                              patch.object(ffn, "try_shared_expert", watch(ffn.try_shared_expert)),
                              patch.object(ffn, "try_routed_expert", watch(ffn.try_routed_expert)),
                              patch.object(narrow, "_welm_narrow_projection_kernel", RouterKernel())):
                            actual = model(x, None, batch).cpu()
                        self.assertEqual(grids, [(32,)])
                    self.assertEqual(calls, [("try_shared_expert", True), ("try_routed_expert", True)])
                    self.assertEqual(len(baseline), 2)
                    ref, got = baseline
                    self.assertEqual(got[0].dtype, torch.float32)
                    self.assertEqual(got[2].dtype, torch.float32)
                    torch.testing.assert_close(got[0], ref[0], atol=1e-6, rtol=1e-6)
                    torch.testing.assert_close(got[1], ref[1], atol=0, rtol=0)
                    torch.testing.assert_close(got[2], ref[2], atol=1e-6, rtol=1e-6)
                    torch.testing.assert_close(actual.float(), expected.float(), atol=1/128, rtol=1/64)


if __name__ == "__main__":
    unittest.main()
