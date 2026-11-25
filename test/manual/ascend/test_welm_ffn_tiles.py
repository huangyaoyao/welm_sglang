"""Per-slot and per-row boundaries for BF16 WeLM experts."""
import ast
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class HostFFNLayoutTest(unittest.TestCase):
    def test_each_program_reads_complete_rows_and_declares_indirect_weights_shared(self):
        src = Path(__file__).resolve().parents[3] / "python/sglang/srt/hardware_backend/npu/graph_runner/welm_ffn_tiles.py"
        names = {"_slot_mm_layout", "_swiglu_layout", "_combine_layout"}
        functions = [n for n in ast.parse(src.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = {}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(src), "exec"), namespace)

        class Tensor:
            def __init__(self, rows):
                self.shape = (rows,)

            def element_size(self):
                return 2

        for batch, group in ((1, 1), (2, 1), (4, 2), (8, 5)):
            x, out, w13, w2, ids, scores = (Tensor(batch) for _ in range(6))
            packed, act, down = (Tensor(batch * 10) for _ in range(3))
            expand = namespace["_slot_mm_layout"](x, w13, ids, packed, 2048, 1024, 512, True, group)
            contraction = namespace["_slot_mm_layout"](act, w2, ids, down, 512, 2048, 512, False, group)
            swiglu = namespace["_swiglu_layout"](packed, act, group)
            combine = namespace["_combine_layout"](down, ids, scores, out, 512)
            self.assertTrue(expand["keep_cube_width"] and contraction["keep_cube_width"])
            self.assertNotIn("keep_cube_width", swiglu)
            self.assertNotIn("keep_cube_width", combine)
            self.assertEqual(expand["shared_reads"], [w13, ids, x])
            self.assertEqual(contraction["shared_reads"], [w2, ids])
            self.assertEqual(expand["writes"], swiglu["reads"])
            self.assertEqual(swiglu["writes"], contraction["reads"])
            self.assertEqual([m["count"] for m in (expand, swiglu, contraction)], [batch * 10] * 3)
            self.assertEqual([m["block"] for m in (expand, swiglu, contraction)], [group] * 3)
            self.assertEqual(combine["reads"], [(down, 10 * 2048 * 2)])
            self.assertEqual(combine["shared_reads"], [ids, scores])
            self.assertEqual(combine["count"], batch)


    def test_routed_dispatch_preserves_specialized_backends_and_hooks(self):
        src = (Path(__file__).resolve().parents[3]
               / "python/sglang/srt/hardware_backend/npu/graph_runner/welm_ffn_tiles.py")
        guard = next(n for n in ast.parse(src.read_text()).body
                     if isinstance(n, ast.FunctionDef) and n.name == "try_routed_expert")
        # The device classes are identity sentinels here. Execute the actual production guard.
        guard.body = [node for node in guard.body if not isinstance(node, ast.ImportFrom)]
        names = {name: type(name, (), {}) for name in (
            "FusedMoE", "StandardTopKOutput", "AscendTPDispatcher", "NPUUnquantMoEMethod",
        )}
        names.update(_plain_module=lambda m: not m.during_hook, try_expert_slots=lambda *args: args)
        exec(compile(ast.Module(body=[guard], type_ignores=[]), str(src), "exec"), names)
        layer = names["FusedMoE"]()
        topk = names["StandardTopKOutput"]()
        topk.topk_ids, topk.topk_weights = object(), object()
        layer.during_hook = layer._dwdp_bound = layer._use_ascend_fuseep = False
        layer.moe_tp_size = layer.moe_ep_size = 1
        layer.num_experts = layer.num_local_experts = 512
        layer.quant_config, layer.num_fused_shared_experts = None, 0
        layer.dispatcher = names["AscendTPDispatcher"]()
        layer.dispatcher._original_dispatch_func = layer.dispatcher._original_combine_func = None
        layer.w13_kernel = names["NPUUnquantMoEMethod"]()
        layer.w2_kernel = names["NPUUnquantMoEMethod"]()
        layer.w13_weight, layer.w2_weight = object(), object()
        layer.moe_runner_config = SimpleNamespace(
            is_gated=True, activation="silu", no_combine=False, apply_router_weight_on_input=False,
            gemm1_clamp_limit=None,
        )
        x = object()
        call = names["try_routed_expert"]
        self.assertEqual(call(x, topk, layer), (x, topk.topk_ids, topk.topk_weights,
                                               layer.w13_weight, layer.w2_weight))
        cases = ((layer, "during_hook", True), (layer, "moe_ep_size", 4), (layer, "moe_tp_size", 2),
                 (layer, "quant_config", object()), (layer, "_dwdp_bound", True),
                 (layer, "_use_ascend_fuseep", True), (layer, "num_fused_shared_experts", 1),
                 (layer.dispatcher, "_original_dispatch_func", object()),
                 (layer.dispatcher, "_original_combine_func", object()),
                 (layer.moe_runner_config, "gemm1_clamp_limit", 2),
                 (layer.moe_runner_config, "apply_router_weight_on_input", True))
        for obj, key, value in cases:
            with self.subTest(key=key), patch.object(obj, key, value):
                self.assertIsNone(call(x, topk, layer))


class DeviceFFNTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            import torch_npu
        except ModuleNotFoundError as exc:
            if exc.name not in ("torch", "torch_npu"):
                raise
            self.skipTest(str(exc))
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires A5")
        self.torch = torch
        self.rng = torch.Generator().manual_seed(998711)

    def random(self, *shape):
        torch = self.torch
        return (torch.randint(-3, 4, shape, generator=self.rng, dtype=torch.int8).to(torch.bfloat16) / 32).npu()

    def test_shared_expert_matches_ordinary_mlp(self):
        from megakernel._fixed import capture_fixed
        from sglang.srt.distributed import parallel_state
        from sglang.srt.hardware_backend.npu.graph_runner.welm_ffn_tiles import try_shared_expert
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import decode_tile_scope
        from sglang.srt.models.welmv4 import Qwen2MoeMLP
        from sglang.srt.runtime_context import get_context, get_parallel

        torch = self.torch
        topology = SimpleNamespace(world_size=1, rank_in_group=0)
        with torch.inference_mode(), get_context().override_server_args(
            device="npu", tp_size=1, model_execution_backend="megakernel-native",
        ), get_parallel().override(tp_rank=0, tp_size=1), patch.object(parallel_state, "_TP", topology):
            layer = Qwen2MoeMLP(2048, 512, "silu", reduce_results=False, tp_rank=0, tp_size=1).to(
                device="npu", dtype=torch.bfloat16)
            for weight in layer.parameters():
                weight.copy_(self.random(*weight.shape))
            for rows in (1, 2, 4, 8):
                with self.subTest(rows=rows), tempfile.TemporaryDirectory() as build:
                    x = self.random(rows, 2048)
                    with decode_tile_scope(), patch.dict(os.environ, MK_NATIVE_TILE_DEPENDENCIES="1"):
                        with layer.act_fn.register_forward_hook(lambda *args: None):
                            self.assertIsNone(try_shared_expert(x, layer))
                        plan = capture_fixed(lambda: try_shared_expert(x, layer), build, tile_chains=True)
                    try:
                        scope, = plan.plans
                        self.assertEqual(list(scope.kernel_names),
                                         ["_welm_row_projection", "_welm_ffn_swiglu", "_welm_row_projection"])
                        self.assertEqual(set(scope.tile_dependencies), {(0, 1), (1, 2)} if rows > 1 else set())
                        for _ in range(3):
                            x.copy_(self.random(*x.shape))
                            expected = layer(x).cpu()
                            plan.outputs.fill_(float("nan"))
                            torch.testing.assert_close(plan.replay().cpu().float(), expected.float(),
                                                       atol=1/128, rtol=1/64)
                    finally:
                        plan.reset()

    def test_routed_experts_share_only_immutable_weights(self):
        from megakernel._fixed import capture_fixed
        from sglang.srt.hardware_backend.npu.graph_runner.welm_ffn_tiles import try_expert_slots
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import decode_tile_scope
        from sglang.srt.hardware_backend.npu.moe.topk import fused_expert_bias_topk_npu

        torch = self.torch
        experts = 32
        w13 = self.random(experts, 1024, 2048)
        w2 = self.random(experts, 2048, 512)
        npu = torch.ops.npu

        def native(x, ids, scores):
            rows, order, counts, _ = npu.npu_moe_init_routing_v2(
                x, ids, active_num=ids.numel(), expert_num=experts, expert_tokens_num_type=1,
                expert_tokens_num_flag=True, active_expert_range=[0, experts], quant_mode=-1,
            )
            def gmm(y, weight):
                return npu.npu_grouped_matmul(
                    x=[y], weight=[weight.transpose(1, 2)], group_list=counts.to(torch.int64),
                    group_list_type=1, split_item=2, group_type=0, output_dtype=x.dtype,
                )[0]
            projected = gmm(npu.npu_swiglu(gmm(rows, w13)), w2)
            return npu.npu_moe_finalize_routing(
                projected, skip1=None, skip2=None, bias=None, scales=scores,
                expanded_src_to_dst_row=order, export_for_source_row=ids, drop_pad_mode=2,
            )

        def routes(batch, step):
            logits = self.random(batch, experts).float() * 20
            bias = torch.linspace(-1, 1, experts, device="npu").roll(step * 4)
            return fused_expert_bias_topk_npu(logits, bias, top_k=10, scoring_func="sigmoid", renormalize=False)

        for batch in (1, 2, 4, 8):
            with self.subTest(batch=batch), torch.inference_mode(), tempfile.TemporaryDirectory() as build:
                x = self.random(batch, 2048)
                scores, ids = routes(batch, 0)
                with decode_tile_scope(), patch.dict(os.environ, MK_NATIVE_TILE_DEPENDENCIES="1"):
                    plan = capture_fixed(lambda: try_expert_slots(x, ids, scores, w13, w2), build, tile_chains=True)
                try:
                    scope, = plan.plans
                    self.assertEqual(list(scope.kernel_names), ["_welm_expert_slot_mm", "_welm_ffn_swiglu",
                                                              "_welm_expert_slot_mm", "_welm_combine_expert_slots"])
                    self.assertIn((0, 1), scope.tile_dependencies, scope.tile_dependencies)
                    self.assertIn((1, 2), scope.tile_dependencies, scope.tile_dependencies)
                    ratio = {1: None, 2: (10, 1), 4: (5, 1), 8: (2, 1)}[batch]
                    self.assertEqual(scope.tile_dependency_ratios.get((2, 3)), ratio,
                                     (batch, scope.tile_dependencies, scope.tile_dependency_ratios))
                    for step in range(3):
                        x.copy_(self.random(*x.shape))
                        new_scores, new_ids = routes(batch, step)
                        scores.copy_(new_scores)
                        ids.copy_(new_ids)
                        if step == 2:
                            scores[-1].zero_()
                        expected = native(x, ids, scores).cpu()
                        if step == 2:
                            ids[-1].fill_(-1)
                        plan.outputs.fill_(float("nan"))
                        actual = plan.replay().cpu()
                        torch.testing.assert_close(actual.float(), expected.float(), atol=1/128, rtol=1/64)
                        if step == 2:
                            self.assertEqual(actual[-1].count_nonzero().item(), 0)
                finally:
                    plan.reset()


if __name__ == "__main__":
    unittest.main()
