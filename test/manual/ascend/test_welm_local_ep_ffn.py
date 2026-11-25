"""Validate a DP4EP4 or DP8EP8 rank's arithmetic on one A5, with no process group."""
import ast
from contextvars import ContextVar
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class HostLocalScopeTest(unittest.TestCase):
    def test_local_ffn_region_preserves_nested_calls_and_resets_after_failure(self):
        src = (Path(__file__).resolve().parents[3]
               / "python/sglang/srt/hardware_backend/npu/graph_runner/welm_local_ep_ffn.py")
        node = next(x for x in ast.parse(src.read_text()).body
                    if isinstance(x, ast.FunctionDef) and x.name == "run_with_local_ffn")
        active = ContextVar("test_local_ffn", default=False)
        namespace = {"_local_ffn": active}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(src), "exec"), namespace)
        invoke = namespace["run_with_local_ffn"]

        def inner(hidden, topk):
            self.assertTrue(active.get())
            self.assertEqual((hidden, topk), ("rows", "route"))
            raise ValueError("dispatch failed")

        def outer(hidden, topk):
            self.assertTrue(active.get())
            with self.assertRaisesRegex(ValueError, "dispatch failed"):
                invoke(inner, hidden, topk)
            self.assertTrue(active.get())
            return hidden

        self.assertEqual(invoke(outer, "rows", "route"), "rows")
        self.assertFalse(active.get())
        with self.assertRaisesRegex(ValueError, "dispatch failed"):
            invoke(inner, "rows", "route")
        self.assertFalse(active.get())


class LocalExpertRankTest(unittest.TestCase):
    def test_actual_dispatch_sorted_ffn_and_partial_combine(self):
        self.check_rank(world=4, multi_stream=False)

    def test_shared_expert_keeps_the_original_stream_and_join(self):
        self.check_rank(world=4, multi_stream=True)

    def test_dp8_local_dispatch_and_partial_combine(self):
        self.check_rank(world=8, multi_stream=False)

    def test_dp8_shared_expert_keeps_original_stream(self):
        self.check_rank(world=8, multi_stream=True)

    def check_rank(self, *, world, multi_stream):
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
        from sglang.srt.hardware_backend.npu.graph_runner import welm_local_ep_ffn as local
        from sglang.srt.hardware_backend.npu.graph_runner import welm_ffn_tiles as tiles
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import decode_tile_scope
        from sglang.srt.layers.moe.fused_moe_triton import layer as fused
        from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher
        from sglang.srt.layers.moe.utils import MoeA2ABackend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models.welmv4 import Qwen2MoeSparseMoeBlock
        from sglang.srt.runtime_context import get_context, get_flags, get_parallel

        rank = 1 if world == 4 else 6
        local_experts = 512 // world
        first_expert = rank * local_experts
        self.enterContext(torch.inference_mode())
        self.enterContext(get_context().override_server_args(
            device="npu", tp_size=world, dp_size=world, ep_size=world, model_execution_backend="megakernel-native",
        ))
        self.enterContext(get_parallel().override(tp_rank=rank, tp_size=world, attn_tp_size=1, attn_tp_rank=0,
                                                  attn_dp_size=world, moe_tp_rank=0, moe_tp_size=1,
                                                  moe_ep_rank=rank, moe_ep_size=world))
        self.enterContext(patch.object(parallel_state, "_TP", SimpleNamespace(world_size=world, rank_in_group=rank)))
        self.enterContext(patch.object(get_flags().moe, "a2a_backend", MoeA2ABackend.DEEPEP))
        self.enterContext(patch.object(expert_location_dispatch, "get_global_expert_location_metadata",
                                      return_value=SimpleNamespace()))
        # FULL transport explicitly selects the separate real local dispatcher.
        # Its unused default dispatcher must never open an EP communication buffer.
        self.enterContext(patch.object(fused, "create_moe_dispatcher", side_effect=AscendTPDispatcher))
        self.enterContext(patch.dict(os.environ, SGLANG_NPU_WELMV4_MK_FFN_TILES="1",
                                    WELM_NPU_MOE_USE_ND="1", SGLANG_NPU_USE_MULTI_STREAM=str(int(multi_stream)),
                                    SGLANG_NPU_WELMV4_MK_MODEL_ROW_TILES="0"))
        self.addCleanup(torch.set_default_dtype, torch.get_default_dtype())
        torch.set_default_dtype(torch.bfloat16)
        cfg = SimpleNamespace(num_hidden_layers=1, num_experts=512, num_experts_per_tok=10,
                              moe_intermediate_size=512, shared_expert_intermediate_size=512, hidden_size=2048,
                              hidden_act="silu", router_score_func="sigmoid", moe_routing_type="expert_bias",
                              norm_topk_prob=False, has_shared_expert_gate=False)
        with torch.device("npu"):
            model = Qwen2MoeSparseMoeBlock(0, cfg)
        self.assertTrue(model.experts.welm_local_ffn_tiles)
        self.assertEqual(tuple(model.experts.w13_weight.shape), (local_experts, 1024, 2048))
        dispatcher = model.experts.local_ep_dispatcher
        self.assertEqual((dispatcher.first_expert_idx, dispatcher.last_expert_idx), (first_expert, first_expert + local_experts))
        rng = torch.Generator().manual_seed(7392)

        def random(shape):
            return (torch.randint(-3, 4, shape, generator=rng, dtype=torch.int8)
                    .to(torch.bfloat16).div_(32).npu())

        for param in model.parameters():
            if param.ndim == 3:
                values = random((8, *param.shape[1:]))
                for start in range(0, len(param), 8):
                    param[start:start+8].copy_(values * (1 + start // 8 % 4) / 4)
            else:
                param.copy_(random(param.shape))
        model.experts.quant_method.process_weights_after_loading(model.experts)
        model.prepare_npu_router_compute_weight_t(torch.bfloat16)
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE, enable_kv_mirror=False,
                                welm_prefill_graph=None, welmv4_npu_deepep_scattered=False)
        events = []
        streams = {}

        class PartialGroup:
            def all_reduce(self, partial):
                events.append("all_reduce")
                return partial  # Only this rank's contribution is under test.

        group = PartialGroup()

        def wrap(fn, name, fast=False):
            def recording(*args, **kwargs):
                result = fn(*args, **kwargs)
                events.append(name if not fast else (name, result is not None))
                streams[name] = torch.npu.current_stream().npu_stream
                return result
            return recording

        self.enterContext(patch.object(dispatcher, "dispatch", wrap(dispatcher.dispatch, "dispatch")))
        self.enterContext(patch.object(dispatcher, "combine", wrap(dispatcher.combine, "combine")))
        self.enterContext(patch.object(local, "try_sorted_experts", wrap(local.try_sorted_experts, "sorted", True)))
        self.enterContext(patch.object(tiles, "try_shared_expert", wrap(tiles.try_shared_expert, "shared", True)))

        def run(x):
            return tuple(t.cpu() for t in model(x, None, batch, use_welm_local_ep_moe=True,
                                                resolved_moe_ep_group=group, return_components=True))

        for rows in (1, 2, 4, 8):
            for idle in (False, True):
                with self.subTest(rows=rows, idle=idle):
                    x = random((rows, 2048))
                    model.expert_bias.fill_(-4)
                    model.expert_bias[:(10 if idle else 5)].fill_(4)
                    if not idle:
                        model.expert_bias[first_expert:first_expert+5].fill_(4)
                    events.clear()
                    expected = run(x)
                    self.assertNotIn(("sorted", True), events)
                    events.clear()
                    with decode_tile_scope(single_pk=False):
                        got = run(x)
                    self.assertEqual(events, [("shared", True), "dispatch", ("sorted", True), "combine", "all_reduce"])
                    self.assertEqual(streams["sorted"], streams["combine"])
                    self.assertEqual(streams["shared"] != streams["dispatch"], multi_stream)
                    for actual, reference in zip(got, expected):
                        torch.testing.assert_close(actual.float(), reference.float(), atol=1/128, rtol=1/64)
                    if idle:
                        self.assertEqual(got[1].count_nonzero().item(), 0)


class PackedRankReplayTest(unittest.TestCase):
    def test_replay_group_counts_and_inactive_tail(self):
        self.check_replay(world=4)

    def test_dp8_packed_counts_and_inactive_tail(self):
        self.check_replay(world=8)

    def check_replay(self, world):
        local_experts = 512 // world
        try:
            import torch
            import torch_npu
        except ModuleNotFoundError as exc:
            if exc.name not in {"torch", "torch_npu"}:
                raise
            self.skipTest(str(exc))
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires A5")
        from megakernel._fixed import capture_fixed
        from sglang.srt.hardware_backend.npu.graph_runner.welm_local_ep_ffn import run_with_local_ffn, try_sorted_experts
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import decode_tile_scope
        from sglang.srt.layers.quantization.unquant import NPUUnquantMoEMethod
        from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo, AscendRunnerInput

        rng = torch.Generator().manual_seed(61234)
        def random(shape):
            return (torch.randint(-3, 4, shape, generator=rng, dtype=torch.int8)
                    .to(torch.bfloat16).div_(32).npu())

        def bank(output, width):
            out = torch.empty((local_experts, output, width), dtype=torch.bfloat16, device="npu")
            prototype = random((8, output, width))
            for first in range(0, local_experts, 8):
                out[first:first+8].copy_(prototype * (1 + first//8 % 4) / 4)
            return out

        self.enterContext(torch.inference_mode())
        quant = AscendQuantInfo(w13_weight=bank(1024, 2048), w2_weight=bank(2048, 512))
        layer = torch.nn.Module()
        layer._dwdp_bound = layer._use_ascend_fuseep = False
        layer.moe_ep_size, layer.moe_tp_size, layer.num_local_experts = world, 1, local_experts
        layer.quant_config = None
        layer.w13_kernel, layer.w2_kernel = NPUUnquantMoEMethod(), NPUUnquantMoEMethod()
        layer.moe_runner_config = SimpleNamespace(activation="silu", gemm1_clamp_limit=None)

        def local(x, counts):
            return try_sorted_experts(AscendRunnerInput(x, None, counts, 1), quant, layer)

        def gmm(x, w, counts):
            return torch.ops.npu.npu_grouped_matmul(
                x=[x], weight=[w.transpose(1, 2)], group_list=counts, group_list_type=1,
                split_item=2, group_type=0, output_dtype=x.dtype,
            )[0]

        for capacity in (1, 2, 4, 8):
            rows = world * capacity * 10
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as build:
                x = random((rows, 2048))
                counts = torch.zeros(local_experts, device="npu", dtype=torch.int64)
                counts[0] = rows
                self.assertIsNone(local(x, counts))
                with decode_tile_scope(), patch.dict(os.environ, MK_NATIVE_TILE_DEPENDENCIES="1"):
                    plan = capture_fixed(lambda: run_with_local_ffn(local, x, counts), build, tile_chains=True)
                try:
                    scope, = plan.plans
                    self.assertEqual(list(scope.kernel_names), ["_welm_sorted_expert_ids", "_welm_expert_slot_mm",
                                     "_welm_ffn_swiglu", "_welm_expert_slot_mm", "_welm_clear_sorted_tail"])
                    self.assertTrue({(1, 2), (2, 3), (3, 4)} <= set(scope.tile_dependencies),
                                    (rows, scope.tile_dependencies))
                    self.assertIn(0, scope.dependencies[1], scope.dependencies)
                    self.assertNotIn((0, 1), scope.tile_dependencies)
                    for live in (rows, rows//5, 0):
                        x.copy_(random(x.shape))
                        counts.zero_()
                        counts[0], counts[local_experts//2], counts[-1] = live//3, live//3, live - 2*(live//3)
                        if live < rows:
                            x[live:].fill_(float("nan"))
                        if live:
                            ref = gmm(torch.ops.npu.npu_swiglu(gmm(x[:live], quant.w13_weight, counts)),
                                      quant.w2_weight, counts).cpu()
                        plan.outputs.fill_(float("nan"))
                        actual = plan.replay().cpu()
                        if live:
                            torch.testing.assert_close(actual[:live].float(), ref.float(), atol=1/128, rtol=1/64)
                        self.assertEqual(actual[live:].count_nonzero().item(), 0)
                finally:
                    plan.reset()


if __name__ == "__main__":
    unittest.main()
