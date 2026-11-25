"""WeLM BF16 single-card LM head contract and A5 replay checks."""
import ast
import tempfile
import unittest
from contextlib import ExitStack
from contextvars import ContextVar
from math import prod
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[3] / "python/sglang/srt/hardware_backend/npu/graph_runner/welm_lm_head.py"


class LayoutTests(unittest.TestCase):
    def test_row_fanout_and_vocab_stripes(self):
        tree = ast.parse(MODULE.read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.Assign) or
                     isinstance(n, ast.FunctionDef) and (n.name.endswith("_layout") or n.name == "_welm_projection_stripes")]
        ns = {"ContextVar": ContextVar}
        exec(compile(tree, str(MODULE), "exec"), ns)
        def tensor(shape, size):
            return SimpleNamespace(shape=shape, numel=lambda: prod(shape), element_size=lambda: size)
        for rows, stripes in ns["_STRIPES"].items():
            width = 155648 // stripes
            self.assertLessEqual(rows * stripes, 32)
            self.assertEqual(width % 512, 0)
            x = tensor((rows, 2048), 2)
            copies = tensor((rows, stripes, 2048), 2)
            w = tensor((155648, 2048), 2)
            bf = tensor((rows, 155648), 2)
            fp = tensor((rows, 155648), 4)
            fan = ns["_fanout_layout"](x, copies, stripes)
            mm = ns["_projection_layout"](copies, w, bf, stripes, width)
            cast = ns["_cast_layout"](bf, fp, width)
            self.assertEqual((fan["count"], fan["reads"], fan["writes"]),
                             (rows, [(x, 4096)], [(copies, stripes * 4096)]))
            self.assertEqual((mm["count"], mm["reads"], mm["shared_reads"], mm["writes"]),
                             (rows * stripes, [(copies, 4096)], [w], [(bf, 2 * width)]))
            self.assertEqual((cast["count"], cast["reads"], cast["writes"]),
                             (rows * stripes, [(bf, 2 * width)], [(fp, 4 * width)]))
            for tp_size in (1, 4, 8):
                local_stripes = ns["_welm_projection_stripes"](rows, tp_size)
                local_width = 155648 // tp_size // local_stripes
                self.assertEqual(local_width % 512, 0)
                self.assertLessEqual(rows * local_stripes, 32)
                self.assertEqual(local_width * local_stripes * tp_size, 155648)
            for owner in range(rows * stripes):
                self.assertEqual((owner * width) // 155648, owner // stripes)
                self.assertEqual(((owner+1) * width - 1) // 155648, owner // stripes)


class LMHeadDeviceTest(unittest.TestCase):
    def _check_vocab(self, rows, retained=None):

        try:
            import torch
            import torch_npu
        except ModuleNotFoundError as error:
            if error.name not in {"torch", "torch_npu"}:
                raise
            self.skipTest(str(error))
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires an Ascend A5")
        from megakernel._fixed import capture_fixed
        from sglang.srt.hardware_backend.npu.graph_runner.welm_lm_head import (
            WelmDecodeLogitsProcessor, try_welm_final_norm,
        )
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import (
            decode_tile_scope, install_welm_tile_layouts,
        )
        from sglang.srt.layers.logits_processor import LogitsProcessor
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
        from sglang.srt.layers.welmv4_op import WelmV4FusedRMSNorm
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode

        width, hidden = 155648, 2048
        rng = torch.Generator().manual_seed(194)
        with torch.inference_mode():
            if retained is None or not retained.shared:
                head = ParallelLMHead.__new__(ParallelLMHead)
                torch.nn.Module.__init__(head)
                head.tp_size = 1
                head.quant_config = head.quant_method = None
                weight = torch.empty(width * hidden, dtype=torch.bfloat16, device="npu").view(width, hidden)
                for start in range(0, width, 2048):
                    stop = min(width, start + 2048)
                    # Dyadic values make all full-K sums exact in FP32 for the input range.
                    block = torch.randint(-2, 3, (stop-start, hidden), dtype=torch.int16, generator=rng)
                    weight[start:stop].copy_(block.to(torch.bfloat16).mul_(1/32))
                weight[0, 0] = 128
                weight[-1, 0] = -128
                head.weight = torch.nn.Parameter(weight, requires_grad=False)
                norm = WelmV4FusedRMSNorm(hidden, eps=1e-6, weight_dtype=torch.bfloat16).npu()
                norm.weight.copy_(torch.linspace(0.6, 1.4, hidden))
                proc = WelmDecodeLogitsProcessor.__new__(WelmDecodeLogitsProcessor)
                torch.nn.Module.__init__(proc)
                proc.vocab_size = width
                proc.use_fp32_lm_head = proc.do_tensor_parallel_all_gather = proc.do_tensor_parallel_all_gather_dp_attn = False
                proc.final_logit_softcapping = proc.logit_scale = proc.rl_on_policy_target = None
                if retained is not None:
                    retained.shared = (head, weight, norm, proc)
            else:
                head, weight, norm, proc = retained.shared
            install_welm_tile_layouts()
            with ExitStack() as local:
                owner = local if retained is None else retained.owner
                build = owner.enter_context(tempfile.TemporaryDirectory())
                x = torch.ones((rows, hidden), dtype=torch.bfloat16, device="npu")
                target = torch.empty((rows+1, width), dtype=torch.float32, device="npu")
                out = target[1:]
                input_ids = torch.arange(rows, dtype=torch.int64, device="npu")
                meta = SimpleNamespace(forward_mode=ForwardMode.DECODE, next_token_logits_buffer=out,
                                       draft_extend_select_index=None, capture_hidden_mode=CaptureHiddenMode.NULL,
                                       extend_return_logprob=False, mm_input_embeds=None)
                def forward():
                    normalized = try_welm_final_norm(norm, x)
                    self.assertIsNotNone(normalized)
                    return proc(input_ids, normalized, head, meta).next_token_logits
                if retained is not None and retained.backend is not None:
                    backend = retained.backend
                    with patch.dict("os.environ", MK_NATIVE_TILE_DEPENDENCIES="1"):
                        backend.capture_one(rows, forward, capture_inputs=(x, out, input_ids))
                    plan, _ = backend._plans[rows]
                else:
                    with decode_tile_scope(), patch.dict("os.environ", MK_NATIVE_TILE_DEPENDENCIES="1"):
                        plan = capture_fixed(forward, build, tile_chains=True)
                    owner.callback(plan.reset)
                found = [(s,i) for s in plan.plans for i,name in enumerate(s.kernel_names)
                         if name == "_welm_head_projection"]
                self.assertEqual(len(found), 1, plan.execution_summary)
                scope, idx = found[0]
                fan = scope.kernel_names.index("_welm_head_fanout")
                cast = scope.kernel_names.index("_welm_head_cast")
                norm_idx = scope.kernel_names.index("rms_norm_kernel")
                self.assertIn(norm_idx, scope.dependencies[fan])
                self.assertIn(fan, scope.dependencies[idx])
                if rows == 1:
                    # MK keeps a single-producer-grid dependency complete.
                    self.assertNotIn((norm_idx, fan), scope.tile_dependencies)
                    self.assertNotIn((fan, idx), scope.tile_dependencies)
                else:
                    self.assertIn((norm_idx, fan), scope.tile_dependencies)
                    self.assertEqual(scope.tile_dependency_ratios[fan, idx], (1, {2:16, 4:8, 8:4}[rows]))
                self.assertIn((idx, cast), scope.tile_dependencies)
                def replay(step):
                    data = torch.randint(-12, 13, x.shape, generator=rng).to(torch.bfloat16)
                    data[:, step * 1024 + 31] = 18
                    signs = torch.tensor([1 if (row+step)%2 else -1 for row in range(rows)])
                    data[:, 0] = signs * 18
                    x.copy_(data)
                    reference_norm = torch.nn.functional.rms_norm(x, (hidden,), norm.weight, norm.eps)
                    expected = LogitsProcessor._get_logits(proc, reference_norm, head, meta, use_logits_buffer=False)
                    with decode_tile_scope():
                        candidate_norm = try_welm_final_norm(norm, x)
                    exact_head = torch.matmul(candidate_norm, weight.T).float()
                    target.fill_(float("nan"))
                    if retained is not None and retained.backend is not None:
                        got = retained.backend.replay(rows, meta)
                    else:
                        got = plan.replay()
                    self.assertEqual(got.data_ptr(), out.data_ptr())
                    self.assertTrue(torch.isnan(target[0]).all().item())
                    self.assertTrue(torch.isfinite(got).all().item())
                    # Projection and cast run after a stored BF16 RMS result. Check
                    # all vocabulary values against the original matmul on that input.
                    torch.testing.assert_close(got, exact_head, atol=0, rtol=0)
                    winner = torch.where(signs > 0, 0, width-1)
                    self.assertTrue(torch.equal(got.argmax(-1).cpu(), winner))
                    self.assertTrue(torch.equal(expected.argmax(-1).cpu(), winner))
                    # The two RMS implementations may round a near-boundary BF16 differently.
                    torch.testing.assert_close(got, expected, atol=1/16, rtol=1/64)
                if retained is None:
                    for step in range(2):
                        replay(step)
                else:
                    retained.replays[rows] = replay

    def _check_retained(self, production_backend):
        # One fixed full-vocabulary weight is used for every live graph.
        with ExitStack() as owner:
            backend = None
            if production_backend:
                try:
                    import torch
                    import torch_npu
                except ModuleNotFoundError as error:
                    if error.name not in {"torch", "torch_npu"}:
                        raise
                    self.skipTest(str(error))
                if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
                    self.skipTest("requires an Ascend A5")
                from sglang.srt.hardware_backend.npu.graph_runner.welm_native_backend import WelmNativeBackend
                from sglang.srt.model_executor.forward_batch_info import ForwardMode
                args = SimpleNamespace(tp_size=1, dp_size=1, ep_size=1, pp_size=1)
                model = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(
                    architectures=["WeLMV4MoeForCausalLM"])), server_args=args, tp_group=SimpleNamespace(barrier=lambda: None),
                    spec_algorithm=SimpleNamespace(is_none=lambda: True), is_draft_worker=False)
                runner = SimpleNamespace(model_runner=model, device_module=torch.npu, capture_forward_mode=ForwardMode.DECODE)
                backend = WelmNativeBackend(runner)
                backend.directory = Path(owner.enter_context(tempfile.TemporaryDirectory()))
                owner.callback(backend.cleanup)
            kept = SimpleNamespace(owner=owner, shared=None, replays={}, backend=backend)
            for rows in (8, 4, 2, 1):
                self._check_vocab(rows, retained=kept)
            import torch
            with torch.inference_mode():
                for step, order in enumerate(((1, 8, 2, 4), (4, 2, 8, 1))):
                    for rows in order:
                        with self.subTest(phase="replay", step=step, rows=rows):
                            kept.replays[rows](step)

    def test_buckets_retained(self):
        self._check_retained(production_backend=False)

    def test_backend_buckets_retained(self):
        self._check_retained(production_backend=True)

    def test_bs1(self):
        self._check_vocab(1)

    def test_bs2(self):
        self._check_vocab(2)

    def test_bs4(self):
        self._check_vocab(4)

    def test_bs8(self):
        self._check_vocab(8)


class LMHeadShardDeviceTest(unittest.TestCase):
    def _simulate_ranks(self, tp_size):
        try:
            import torch
            import torch_npu
        except ModuleNotFoundError as error:
            if error.name not in {"torch", "torch_npu"}:
                raise
            self.skipTest(str(error))
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires an Ascend A5")
        from megakernel._fixed import capture_fixed
        from sglang.srt.hardware_backend.npu.graph_runner.welm_lm_head import try_welm_bf16_projection
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import decode_tile_scope

        width, hidden = 155648 // tp_size, 2048
        rng = torch.Generator().manual_seed(447 + tp_size)
        with torch.inference_mode(), ExitStack() as stack:
            weight = torch.empty(width * hidden, dtype=torch.bfloat16, device="npu").view(width, hidden)
            for start in range(0, width, 2048):
                stop = min(width, start + 2048)
                part = torch.randint(-3, 4, (stop-start, hidden), generator=rng, dtype=torch.int16)
                weight[start:stop].copy_(part.to(torch.bfloat16).mul_(1 / 32))
            inputs, plans, outputs, references = {}, {}, {}, {}
            for rows in (8, 4, 2, 1):
                x = torch.randint(-12, 13, (rows, hidden), generator=rng).to(torch.bfloat16).npu()
                x[:, 0] = 8
                inputs[rows] = x
                build = stack.enter_context(tempfile.TemporaryDirectory())
                with patch.dict("os.environ", MK_NATIVE_TILE_DEPENDENCIES="1"), decode_tile_scope():
                    plan = capture_fixed(lambda x=x: try_welm_bf16_projection(x, weight, tp_size),
                                         build, tile_chains=True)
                stack.callback(plan.reset)
                plans[rows] = plan
                self.assertEqual(plan.host_boundary_count, 0, plan.execution_summary)
                self.assertEqual(len(plan.plans), 1)
                scope, = plan.plans
                self.assertEqual(tuple(scope.kernel_names), ("_welm_head_fanout", "_welm_head_projection"))
                if rows > 1:
                    self.assertEqual(scope.tile_dependency_ratios[0, 1], (1, 16 // tp_size))
                # Local projection returns BF16. The original processor gathers before converting.
                outputs[rows], references[rows] = [], []
            for rank in range(tp_size):
                weight[0, 0] = 2 * rank
                weight[-1, 0] = -2 * rank
                for rows, plan in plans.items():
                    got = plan.replay().cpu()
                    self.assertEqual(got.dtype, torch.bfloat16)
                    self.assertEqual(tuple(got.shape), (rows, width))
                    outputs[rows].append(got)
                    references[rows].append(torch.matmul(inputs[rows], weight.T).cpu())
            for rows in plans:
                got = torch.cat(outputs[rows], dim=-1).float()
                expected = torch.cat(references[rows], dim=-1).float()
                self.assertEqual(tuple(got.shape), (rows, 155648))
                self.assertTrue(torch.isfinite(got).all())
                self.assertFalse(torch.equal(outputs[rows][0][:, 0], outputs[rows][-1][:, 0]))
                torch.testing.assert_close(got, expected, atol=0, rtol=0)

    def test_simulated_tp4(self):
        self._simulate_ranks(4)

    def test_simulated_tp8(self):
        self._simulate_ranks(8)


if __name__ == "__main__":
    unittest.main()
