"""Run the actual WeLM logits dispatch on a host without Torch/Ascend."""
import ast
import sys
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SOURCE = (Path(__file__).resolve().parents[3] / "python/sglang/srt/hardware_backend"
          / "npu/graph_runner/welm_lm_head.py")


class WelmLogitsDispatchTest(unittest.TestCase):
    def setUp(self):
        class Original:
            def _get_logits(self, *args):
                return args
            def _compute_lm_head(self, x, head, bias):
                return ("original-head", x)
        class Head:
            pass
        self.head = Head()
        self.head.weight, self.head.tp_size, self.head.quant_method = object(), 1, None
        self.active = Mock(return_value=True)
        self.quantized = Mock(return_value=False)
        self.compiling = Mock(return_value=False)
        self.try_logits = Mock(return_value=object())
        self.local = Mock(side_effect=lambda x, weight, tp: ("local-bf16", x, tp))
        self.local_context = ContextVar("test_local_head", default=None)
        self.buffer = object()
        self.decode = Mock(return_value=True)
        self.meta = SimpleNamespace(next_token_logits_buffer=self.buffer, forward_mode=SimpleNamespace(is_decode=self.decode))
        ns = dict(torch=SimpleNamespace(compiler=SimpleNamespace(is_compiling=self.compiling)),
                  LogitsProcessor=Original, ParallelLMHead=Head, welm_decode_tiles_active=self.active,
                  should_apply_lm_head_quant_method=self.quantized, try_welm_logits=self.try_logits,
                  _VOCAB=155648, _LOCAL_HEAD=self.local_context, try_welm_bf16_projection=self.local)
        mod = ast.parse(SOURCE.read_text())
        mod.body = [n for n in mod.body if isinstance(n, ast.ClassDef) and n.name == "WelmDecodeLogitsProcessor"]
        exec(compile(mod, str(SOURCE), "exec"), ns)
        self.proc = ns["WelmDecodeLogitsProcessor"]()
        self.proc.vocab_size = 155648
        self.proc.rl_on_policy_target = self.proc.logit_scale = self.proc.final_logit_softcapping = None
        self.proc.do_tensor_parallel_all_gather_dp_attn = self.proc.do_tensor_parallel_all_gather = False
        self.proc.use_fp32_lm_head = False

    def test_ordinary_decode_buffer_identity(self):
        x = object()
        self.assertIs(self.proc._get_logits(x, self.head, self.meta), self.try_logits.return_value)
        self.try_logits.assert_called_with(x, self.head.weight, self.buffer)
        self.proc._get_logits(x, self.head, self.meta, use_logits_buffer=False)
        self.try_logits.assert_called_with(x, self.head.weight, None)
        self.try_logits.return_value = None
        self.assertEqual(self.proc._get_logits(x, self.head, self.meta), (x, self.head, self.meta, None, True))

    def test_vocab_shards_keep_parent_communication_and_precision(self):
        x = object()
        # Execute the real SGLang parent orchestration, replacing only collective transport.
        parent_source = SOURCE.parents[3] / "layers/logits_processor.py"
        tree = ast.parse(parent_source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LogitsProcessor")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_logits")
        for arg in method.args.args:
            arg.annotation = None
        method.returns = None
        ns = {}
        exec(compile(ast.Module([method], []), str(parent_source), "exec"), ns)
        type(self.proc).__bases__[0]._get_logits = ns["_get_logits"]
        self.proc.use_attn_tp_group = False
        self.proc._gather_dp_attn_hidden_states = lambda x, meta: (("hidden-gather", x), x)
        self.proc._logits_gatherer = lambda y: ("vocab-gather", y)
        self.proc._scatter_dp_attn_logits = lambda y, x, meta: ("scatter", y)
        self.proc._copy_logits_to_buffer = lambda y, meta, use_buffer: ("copy", y, use_buffer)
        self.proc.do_tensor_parallel_all_gather = True
        self.proc.do_tensor_parallel_all_gather_dp_attn = True
        for tp in (4, 8):
            self.head.tp_size = tp
            got = self.proc._get_logits(x, self.head, self.meta)
            self.assertEqual(got, ("copy", ("scatter", ("vocab-gather", ("local-bf16", ("hidden-gather", x), tp))), True))
            self.assertIsNone(self.local_context.get())
            self.assertEqual(self.proc._compute_lm_head(x, self.head), ("original-head", x))
        self.local.side_effect = None
        self.local.return_value = None
        got = self.proc._get_logits(x, self.head, self.meta, use_logits_buffer=False)
        self.assertEqual(got, ("copy", ("scatter", ("vocab-gather", ("original-head", ("hidden-gather",x)))), False))
        self.local.side_effect = ValueError("projection error")
        with self.assertRaisesRegex(ValueError, "projection error"):
            self.proc._get_logits(x, self.head, self.meta)
        self.assertIsNone(self.local_context.get())
        self.try_logits.assert_not_called()

    def test_model_final_residual_rebuild_and_cast(self):
        model = SOURCE.parents[3] / "models/welmv4.py"
        tree = ast.parse(model.read_text())
        model_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen2MoeModel")
        forward = next(n for n in model_class.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        final = next(n for n in ast.walk(forward) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "hidden_states.shape[0] != 0"
                     and any(isinstance(v, ast.Name) and v.id == "can_rebuild_final_mlp" for v in ast.walk(n)))
        code = compile(ast.Module([final], []), str(model), "exec")
        class Trace:
            shape = (2, 2048)
            def __init__(self, sources):
                self.sources = sources
            def float(self):
                return Trace(("fp32", self.sources))
            def __add__(self, other):
                return Trace(("+", self.sources, other.sources))
            def to(self, dtype):
                return Trace((dtype, self.sources))
        for tp, dp, shared, decode, enabled, accepted in (
                (1, False, True, True, True, True), (1, False, False, True, True, True),
                (2, False, True, True, True, True), (1, True, True, True, True, True),
                (1, False, True, False, True, False), (1, False, True, True, False, False),
                (1, False, True, True, True, False)):
            with self.subTest(tp=tp, dp=dp, shared=shared, decode=decode, enabled=enabled, accepted=accepted):
                layer = SimpleNamespace(mlp=SimpleNamespace(tp_size=tp), final_mlp_experts_output=Trace("expert"),
                                        final_mlp_shared_output=Trace("shared") if shared else None)
                norm = SimpleNamespace(weight=SimpleNamespace(dtype="bf16", shape=(2048,)), eps=1e-6)
                fallback = Mock(return_value=object())
                optimized = Mock(return_value=object() if accepted else None)
                state = dict(self=SimpleNamespace(norm=norm, end_layer=1, layers=[layer], _mk_lm_head_tiles=enabled),
                             hidden_states=Trace("hidden"), residual=Trace("residual"), F=SimpleNamespace(rms_norm=fallback),
                             is_dp_attention_enabled=lambda: dp, torch=SimpleNamespace(compiler=SimpleNamespace(is_compiling=lambda:False)),
                             forward_batch=SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: decode)))
                module = "sglang.srt.hardware_backend.npu.graph_runner.welm_lm_head"
                with patch.dict(sys.modules, {module: SimpleNamespace(try_welm_final_norm=optimized)}):
                    exec(code, state)
                rebuild = tp == 1 and not dp
                before = ("+", ("fp32","expert" if rebuild else "hidden"), ("fp32","residual"))
                if rebuild and shared:
                    before = ("+", before, ("fp32","shared"))
                pre = state["pre_norm_hidden_states"]
                self.assertEqual(pre.sources, ("bf16", before))
                if decode and enabled:
                    optimized.assert_called_once_with(norm, pre)
                else:
                    optimized.assert_not_called()
                if decode and enabled and accepted:
                    fallback.assert_not_called()
                    self.assertIs(state["hidden_states"], optimized.return_value)
                else:
                    fallback.assert_called_once_with(pre, norm.weight.shape, norm.weight, eps=norm.eps)
                    self.assertIs(state["hidden_states"], fallback.return_value)

    def test_final_norm_and_head_configurations(self):
        self.assertTrue(self.proc.supports_welm_head_tiles())
        for field, value in (("use_fp32_lm_head", True), ("rl_on_policy_target", "bf16"),
                             ("logit_scale", 0.5), ("final_logit_softcapping", 4.0),
                             ("do_tensor_parallel_all_gather", True),
                             ("do_tensor_parallel_all_gather_dp_attn", True)):
            with self.subTest(field=field):
                old = getattr(self.proc, field)
                setattr(self.proc, field, value)
                self.assertFalse(self.proc.supports_welm_head_tiles())
                setattr(self.proc, field, old)

    def test_native_and_specialized_variants(self):
        x = object()
        for obj, attr, value in (
            (self.active, "return_value", False), (self.compiling, "return_value", True),
            (self.decode, "return_value", False), (self.quantized, "return_value", True),
            (self.head, "tp_size", 2), (self.proc, "vocab_size", 1024),
            (self.proc, "use_fp32_lm_head", True), (self.proc, "rl_on_policy_target", "target"),
            (self.proc, "logit_scale", 2), (self.proc, "final_logit_softcapping", 8),
            (self.proc, "do_tensor_parallel_all_gather", True),
            (self.proc, "do_tensor_parallel_all_gather_dp_attn", True),
        ):
            with self.subTest(attr=attr, value=value):
                before = getattr(obj, attr)
                setattr(obj, attr, value)
                self.assertEqual(self.proc._get_logits(x, self.head, self.meta),
                                 (x, self.head, self.meta, None, True))
                setattr(obj, attr, before)
        bias = object()
        self.assertEqual(self.proc._get_logits(x, self.head, self.meta, embedding_bias=bias),
                         (x, self.head, self.meta, bias, True))
        self.try_logits.assert_not_called()


if __name__ == "__main__":
    unittest.main()
