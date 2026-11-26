"""Host lifecycle tests for fixed-buffer WeLM buckets without CANN or torch."""

from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt"
PACKAGE = "sglang.srt.hardware_backend.npu.graph_runner"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WelmNativeBackendTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        directory = tmp.name
        self.addCleanup(tmp.cleanup)
        self.events = []
        self.source = [0]
        self.plan = Mock(
            operator_count=3, scope_count=2, fused_operator_count=3,
            host_boundary_count=0,
            plans=[
                SimpleNamespace(tile_dependencies=[(0, 1)], kernel_names=("k_norm", "rope"),
                                dependencies=[[], [0]], tile_dependency_ratios={}),
                SimpleNamespace(tile_dependencies=[], kernel_names=("projection",), dependencies=[[]], tile_dependency_ratios={}),
            ],
        )
        self.plan.replay.side_effect = lambda: self.source[0]
        self.capture = Mock(return_value=self.plan)
        self.retain_if_failed = Mock(return_value=False)
        modules = {
            "torch": SimpleNamespace(Tensor=object, bfloat16="bf16"),
            "megakernel._failure": SimpleNamespace(retain_if_failed=self.retain_if_failed),
            "megakernel._runtime": SimpleNamespace(_HOME=directory),
            "megakernel._launch_capture": SimpleNamespace(_library=lambda: None),
            "megakernel._fixed": SimpleNamespace(
                capture_fixed=self.capture, prepare_static=lambda fn, _: fn()
            ),
            "megakernel.triton": SimpleNamespace(prepare_bindings=nullcontext),
            f"{PACKAGE}.welm_tile_layouts": SimpleNamespace(
                install_welm_tile_layouts=lambda: self.events.append("install"),
                decode_tile_scope=lambda **kwargs: patch.dict({}), refresh_welm_moe_tiles=lambda: None,
            ),
        }
        mocks = patch.dict(sys.modules, modules)
        mocks.start()
        self.addCleanup(mocks.stop)
        base_name = "sglang.srt.model_executor.runner_backend.base_cuda_graph_backend"
        sys.modules[base_name] = load(
            base_name, ROOT / "model_executor/runner_backend/base_cuda_graph_backend.py"
        )
        self.backend_cls = load(
            f"{PACKAGE}.welm_native_backend",
            ROOT / "hardware_backend/npu/graph_runner/welm_native_backend.py",
        ).WelmNativeBackend

    def runner(self, arch):
        config = SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch]), dtype="bf16", quantization=None)
        return SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=config, tp_group=Mock(), is_draft_worker=False,
                spec_algorithm=SimpleNamespace(is_none=lambda: True),
                server_args=SimpleNamespace(tp_size=1, dp_size=1, ep_size=1, pp_size=1),
            ),
            capture_forward_mode=SimpleNamespace(is_decode=lambda: True),
            device_module=SimpleNamespace(synchronize=lambda: None, get_device_name=lambda: "Ascend950DT"),
        )

    def test_capture_replay_replacement_and_failure(self):
        backend = self.backend_cls(self.runner("WeLMV4MoeForCausalLM"))
        self.capture.side_effect = lambda fn, directory, prepared, tile_chains: (self.assertTrue(tile_chains), fn(), self.plan)[2]
        run = lambda: self.events.append("forward")
        hook = lambda: self.events.append("reset")
        with self.assertLogs(f"{PACKAGE}.welm_native_backend", level="DEBUG") as logs:
            with backend.capture_session(None):
                backend.capture_one(8, run, post_warmup_hook=hook)
        self.assertIn("1 tile edges", logs.output[0])
        self.assertIn("8 scope 0 tile edges [('k_norm', 'rope')]", logs.output[1])
        self.assertEqual(self.events, ["install"] + ["forward", "reset"] * 3 + ["forward"])
        self.source[0] = 9
        self.assertEqual(backend.replay(8, None), 9)

        self.capture.side_effect = ValueError("unverified ABI")
        for key in (8, 16):
            with self.assertRaisesRegex(ValueError, "unverified ABI"):
                backend.capture_one(key, run)
        self.assertTrue(backend.can_run(None, 8))
        self.assertFalse(backend.can_run(None, 16))
        self.plan.reset.assert_not_called()
        backend.cleanup()
        self.plan.reset.assert_called_once_with()
        self.assertFalse(backend.can_run(None, 8))

    def test_draft_update_and_replay(self):
        backend = self.backend_cls(self.runner("WeLMV4MoeForCausalLMNextN"))
        payload = [{"actual_seq_lengths_kv": [3, 4]}]
        backend.capture_one(2, lambda: None)
        backend.replay_with_input_update(2, None, cpu_update_input=payload)
        self.plan.update.assert_called_once_with(cpu_update_input=payload)
        self.plan.replay.assert_called_once_with()

    def test_worker_quarantine_retains_pending_and_previous_buckets(self):
        backend = self.backend_cls(self.runner("WeLMV4MoeForCausalLM"))
        draft = self.backend_cls(self.runner("WeLMV4MoeForCausalLMNextN"))
        backend.capture_one(8, lambda: None)
        draft.capture_one(2, lambda: None)
        inputs = object()
        run = Mock()

        def fail_submission(*args, **kwargs):
            self.retain_if_failed.return_value = True
            raise RuntimeError("submission failed")

        self.capture.side_effect = fail_submission
        with self.assertRaisesRegex(RuntimeError, "submission failed"):
            backend.capture_one(16, run, inputs)
        self.retain_if_failed.assert_called_with(backend, run, inputs)
        for operation in (
            lambda: backend.can_run(None, 8),
            lambda: backend.replay(8, None),
            lambda: draft.replay_with_input_update(2, None, cpu_update_input=[]),
            lambda: draft.capture_one(2, run),
        ):
            with self.assertRaisesRegex(RuntimeError, "restart the worker"):
                operation()
        backend.cleanup()
        draft.cleanup()
        self.plan.replay.assert_not_called()
        self.plan.update.assert_not_called()
        self.plan.reset.assert_not_called()

    def test_single_pk_requires_supported_capture_and_a_single_decode_device(self):
        module = sys.modules["megakernel._fixed"]
        test = self
        class Key:
            size = 1

        def capable(fn, directory, *, prepared, tile_chains, retain_intermediates=False):
            policy.append(retain_intermediates)
            return test.plan

        policy = []
        module.capture_fixed = capable
        for variant in ("single", "bs2", "bs4", "bs8", "bs3", "bs16", "tp", "dp", "ep", "pp", "draft", "verify", "mtp", "a3", "tile", "default"):
            with self.subTest(variant=variant):
                runner = self.runner("WeLMV4MoeForCausalLM")
                key = Key()
                if variant.startswith("bs"):
                    key.size = int(variant[2:])
                if variant in ("tp", "dp", "ep", "pp"):
                    setattr(runner.model_runner.server_args, variant + "_size", 2)
                if variant == "draft":
                    runner.model_runner.is_draft_worker = True
                if variant == "verify":
                    runner.capture_forward_mode = SimpleNamespace(is_decode=lambda: False)
                if variant == "mtp":
                    runner.model_runner.spec_algorithm = SimpleNamespace(is_none=lambda: False)
                if variant == "a3":
                    runner.device_module.get_device_name = lambda: "Ascend910_9392"
                with patch.dict(os.environ, {"MK_NATIVE_TILE_DEPENDENCIES": "0"}):
                    if variant == "tile":
                        os.environ["MK_NATIVE_TILE_DEPENDENCIES"] = "1"
                    elif variant == "default":
                        os.environ.pop("MK_NATIVE_TILE_DEPENDENCIES")
                    self.backend_cls(runner).capture_one(key, lambda: None)
                self.assertEqual(policy[-1], variant in ("single", "bs2", "bs4", "bs8"))
        # An older MK backend lacks the option and still takes the original path.
        module.capture_fixed = lambda fn, directory, *, prepared, tile_chains: self.plan
        with patch.dict(os.environ, {"MK_NATIVE_TILE_DEPENDENCIES": "0"}):
            self.backend_cls(self.runner("WeLMV4MoeForCausalLM")).capture_one(Key(), lambda: None)

    def test_native_ops_preserve_dispatch_in_both_plans(self):
        from collections import namedtuple

        key = namedtuple("Key", "size")(1)
        # Exercise the real scope used by forward_fn during warmup and capture.
        layouts = load(f"{PACKAGE}.welm_tile_layouts",
                       ROOT / "hardware_backend/npu/graph_runner/welm_tile_layouts.py")
        # load() executes without installing the module in sys.modules.
        scope_globals = self.backend_cls.capture_one.__globals__
        source = self.plan.plans[0]
        source.kernel_names = ("q", "static_kernel_FlashAttn_example", "gate",
                               "static_kernel_FlashAttnMetadata_example")
        source.dependencies = [[], [0], [1], [2]]
        source.tile_dependencies = [(0, 1), (1, 2), (2, 3)]
        def capture(fn, directory, *, prepared, tile_chains, retain_intermediates=False):
            fn()
            self.assertTrue(retain_intermediates)
            return self.plan

        def check_dispatch():
            self.assertTrue(layouts.welm_original_kernels_active())
            self.assertTrue(layouts.welm_native_gate_first_active())
            self.assertFalse(layouts.welm_decode_tiles_active())
            self.assertFalse(layouts.welm_single_pk_decode_active())
            self.assertFalse(layouts.welm_model_row_tiles_active())
            self.assertFalse(layouts.welm_page_gate_tiles_active())
            self.assertIsNone(layouts.welm_decode_tile_workspace())

        settings = {"SGLANG_NPU_WELMV4_MK_NATIVE_OPS_ONLY": "1",
                    "WELM_NPU_USE_FLASH_ATTN": "1",
                    "SGLANG_NPU_WELMV4_MK_NATIVE_FLASH_GATE_FIRST": "1"}
        utils = SimpleNamespace(get_bool_env_var=lambda name, *args: os.getenv(name) == "1")
        with patch.dict(sys.modules, {"sglang.srt.utils": utils}), \
                patch.dict(scope_globals, decode_tile_scope=layouts.decode_tile_scope), \
                patch.object(sys.modules["megakernel._fixed"], "capture_fixed", capture):
            for fine in ("0", "1"):
                with self.subTest(fine=fine), patch.dict(os.environ, settings | {"MK_NATIVE_TILE_DEPENDENCIES": fine}):
                    with self.assertLogs(f"{PACKAGE}.welm_native_backend", level="INFO") as logs:
                        self.backend_cls(self.runner("WeLMV4MoeForCausalLM")).capture_one(key, check_dispatch)
                    self.assertTrue(any("1 fused FlashAttn, 1 incoming and 1 outgoing Flash tile edges" in entry
                                        for entry in logs.output), logs.output)
                    self.assertFalse(layouts.welm_original_kernels_active())
                    self.assertFalse(layouts.welm_native_gate_first_active())

    def test_candidate_regions_exclude_quantized_and_fp16_dtypes(self):
        for dtype, quant, expected in (("bf16", None, True), ("fp16", None, False), ("bf16", "fp8", False)):
            with self.subTest(dtype=dtype, quant=quant):
                runner = self.runner("WeLMV4MoeForCausalLM")
                runner.model_runner.model_config.dtype = dtype
                runner.model_runner.model_config.quantization = quant
                self.assertEqual(self.backend_cls(runner)._native_region_worker, expected)

    def test_reject_unrelated_model(self):
        with self.assertRaisesRegex(ValueError, "WeLMV4"):
            self.backend_cls(self.runner("OtherForCausalLM"))
        self.assertFalse(self.events)

    def test_npu_decode_backend_selection(self):
        base = "sglang.srt.model_executor"
        classic = Mock()
        deps = {
            f"{base}.cuda_graph_config": SimpleNamespace(Backend=SimpleNamespace(FULL="full")),
            f"{PACKAGE}.welm_native_backend": SimpleNamespace(WelmNativeBackend=self.backend_cls),
            f"{PACKAGE}.npu_cudagraph_backend": SimpleNamespace(NPUCudaGraphBackend=classic),
        }
        for module, name in (
            ("breakable_cuda_graph_backend", "BreakableCudaGraphBackend"),
            ("full_cuda_graph_backend", "FullCudaGraphBackend"),
            ("tc_piecewise_cuda_graph_backend", "TcPiecewiseCudaGraphBackend"),
        ):
            deps[f"{base}.runner_backend.{module}"] = SimpleNamespace(**{name: object})
        with patch.dict(sys.modules, deps):
            resolver = load("runner_resolver", ROOT / "model_executor/runner_backend/utils.py")
            runner = self.runner("WeLMV4MoeForCausalLM")
            runner.model_runner.device = "npu"
            args = SimpleNamespace(
                cuda_graph_config=None, enable_memory_saver=False,
                tp_size=1, dp_size=1, ep_size=1, pp_size=1,
                model_execution_backend="megakernel-native",
            )
            runner.model_runner.server_args = args
            self.assertIsInstance(resolver.resolve_decode_backend(runner), self.backend_cls)
            classic.assert_not_called()
            args.model_execution_backend = "native"
            self.assertIs(resolver.resolve_decode_backend(runner), classic.return_value)
            classic.assert_called_once_with(runner, enable_memory_saver=False)


if __name__ == "__main__":
    unittest.main()
