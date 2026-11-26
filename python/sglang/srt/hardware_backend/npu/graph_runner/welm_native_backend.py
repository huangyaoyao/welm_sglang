"""Fixed-buffer WeLM execution using MegaKernel's native capture and replay."""

from inspect import signature
import logging
import os
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path

from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)

from .welm_tile_layouts import decode_tile_scope, refresh_welm_moe_tiles

logger = logging.getLogger(__name__)


class WelmNativeBackend(BaseCudaGraphBackend):
    def __init__(self, runner):
        model_runner = runner.model_runner
        architectures = model_runner.model_config.hf_config.architectures or ()
        if not {"WeLMV4MoeForCausalLM", "WeLMV4MoeForCausalLMNextN"}.intersection(
            architectures
        ):
            raise ValueError("megakernel-native requires a WeLMV4 target or draft")

        import torch
        from megakernel._failure import retain_if_failed
        from megakernel._launch_capture import _library
        from megakernel._runtime import _HOME
        from .welm_tile_layouts import install_welm_tile_layouts

        _library()  # Verify the launch interposer is loaded before CANN.
        install_welm_tile_layouts()
        self.device = runner.device_module
        self._retain_if_failed = retain_if_failed
        args = model_runner.server_args
        mode = getattr(runner, "capture_forward_mode", None)
        self._single_pk_worker = (
            "WeLMV4MoeForCausalLM" in architectures
            and mode is not None
            and mode.is_decode()
            and not model_runner.is_draft_worker
            and model_runner.spec_algorithm.is_none()
            and (args.tp_size, args.dp_size, args.ep_size, args.pp_size) == (1, 1, 1, 1)
            and self.device.get_device_name().startswith("Ascend950")
        )
        config = model_runner.model_config
        self._native_region_worker = (
            self._single_pk_worker and getattr(config, "quantization", None) is None
            and getattr(config, "dtype", None) is not None and config.dtype == torch.bfloat16
        )
        self.tp_group = model_runner.tp_group
        # Descriptor exports and compiler diagnostics are local to a worker.
        self.directory = Path(_HOME) / "sglang_native_compile" / str(os.getpid())
        self.directory /= type(runner).__name__
        self._plans = {}

    def capture_session(self, stream):
        # The runner establishes its device stream before calling the backend.
        return nullcontext()

    def replay_session(self):
        return nullcontext()

    def _check_worker(self):
        if self._retain_if_failed(self):
            raise RuntimeError(
                "MK quarantined a failed device submission. Please restart the worker."
            )

    def capture_one(
        self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None
    ):
        from megakernel._fixed import capture_fixed, prepare_static
        from megakernel.triton import prepare_bindings

        self._check_worker()
        key = sha256(repr(shape_key).encode()).hexdigest()[:16]
        directory = self.directory / key
        directory.mkdir(exist_ok=True, parents=True)
        native_ops_only = (
            self._single_pk_worker
            and os.environ.get("SGLANG_NPU_WELMV4_MK_NATIVE_OPS_ONLY") == "1"
        )
        retain_intermediates = (
            self._single_pk_worker
            and getattr(shape_key, "size", None) in (1, 2, 4, 8)
            and (native_ops_only or os.environ.get("MK_NATIVE_TILE_DEPENDENCIES") == "0")
            and "retain_intermediates" in signature(capture_fixed).parameters
        )
        with decode_tile_scope(
            single_pk=retain_intermediates and not native_ops_only,
            native_ops_only=native_ops_only,
            batch=getattr(shape_key, "size", None) if self._native_region_worker else None,
        ):
            for _ in range(2):
                self.device.synchronize()
                self.tp_group.barrier()
                forward_fn()
                if post_warmup_hook:
                    post_warmup_hook()

            # Preparation executes a dummy forward. Restore the attention metadata
            # before recording the actual plan, as after each ordinary warmup.
            self.device.synchronize()
            self.tp_group.barrier()
            refresh_welm_moe_tiles()
            with prepare_bindings():
                prepare_static(forward_fn, directory)
            if post_warmup_hook:
                post_warmup_hook()
            self.device.synchronize()
            self.tp_group.barrier()
            try:
                opts = {"retain_intermediates": True} if retain_intermediates else {}
                plan = capture_fixed(forward_fn, directory, prepared=True, tile_chains=True, **opts)
            except BaseException:
                # These caller-owned buffers may still be in flight before a plan
                # exists. MK retains them only for a quarantined device submission.
                self._retain_if_failed(self, forward_fn, capture_inputs)
                raise
        previous = self._plans.get(shape_key)
        self._plans[shape_key] = (plan, capture_inputs)
        if previous is not None:
            previous[0].reset()
        logger.info(
            "WeLM MK capture %s: %d kernels, %d scopes, %d fused kernels, "
            "%d Torch boundaries, %d tile edges",
            shape_key,
            plan.operator_count,
            plan.scope_count,
            plan.fused_operator_count,
            plan.host_boundary_count,
            sum(len(scope.tile_dependencies) for scope in plan.plans),
        )
        if native_ops_only:
            flashes = incoming = outgoing = 0
            for scope in plan.plans:
                flash = {i for i, name in enumerate(scope.kernel_names)
                         if name.startswith("static_kernel_FlashAttn_")}
                flashes += len(flash)
                incoming += sum(dst in flash for _, dst in scope.tile_dependencies)
                outgoing += sum(src in flash for src, _ in scope.tile_dependencies)
            logger.info("WeLM native operators %s: %d fused FlashAttn, %d incoming and %d outgoing Flash tile edges",
                        shape_key, flashes, incoming, outgoing)
        if logger.isEnabledFor(logging.DEBUG):
            for index, scope in enumerate(plan.plans):
                if scope.tile_dependencies:
                    pairs = [
                        (scope.kernel_names[parent], scope.kernel_names[child])
                        for parent, child in scope.tile_dependencies
                    ]
                    logger.debug(
                        "WeLM MK capture %s scope %d tile edges %s", shape_key, index, pairs
                    )

                selected = set(scope.tile_dependencies)
                coarse = [
                    (scope.kernel_names[src], scope.kernel_names[dst])
                    for dst, parents in enumerate(scope.dependencies) for src in parents
                    if (src, dst) not in selected
                ]
                logger.debug(
                    "WeLM MK capture %s scope %d kernels %s coarse edges %s tile ratios %s",
                    shape_key, index, scope.kernel_names, coarse, scope.tile_dependency_ratios,
                )
                memory = getattr(scope, "memory_dependencies", None)
                if isinstance(memory, (tuple, list)) and len(memory) == len(scope.dependencies):
                    # Scope-local IDs identify repeated instances of the same kernel.
                    # A stream-only dependency also constrains legal refinements.
                    indexed = [
                        (src, dst, "tile" if (src, dst) in selected else "full",
                         "storage" if src in memory[dst] else "stream")
                        for dst, parents in enumerate(scope.dependencies) for src in parents
                    ]
                    logger.debug("WeLM MK capture %s scope %d indexed dependencies %s",
                                 shape_key, index, indexed)
            logger.debug("WeLM MK capture %s boundaries %s", shape_key, plan.execution_summary)

    def can_run(self, forward_batch, shape_key):
        self._check_worker()
        return shape_key in self._plans

    def replay(self, shape_key, static_forward_batch, **kwargs):
        self._check_worker()
        return self._plans[shape_key][0].replay()

    def replay_with_input_update(
        self, shape_key, seq_lens, attr_name=None, attr_type=None, cpu_update_input=None
    ):
        import torch

        self._check_worker()
        if cpu_update_input is None:
            if isinstance(attr_type, torch.Tensor):
                seq_lens = torch.as_tensor(seq_lens, dtype=torch.int32, device="cpu")
            cpu_update_input = [{attr_name: seq_lens}]
        plan, _ = self._plans[shape_key]
        plan.update(cpu_update_input=cpu_update_input)
        return plan.replay()

    def cleanup(self):
        if self._retain_if_failed(self):
            return
        for plan, _ in self._plans.values():
            plan.reset()
        self._plans.clear()
