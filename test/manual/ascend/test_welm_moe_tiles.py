"""CPU checks for native WeLM expert, routing and logits tile declarations."""

import ast
import importlib.util
import sys
import unittest
from hashlib import sha256
from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "python/sglang/srt/hardware_backend/npu/graph_runner/welm_moe_tiles.py"
spec = importlib.util.spec_from_file_location("welm_moe_tiles", PATH)
tiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiles)


class Tensor:
    def __init__(self, shape, size=4):
        self.shape, self.size = shape, size

    def numel(self):
        return prod(self.shape)

    def element_size(self):
        return self.size

    def is_contiguous(self):
        return True


class MoeTilesTest(unittest.TestCase):
    def test_l1_programs_match_source_partition_and_tail(self):
        x, out = Tensor((15, 8), 2), Tensor((15, 8))
        layout = tiles._routing_l1_layout(x, out, 15, 8, 4)
        self.assertEqual((layout["count"], layout["block"]), (15, 4))
        self.assertEqual(layout["reads"], [(x, 16)])
        self.assertEqual(layout["writes"], [(out, 32)])
        self.assertIsNone(tiles._routing_l1_layout(x, out, 15, 8, 16))
        self.assertIsNone(tiles._routing_l1_layout(x, out, 15, 7, 4))

    def test_expert_counts_change_only_the_active_prefix(self):
        x, groups = Tensor((8, 64), 2), Tensor((3,), 8)
        out, scale = Tensor((8, 32), 1), Tensor((8,))
        args = dict(TOTAL_COLS=64, HALF_COLS=32, NUM_EXPERTS=3,
                    NUM_CORES=8, GROUP_LIST_TYPE=1, SCALE=True)
        layout = tiles._expert_swiglu_layout(x, groups, out, scale, **args)
        for counts in ((0, 0, 0), (0, 2, 1), (3, 0, 5)):
            total = sum(counts)
            block = (total - 1) // args["NUM_CORES"] + 1
            for pid in range(args["NUM_CORES"]):
                active = range(pid * block, min((pid + 1) * block, total))
                self.assertTrue(all(row == pid for row in active))
        self.assertEqual(layout["reads"], [(x, 128)])
        self.assertEqual(layout["writes"], [(out, 32), (scale, 4)])
        self.assertEqual(layout["shared_reads"], [groups])
        bf16 = tiles._expert_swiglu_layout(x, groups, out, scale, **(args | {"SCALE": False}))
        self.assertEqual(bf16["writes"], [(out, 32)])
        self.assertEqual(bf16["shared_reads"], [groups, scale])
        self.assertIsNone(tiles._expert_swiglu_layout(
            Tensor((16, 64)), groups, Tensor((16, 32)), Tensor((16,)), **args
        ))
        self.assertIsNone(tiles._expert_swiglu_layout(
            x, groups, out, scale, **(args | {"GROUP_LIST_TYPE": 0})
        ))

    def test_only_supported_small_row_launches_shrink(self):
        kernel = MagicMock()
        x = Tensor((40, 256))
        for active, group_type, expected in ((True, 1, 40), (False, 1, 64), (True, 0, 64)):
            with self.subTest(active=active, group_type=group_type):
                expert = tiles._ActiveRowGrid(kernel, tiles._expert_swiglu_layout, lambda: active)
                expert[(64,)](x, None, None, None, NUM_CORES=64, GROUP_LIST_TYPE=group_type)
                kernel.__getitem__.assert_called_with((expected,))
                kernel.__getitem__.return_value.assert_called_with(
                    x, None, None, None, NUM_CORES=expected, GROUP_LIST_TYPE=group_type
                )
        route = tiles._ActiveRowGrid(kernel, tiles._routing_l1_layout, lambda: True)
        route[(64,)](x, None, 40, 256, 64)
        kernel.__getitem__.assert_called_with((40,))
        kernel.__getitem__.return_value.assert_called_with(x, None, 40, 256, 40)
        route[(16,)](x, None, 40, 256, 16)
        kernel.__getitem__.assert_called_with((16,))

    def test_dp_logits_copy_retains_the_entire_indirect_side(self):
        local, arena = Tensor((3, 128)), Tensor((16, 128))
        offset, size = Tensor((1,)), Tensor(())
        scatter = tiles._dp_copy_layout(local, arena, offset, size, True, 128, 256)
        self.assertEqual((scatter["count"], scatter["block"]), (384, 256))
        self.assertEqual(scatter["writes"], [(local, 4)])
        self.assertEqual(scatter["shared_reads"], [arena, offset, size])
        gather = tiles._dp_copy_layout(arena, local, offset, size, False, 128, 256)
        self.assertEqual(gather["reads"], [(local, 4)])
        self.assertEqual(gather["writes"], [])
        self.assertEqual(gather["whole_writes"], [arena])
        self.assertIsNone(tiles._dp_copy_layout(arena, local, offset, size, True, 128, 256))
        self.assertIsNone(tiles._dp_copy_layout(local, arena, offset, size, True, 64, 256))

    def test_installs_loaded_kernels_once_and_rejects_changed_source(self):
        module = ModuleType("test_welm_loaded_routing_kernel")
        original, wrapper = SimpleNamespace(src="def tested(): pass"), object()
        module.kernel = original
        digest = sha256(original.src.encode()).hexdigest()
        native = Mock(return_value=Mock(return_value=wrapper))
        with patch.object(tiles, "_KERNELS", ((module.__name__, "kernel", None, digest),)):
            tiles.install_moe_tiles(native)
            native.assert_not_called()
            with patch.dict(sys.modules, {module.__name__: module}):
                tiles.install_moe_tiles(native)
                tiles.install_moe_tiles(native)
                native.return_value.assert_called_once_with(original)
                self.assertIs(module.kernel, wrapper)
        changed = ModuleType(module.__name__)
        changed.kernel = SimpleNamespace(src="def tested(): return 0")
        with self.assertLogs(tiles.logger, level="WARNING"):
            tiles._install(changed, "kernel", None, digest, native)
        self.assertEqual(changed.kernel.src, "def tested(): return 0")

    def test_external_fingerprints_match_local_allgather_deepep_sources(self):
        checkout = ROOT.parent / "sgl-kernel-npu/python/sgl_kernel_npu"
        if not checkout.is_dir():
            self.skipTest("allgather-deepep source checkout is absent")
        for module, name, _, digest in tiles._KERNELS:
            if digest is None:
                continue
            source = (checkout / (module.replace(".", "/") + ".py")).read_text()
            fn = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.FunctionDef) and node.name == name)
            body = ast.get_source_segment(source, fn)
            self.assertEqual(sha256(body.strip().encode()).hexdigest(), digest)


class MoeTilesDeviceTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            import torch_npu
        except ImportError as exc:
            self.skipTest(f"A5 regression requires torch_npu: {exc}")
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires an Ascend A5 NPU")
        self.torch = torch

    def test_expert_rows_match_eager_when_dispatch_counts_change(self):
        self._check_expert_rows()

    def test_expert_capacity_below_core_count(self):
        self._check_expert_rows(small=True)

    def test_ep4_bf16_local_expert_capacity(self):
        # WeLM has 512 routed experts, so EP4 owns 128 per rank.
        self._check_expert_rows(small=True, experts=128)

    def _check_expert_rows(self, small=False, experts=2):
        from megakernel._fixed import capture_fixed
        from sgl_kernel_npu.activation.swiglu_quant import swiglu_quant
        from sgl_kernel_npu.norm.l1_norm import l1_norm
        from sgl_kernel_npu.utils.triton_utils import get_device_properties
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import (
            decode_tile_scope, refresh_welm_moe_tiles,
        )

        torch = self.torch
        _, cores = get_device_properties()
        rows = min(40, cores - 1) if small else cores
        width = 256
        x = torch.empty((rows, width * 2), device="npu", dtype=torch.bfloat16)
        x.fill_(1)
        groups = torch.zeros(experts, device="npu", dtype=torch.int64)
        groups[0] = rows

        def forward():
            activation, _ = swiglu_quant(x, groups, 1, need_quant=False)
            return activation, l1_norm(activation)

        refresh_welm_moe_tiles()
        with torch.inference_mode(), TemporaryDirectory() as build:
            with decode_tile_scope():
                plan = capture_fixed(forward, build)
            try:
                self.assertEqual(plan.scope_count, 1, plan.execution_summary)
                self.assertEqual(plan.host_boundary_count, 0, plan.execution_summary)
                self.assertEqual(plan.plans[0].tile_dependencies, [(0, 1)])
                seed = torch.Generator().manual_seed(13)
                for valid in (rows, rows // 2, 0):
                    new_x = 1 + torch.rand(x.shape, generator=seed)
                    x.copy_(new_x.to(torch.bfloat16))
                    groups.zero_()
                    groups[-1] = rows
                    # Ordinary eager globals compute all rows of the reference.
                    eager_activation, eager_norm = forward()
                    expected_activation, expected_norm = eager_activation.cpu(), eager_norm.cpu()
                    expected_activation[valid:] = 7
                    expected_norm[valid:] = 1 / width
                    plan.outputs[0].fill_(7)
                    groups.zero_()
                    groups[0] = valid // 2
                    groups[-1] = valid - valid // 2
                    activation, norm = plan.replay()
                    torch.testing.assert_close(activation.cpu(), expected_activation, atol=0, rtol=0)
                    torch.testing.assert_close(norm.cpu(), expected_norm, atol=0, rtol=0)
            finally:
                plan.reset()

    def test_dp_logits_copy_replay_changes_offsets_and_valid_rows(self):
        from megakernel._fixed import capture_fixed
        from sglang.kernels.ops.memory.memcpy_triton import memcpy_triton
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import (
            decode_tile_scope, refresh_welm_moe_tiles,
        )

        torch = self.torch
        cols = 8192
        cpu = torch.arange(8 * cols, dtype=torch.int32).reshape(8, cols)
        inp = cpu.to("npu")
        local = torch.empty((3, cols), device="npu", dtype=torch.int32)
        gathered = torch.empty_like(inp)
        start, dest, size = [
            torch.tensor(value, device="npu", dtype=torch.int32) for value in (1, 2, 3)
        ]

        def forward():
            # These are the unmodified DP scatter and gather launcher signatures.
            memcpy_triton(local, inp, 0, start, size, True)
            memcpy_triton(gathered, local, 0, dest, size, False)
            return local, gathered

        refresh_welm_moe_tiles()
        with torch.inference_mode(), TemporaryDirectory() as build, decode_tile_scope():
            plan = capture_fixed(forward, build)
            try:
                self.assertEqual(plan.scope_count, 1, plan.execution_summary)
                self.assertEqual(plan.host_boundary_count, 0, plan.execution_summary)
                self.assertEqual(plan.plans[0].tile_dependencies, [(0, 1)])
                for step, (src_row, dst_row, n) in enumerate(((4, 2, 3), (1, 5, 2), (7, 1, 0))):
                    new_input = cpu + step
                    inp.copy_(new_input)
                    local.fill_(-7)
                    gathered.fill_(-9)
                    start.fill_(src_row)
                    dest.fill_(dst_row)
                    size.fill_(n)
                    local_out, gathered_out = plan.replay()
                    expected_local = torch.full((3, cols), -7, dtype=torch.int32)
                    expected_local[:n].copy_(new_input[src_row:src_row + n])
                    expected_gather = torch.full_like(cpu, -9)
                    expected_gather[dst_row:dst_row + n].copy_(expected_local[:n])
                    self.assertTrue(torch.equal(local_out.cpu(), expected_local))
                    self.assertTrue(torch.equal(gathered_out.cpu(), expected_gather))
            finally:
                plan.reset()


if __name__ == "__main__":
    unittest.main()
