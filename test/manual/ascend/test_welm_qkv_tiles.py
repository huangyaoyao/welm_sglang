"""Capture the fixed-bucket BF16 projection and validate its exact Cube/Vector edges."""
import tempfile
import unittest
from unittest.mock import patch


class WelmQKVTileTest(unittest.TestCase):
    def test_changed_inputs_and_output_ownership(self):
        for rows, programs_per_row in ((1, 14), (2, 14), (4, 7), (8, 2)):
            with self.subTest(rows=rows):
                self._check_bucket(rows, programs_per_row)

    def _check_bucket(self, rows, programs_per_row):
        try:
            import torch
            import torch_npu
        except ImportError as exc:
            self.skipTest(str(exc))
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires an A5 NPU")

        from megakernel._fixed import capture_fixed
        from sglang.srt.hardware_backend.npu.graph_runner.welm_qkv_tiles import try_project_qkv
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import (
            decode_tile_scope, install_welm_tile_layouts,
        )
        from sglang.srt.layers.welmv4_npu_op import welmv4_inplace_rope_npu
        from sglang.srt.layers.welmv4_op import mmq_style_k_rms_norm

        rng = torch.Generator().manual_seed(233)
        x = torch.ones((rows, 2048), dtype=torch.bfloat16, device="npu")
        weight = (torch.randint(-3, 4, (7168, 2048), generator=rng) / 32).to(x)
        gamma = torch.randn(256, generator=rng).to(x)
        angles = torch.randn((64, 32), generator=rng)
        table = torch.cat((angles.cos(), angles.sin()), dim=-1).to(x.device)
        positions = torch.arange(rows, dtype=torch.int32, device=x.device)

        def forward(tiled):
            if tiled:
                q, k, v = try_project_qkv(x, weight)
            else:
                q, k, v = (x @ weight.T).split((6144, 512, 512), dim=-1)
                q, k = q.contiguous(), k.contiguous()
            norm = mmq_style_k_rms_norm(k.view(rows, 2, 256), gamma, 1e-5)
            q, norm = welmv4_inplace_rope_npu(
                q.view(rows, 24, 256), norm, positions, table, head_dim=256, rope_dim=64
            )
            return q, norm, v

        install_welm_tile_layouts()
        self.assertIsNone(try_project_qkv(x, weight))
        with torch.inference_mode(), decode_tile_scope():
            self.assertIsNone(try_project_qkv(x.new_zeros((3, 2048)), weight))
            self.assertIsNone(try_project_qkv(x, weight[:-1]))
            unaligned = x.new_empty(x.numel() + 1)[1:].view_as(x)
            self.assertIsNone(try_project_qkv(unaligned, weight))

        with torch.inference_mode(), tempfile.TemporaryDirectory() as build:
            with decode_tile_scope(), patch.dict("os.environ", MK_NATIVE_TILE_DEPENDENCIES="1"):
                plan = capture_fixed(lambda: forward(True), build, tile_chains=True)
            try:
                self.assertEqual(plan.scope_count, 1, plan.execution_summary)
                self.assertEqual(plan.host_boundary_count, 0, plan.execution_summary)
                scope, = plan.plans
                ratio = (programs_per_row, 1)
                expected_ratios = {(0, 1): ratio, (0, 2): ratio} if rows > 1 else {}
                self.assertEqual(scope.tile_dependency_ratios, expected_ratios)
                edges = {(0, 1), (0, 2), (1, 3), (2, 3)} if rows > 1 else set()
                self.assertEqual(set(scope.tile_dependencies), edges)
                for _ in range(3):
                    x.copy_(torch.randint(-3, 4, x.shape, generator=rng))
                    positions.copy_(torch.randperm(64, generator=rng, dtype=positions.dtype)[:rows])
                    reference = tuple(t.cpu() for t in forward(False))
                    for t in plan.outputs:
                        t.fill_(float("nan"))
                    for actual, expected in zip(plan.replay(), reference):
                        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            finally:
                plan.reset()


if __name__ == "__main__":
    unittest.main()
