"""A5 regression for the original decode K-RMSNorm -> in-place RoPE pair.

After sourcing the matching MegaKernel set_env.sh and installing SGLang, run
python -m unittest discover -s test/manual/ascend -p test_welm_native_tile_dependencies.py -v
from the SGLang checkout. The MK runtime and libmk_capture must be rebuilt.
"""
import tempfile
import unittest
from unittest.mock import patch


class WelmNativeTileTest(unittest.TestCase):
    def test_changed_rows_match_eager_with_a_tile_edge(self):
        self._check_changed_rows(packed=False)

    def test_packed_qkv_key_matches_dense_eager(self):
        self._check_changed_rows(packed=True)

    def _check_changed_rows(self, packed):
        try:
            import torch
            import torch_npu
        except ImportError as exc:
            self.skipTest(f"A5 device test requires torch and torch_npu: {exc}")
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires an Ascend A5 NPU")

        from megakernel._fixed import capture_fixed
        from megakernel.ops.tensor_copy import copy_kernel
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import (
            decode_tile_scope, install_welm_tile_layouts,
        )
        from sglang.srt.layers.welmv4_npu_op import welmv4_inplace_rope_npu
        from sglang.srt.layers.welmv4_op import mmq_style_k_rms_norm

        generator = torch.Generator().manual_seed(123)
        rows, dim = 8, 256
        gamma = torch.randn(dim, generator=generator).to(device="npu", dtype=torch.bfloat16)
        angles = torch.randn(64, 32, generator=generator)
        table = torch.cat((angles.cos(), angles.sin()), dim=-1).to("npu")

        def forward(q, k, positions):
            norm = mmq_style_k_rms_norm(k, gamma, 1e-5)
            return welmv4_inplace_rope_npu(q, norm, positions, table, head_dim=dim, rope_dim=64)

        with torch.inference_mode(), tempfile.TemporaryDirectory() as directory:
            examples = []
            for _ in range(3):
                q = torch.randn(rows, 24, dim, generator=generator).to(torch.bfloat16)
                k = torch.randn(rows, 2, dim, generator=generator).to(torch.bfloat16)
                positions = torch.randperm(64, generator=generator, dtype=torch.int32)[:rows].clone()
                # Eager global kernels on private buffers before installing MK bindings.
                q_ref, k_ref = forward(q.to("npu"), k.to("npu"), positions.to("npu"))
                examples.append((q, k, positions, q_ref.cpu(), k_ref.cpu()))

            q, k, positions = (t.to("npu") for t in examples[0][:3])
            def captured():
                if packed:
                    copy_kernel[(rows,)](source, storage, source.numel(), BLOCK=32 * dim, ROUND_FP32=False)
                return forward(q, k, positions)

            if packed:
                # The packed Q and V columns contain sentinels on both sides of K.
                storage = torch.full((rows, 32, dim), -7.0, dtype=k.dtype, device=k.device)
                source = storage.clone()
                k = storage[:, 24:26]
                source[:, 24:26].copy_(examples[0][1])
            install_welm_tile_layouts()
            with (
                decode_tile_scope(),
                patch.object(torch.npu, "NPUGraph", side_effect=AssertionError("unexpected NPUGraph")),
                patch.object(torch.npu, "graph", side_effect=AssertionError("unexpected ACL Graph")),
            ):
                plan = capture_fixed(captured, directory, tile_chains=True)
                try:
                    self.assertEqual(plan.scope_count, 1, plan.execution_summary)
                    self.assertEqual(plan.fused_operator_count, 3 if packed else 2, plan.execution_summary)
                    self.assertEqual(plan.host_boundary_count, 0, plan.execution_summary)
                    expected_edges = [(0, 1), (1, 2)] if packed else [(0, 1)]
                    self.assertEqual(plan.plans[0].tile_dependencies, expected_edges)
                    for q_input, k_input, pos, q_ref, k_ref in examples:
                        q.copy_(q_input)
                        if packed:
                            source[:, 24:26].copy_(k_input)
                            storage.zero_()
                        else:
                            k.copy_(k_input)
                        positions.copy_(pos)
                        q_out, k_out = plan.replay()
                        # Check every row and every head, including the untouched RoPE prefix.
                        torch.testing.assert_close(q_out.cpu(), q_ref, rtol=0, atol=0)
                        torch.testing.assert_close(k_out.cpu(), k_ref, rtol=0, atol=0)
                finally:
                    plan.reset()


if __name__ == "__main__":
    unittest.main()
