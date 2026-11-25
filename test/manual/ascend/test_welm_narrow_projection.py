"""Exercise the BS1/2/4/8 attention gate and FP32 MoE router inside one A5 parent."""
import tempfile
import unittest


class NarrowProjectionTest(unittest.TestCase):
    def test_changed_weights_and_close_router_scores(self):
        for rows in (1, 2, 4, 8):
            with self.subTest(batch_size=rows):
                self._check_bucket(rows)

    def _check_bucket(self, rows):
        import torch
        import torch_npu

        if not torch.npu.is_available():
            self.skipTest("requires an NPU")
        torch.npu.set_device(0)
        if not torch.npu.get_device_name(0).startswith("Ascend950"):
            self.skipTest("requires Ascend950")
        from megakernel._fixed import capture_fixed
        from megakernel.triton import prepare_bindings
        from sglang.srt.hardware_backend.npu.graph_runner.welm_narrow_projection import narrow_projection
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import decode_tile_scope

        rng = torch.Generator().manual_seed(163)
        x = torch.zeros(rows, 2048, dtype=torch.bfloat16, device="npu")
        gate = x.new_zeros((24, 2048))
        router = x.new_zeros((2048, 512))
        # The same logical router weight can also be stored as contiguous [N,K].
        router_nk = x.new_zeros((512, 2048))
        router_view = router_nk.t()
        self.assertIsNone(narrow_projection(x, gate, router=False))
        with decode_tile_scope():
            self.assertIsNone(narrow_projection(x, gate, router=False))
        with decode_tile_scope(single_pk=True):
            self.assertIsNone(narrow_projection(x.new_zeros((3, 2048)), gate, router=False))
            self.assertIsNone(narrow_projection(x, gate[:16], router=False))
            self.assertIsNone(narrow_projection(x, gate.t().contiguous().t(), router=False))
        with torch.inference_mode(), tempfile.TemporaryDirectory() as build:
            def forward():
                return (narrow_projection(x, gate, router=False),
                        narrow_projection(x, router, router=True),
                        narrow_projection(x, router_view, router=True))

            with decode_tile_scope(single_pk=True):
                with prepare_bindings():
                    forward()
                torch.npu.synchronize()
                plan = capture_fixed(forward, build, tile_chains=False)
            try:
                self.assertEqual(plan.scope_count, 1, plan.execution_summary)
                self.assertEqual(plan.host_boundary_count, 0)
                self.assertEqual(plan.fused_operator_count, 3)
                self.assertFalse(plan.execution_summary["native_globals"])
                for i in range(5):
                    if i < 4:
                        values = torch.randn((rows, 2048), generator=rng).bfloat16()
                        gate_values = torch.randn((24, 2048), generator=rng).bfloat16()
                        router_values = torch.randn((2048, 512), generator=rng).bfloat16()
                    else:
                        # Logits within one BF16 ULP. FP32 scores remain distinct,
                        # and the exact ordering of the ten highest scores matters.
                        values = torch.zeros((rows, 2048), dtype=torch.bfloat16)
                        values[:, 0] = 1
                        values[:, 1] = torch.arange(1, rows + 1)
                        gate_values = torch.ones((24, 2048), dtype=torch.bfloat16)
                        router_values = torch.zeros((2048, 512), dtype=torch.bfloat16)
                        router_values[0,:] = 1
                        router_values[1,:16] = torch.arange(16) * (2.0**-14)
                    x.copy_(values)
                    gate.copy_(gate_values)
                    router.copy_(router_values)
                    router_nk.copy_(router_values.t())
                    gate_ref = torch.nn.functional.linear(x, gate).cpu()
                    router_ref = torch.mm(x, router, out_dtype=torch.float32).cpu()
                    for out in plan.outputs:
                        out.fill_(float('nan'))
                    g, a, b = (o.cpu() for o in plan.replay())
                    self.assertEqual(g.shape, (rows, 24))
                    self.assertEqual(a.shape, (rows, 512))
                    self.assertEqual(g.dtype, torch.bfloat16)
                    self.assertEqual(a.dtype, torch.float32)
                    torch.testing.assert_close(g, gate_ref, rtol=0, atol=0)
                    for actual in (a, b):
                        torch.testing.assert_close(actual, router_ref, rtol=1e-4, atol=5e-4)
                        self.assertTrue(torch.equal(actual.topk(10).indices, router_ref.topk(10).indices))
                    if i == 4:
                        self.assertTrue(torch.equal(a, router_ref))
            finally:
                plan.reset()


if __name__ == "__main__":
    unittest.main()
