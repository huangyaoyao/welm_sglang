"""Check the original OE decode JIT and its compact four-plane storage on A5."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class OriginalOeDecode(unittest.TestCase):
    def test_compact_hash_native_child(self):
        try:
            import torch
            import torch_npu
        except ImportError as error:
            self.skipTest(str(error))
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires an Ascend A5 NPU")
        from megakernel._fixed import capture_fixed
        from megakernel.triton import prepare_bindings
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import (
            decode_tile_scope, install_welm_tile_layouts,
        )
        from sglang.srt.layers import welmv4_npu_op as kernels

        vocab = 155648
        sizes = (16000008, 16000016, 16000024, 16000032)
        for batch in (1, 2, 4, 8, 12, 16):
            pitch = 1 << (batch - 1).bit_length()
            with self.subTest(batch=batch), torch.inference_mode(), TemporaryDirectory() as build:
                table = torch.zeros((17, 512), dtype=torch.int32, device="npu")
                ids = torch.zeros(batch, dtype=torch.int64, device="npu")
                rows = torch.zeros_like(ids)
                pos = torch.zeros(batch, dtype=torch.int32, device="npu")

                def forward():
                    return kernels.welmv4_oe_hash_decode_4way_npu(
                        ids, table, rows, pos, vocab_size=vocab, oe_vocab_sizes=sizes,
                    )

                install_welm_tile_layouts()
                with decode_tile_scope(native_ops_only=True):
                    with prepare_bindings():
                        forward()
                    plan = capture_fixed(forward, Path(build), tile_chains=True)
                try:
                    self.assertEqual(plan.host_boundary_count, 0, plan.execution_summary)
                    self.assertEqual(plan.fused_operator_count, 1, plan.execution_summary)
                    providers = [(task.get("provider"), task.get("name"))
                                 for task in plan.tasks if isinstance(task, dict)]
                    self.assertTrue(any(provider == "triton" and "_welmv4_oe_hash_decode_4way_kernel" in name
                                        for provider, name in providers), providers)
                    self.assertEqual(plan.outputs.stride(), (pitch, 1))
                    self.assertEqual(plan.outputs.untyped_storage().nbytes(), 4 * pitch * 4)
                    ptr = plan.outputs.data_ptr()
                    for step in range(4):
                        gen = torch.Generator().manual_seed(773 + step)
                        history = torch.randint(vocab, (17, 512), generator=gen, dtype=torch.int32)
                        current = torch.randint(vocab, (batch,), generator=gen)
                        req = torch.randperm(17, generator=gen)[:batch]
                        positions = torch.tensor([0, 1, 2, 511] * 4, dtype=torch.int32).roll(step)[:batch]
                        for dst, src in zip((table, ids, rows, pos), (history, current, req, positions)):
                            dst.copy_(src)
                        reference = torch.empty(4, batch, dtype=torch.int32)
                        for j in range(batch):
                            r, p, token = int(req[j]), int(positions[j]), int(current[j])
                            h1 = int(history[r, p - 1]) if p >= 1 else 0
                            h2 = int(history[r, p - 2]) if p >= 2 else 0
                            for branch, modulus in enumerate(sizes):
                                value = token + h1 * vocab
                                if branch >= 2:
                                    value += h2 * vocab * vocab
                                reference[branch, j] = ((value * 2654435761) & 0xffffffff) % modulus
                        torch.testing.assert_close(forward().cpu(), reference, rtol=0, atol=0)
                        # Detect writes past the compact view even if the NPU allocator
                        # normally rounds small allocations up to a larger size class.
                        guard = torch.full((16 + 4 * pitch,), -1, device="npu", dtype=torch.int32)
                        view = guard[8:-8].view(4, pitch)
                        kernels._welmv4_oe_hash_decode_4way_kernel[(1,)](
                            ids, table, rows, pos, view, batch, 512, 1,
                            VOCAB_SIZE=vocab, OE_V0=sizes[0], OE_V1=sizes[1],
                            OE_V2=sizes[2], OE_V3=sizes[3], HASH_MUL=2654435761,
                            BLOCK_B=pitch, PADDED_OUTPUT=True, compile_mode="simd",
                        )
                        actual_guard = guard.cpu()
                        self.assertTrue(actual_guard[:8].eq(-1).all())
                        self.assertTrue(actual_guard[-8:].eq(-1).all())
                        torch.testing.assert_close(actual_guard[8:-8].view(4, pitch)[:, :batch], reference, rtol=0, atol=0)
                        plan.outputs.fill_(-1)
                        result = plan.replay()
                        self.assertEqual(result.data_ptr(), ptr)
                        torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)
                finally:
                    plan.reset()


if __name__ == "__main__":
    unittest.main()
