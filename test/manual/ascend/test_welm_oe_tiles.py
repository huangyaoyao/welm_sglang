"""Check the four separate output planes and the decode-only registration."""
from contextvars import copy_context
from math import prod
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = (Path(__file__).resolve().parents[3] /
        "python/sglang/srt/hardware_backend/npu/graph_runner")
oe = runpy.run_path(str(ROOT / "welm_oe_tiles.py"))
bindings = runpy.run_path(str(ROOT / "welm_tile_layouts.py"))


def tensor(shape, width=4):
    return SimpleNamespace(shape=shape, numel=lambda: prod(shape), element_size=lambda: width)


class OeTileTest(unittest.TestCase):
    def test_decode_table_read_and_four_output_planes(self):
        ids, rows, starts = [tensor((257,)) for _ in range(3)]
        table, out = tensor((512, 128)), tensor((4, 257))
        layout = oe["_decode_hash"](ids, table, rows, starts, out, 257, 128, 3, BLOCK_B=128)
        self.assertEqual(layout["plane_writes"], [(out, 4, 4)])
        self.assertEqual(layout["reads"], [(ids, 4), (rows, 4), (starts, 4)])
        self.assertEqual(layout["shared_reads"], [table])
        self.assertIsNone(oe["_decode_hash"](
            ids, table, rows, starts, tensor((257, 4)), 257, 128, 3, BLOCK_B=128))

    def test_draft_verify_keeps_indirect_offsets_opaque(self):
        history = [tensor((16,)) for _ in range(3)]
        out = tensor((4, 16))
        draft = oe["_draft_hash"](*history, out, 16, 2, BLOCK_B=8)
        self.assertEqual(draft["reads"], [(t, 4) for t in history])
        ids, table = tensor((16,)), tensor((20, 100))
        metadata = [tensor((4,)) for _ in range(4)]
        args = (ids, table, *metadata, out, 16, 100)
        verify = oe["_verify_hash"](*args, 1, 4, BLOCK_T=512)
        self.assertEqual(verify["whole_writes"], [out])
        self.assertEqual(verify["shared_reads"], [ids, table])
        self.assertIsNone(oe["_verify_hash"](*args, 2, 8, BLOCK_T=512))

    def test_decode_binding_inactive_in_eager_and_restored_after_failure(self):
        normal_context = copy_context()
        def native(**kwargs):
            return kwargs
        with patch.dict(sys.modules, {"megakernel.triton": SimpleNamespace(native=native)}):
            options = bindings["_decode_native"](tile_layout=lambda: {"count": 8})
        active = options["enabled"]
        self.assertFalse(active())
        with self.assertRaisesRegex(RuntimeError, "capture"):
            with bindings["decode_tile_scope"]():
                self.assertTrue(active())
                self.assertFalse(normal_context.run(active))
                self.assertTrue(options["tile_layout"]()["allow_subviews"])
                raise RuntimeError("capture")
        self.assertFalse(active())


class OeTileDeviceTest(unittest.TestCase):
    def test_decode_hash_to_draft_hash_reads_one_plane(self):
        try:
            import torch
            import torch_npu
        except ImportError as error:
            self.skipTest(f"A5 test requires torch and torch_npu: {error}")
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires an Ascend A5 NPU")
        from tempfile import TemporaryDirectory
        from megakernel._fixed import capture_fixed
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import (
            decode_tile_scope, install_welm_tile_layouts,
        )
        from sglang.srt.layers.welmv4_npu_op import (
            welmv4_oe_hash_decode_4way_npu, welmv4_oe_hash_explicit_history_4way_npu,
        )

        batch, context = 259, 16  # Three programs with a masked tail.
        options = dict(vocab_size=32000, oe_vocab_sizes=(997, 991, 983, 977))
        gen = torch.Generator().manual_seed(27)
        table = torch.randint(32000, (batch, context), dtype=torch.int32, generator=gen).to("npu")
        with torch.inference_mode(), TemporaryDirectory() as build:
            def forward(ids, req, pos, previous):
                hashed = welmv4_oe_hash_decode_4way_npu(ids, table, req, pos, **options)
                return hashed, welmv4_oe_hash_explicit_history_4way_npu(
                    hashed[0], previous, ids, **options)

            examples = []
            for _ in range(3):
                values = [torch.randint(32000, (batch,), generator=gen, dtype=torch.int32),
                          torch.randperm(batch, generator=gen, dtype=torch.int32),
                          torch.randint(context, (batch,), generator=gen, dtype=torch.int32),
                          torch.randint(32000, (batch,), generator=gen, dtype=torch.int32)]
                operands = tuple(t.to("npu") for t in values)
                expected = tuple(t.cpu() for t in forward(*operands))
                examples.append((operands, expected))
            fixed = tuple(t.clone() for t in examples[0][0])
            install_welm_tile_layouts()
            with decode_tile_scope():
                graph = capture_fixed(lambda: forward(*fixed), build, tile_chains=True)
            try:
                self.assertEqual(graph.host_boundary_count, 0, graph.execution_summary)
                self.assertEqual(graph.scope_count, 1)
                self.assertEqual(graph.plans[0].tile_dependencies, [(0, 1)])
                for operands, expected in examples:
                    for dst, src in zip(fixed, operands):
                        dst.copy_(src)
                    for result, eager in zip(graph.replay(), expected):
                        torch.testing.assert_close(result.cpu(), eager, rtol=0, atol=0)
            finally:
                graph.reset()


if __name__ == "__main__":
    unittest.main()
