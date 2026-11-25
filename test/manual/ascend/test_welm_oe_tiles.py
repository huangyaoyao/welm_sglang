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
    def setUp(self):
        self.enterContext(patch.dict(sys.modules, {
            "sglang.srt.utils": SimpleNamespace(get_bool_env_var=lambda _: False),
        }))

    def test_single_pk_policy_is_local_to_explicit_decode_scope(self):
        active = bindings["welm_single_pk_decode_active"]
        scope = bindings["decode_tile_scope"]
        self.assertFalse(active())
        with scope(single_pk=True):
            self.assertTrue(active())
            with scope():
                self.assertTrue(bindings["welm_decode_tiles_active"]())
                self.assertFalse(active())
            self.assertTrue(active())
        self.assertFalse(active())

    def test_four_lookup_writes_one_complete_output_row_per_program(self):
        for batch in (1, 2, 4, 8):
            ids, out = tensor((4, batch)), tensor((batch, 2048), width=2)
            tables = [tensor((17, 512), width=2) for _ in range(4)]
            layout = oe["_lookup_4way"](ids, *tables, out, PITCH=128, WIDTH=512)
            self.assertEqual((layout["count"], layout["block"]), (batch, 1))
            self.assertEqual(layout["shared_reads"], [ids, *tables])
            self.assertEqual(layout["writes"], [(out, 4096)])

    def test_decode_table_read_and_four_output_planes(self):
        ids, rows, starts = [tensor((257,)) for _ in range(3)]
        table, out = tensor((512, 128)), tensor((4, 257))
        layout = oe["_decode_hash"](ids, table, rows, starts, out, 257, 128, 3, BLOCK_B=128)
        self.assertEqual(layout["plane_writes"], [(out, 4, 4)])
        self.assertEqual(layout["reads"], [(ids, 4), (rows, 4), (starts, 4)])
        self.assertEqual(layout["shared_reads"], [table])
        self.assertIsNone(oe["_decode_hash"](
            ids, table, rows, starts, tensor((257, 4)), 257, 128, 3, BLOCK_B=128))

    def test_decode_and_draft_describe_full_output_tiles_with_compact_inputs(self):
        ids, rows, starts = [tensor((259,)) for _ in range(3)]
        table, out = tensor((259, 16)), tensor((4, 384))
        decode = oe["_decode_hash"](
            ids, table, rows, starts, out, 259, 16, 3, BLOCK_B=128,
            PADDED_OUTPUT=True,
        )
        draft = oe["_draft_hash"](
            ids, rows, starts, out, 259, 3, BLOCK_B=128, PADDED_OUTPUT=True,
        )
        for layout in (decode, draft):
            self.assertEqual(layout["count"], 259)
            self.assertEqual(layout["reads"], [(ids, 4), (rows, 4), (starts, 4)])
            self.assertEqual(layout["plane_writes"], [(out, 4, 4, 384)])
        self.assertEqual(decode["shared_reads"], [table])

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
    def test_four_lookup_concat_capture_and_fallbacks(self):
        from contextlib import nullcontext
        from tempfile import TemporaryDirectory

        try:
            import torch
            import torch_npu
        except ImportError as error:
            self.skipTest(f"A5 test requires torch and torch_npu: {error}")
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires an Ascend A5 NPU")
        from megakernel._fixed import capture_fixed
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import (
            decode_tile_scope, install_welm_tile_layouts,
        )
        from sglang.srt.layers import vocab_parallel_embedding as embedding

        # Single-rank fixture for the public concat/reduce path, without a process group.
        layer = embedding.VocabParallelEmbedding.__new__(embedding.VocabParallelEmbedding)
        torch.nn.Module.__init__(layer)
        layer.tp_size = 1

        def forward(inputs, dim):
            return layer.concat_and_reduce_local_outputs(inputs, dim)

        install_welm_tile_layouts()
        gen = torch.Generator().manual_seed(42)
        with (
            torch.inference_mode(),
            patch.object(embedding, "get_tp_group", return_value=None),
            patch.object(embedding, "is_allocation_symmetric", return_value=False),
            patch.object(embedding, "use_symmetric_memory", return_value=nullcontext()),
        ):
            for rows, dim in ((1, -1), (2, 1), (8, -1)):
                with self.subTest(rows=rows, dim=dim), TemporaryDirectory() as build:
                    inputs = tuple(torch.zeros((rows, 512), device="npu", dtype=torch.bfloat16)
                                   for _ in range(4))
                    with decode_tile_scope():
                        plan = capture_fixed(lambda: forward(inputs, dim), build, tile_chains=True)
                    try:
                        self.assertEqual(plan.host_boundary_count, 0, plan.execution_summary)
                        self.assertEqual(plan.scope_count, 1, plan.execution_summary)
                        self.assertEqual(plan.fused_operator_count, 1, plan.execution_summary)
                        for _ in range(3):
                            values = [torch.randn(x.shape, generator=gen).bfloat16() for x in inputs]
                            for target, value in zip(inputs, values):
                                target.copy_(value)
                            plan.outputs.fill_(float("nan"))
                            actual = plan.replay().cpu()
                            torch.testing.assert_close(actual, torch.cat(values, dim), rtol=0, atol=0)
                    finally:
                        plan.reset()

            # Full-width decode has an opt-in native implementation. Every other
            # input contract still follows the existing torch.cat path.
            for case, xs, dim, active in (
                ("eager", inputs, -1, False),
                ("row dimension", inputs, 0, True),
                ("three branches", inputs[:3], -1, True),
                ("strided", tuple(torch.stack((x, x), dim=-1)[..., 0] for x in inputs), -1, True),
                ("unequal widths", (inputs[0][:, :128], *inputs[1:]), -1, True),
                ("float32", tuple(x.float() for x in inputs), -1, True),
            ):
                with self.subTest(case=case), (decode_tile_scope() if active else nullcontext()):
                    reference = torch.cat(xs, dim)
                    with patch.object(torch, "cat", wraps=torch.cat) as default_cat:
                        actual = forward(xs, dim)
                    default_cat.assert_called_once()
                    torch.testing.assert_close(actual.cpu(), reference.cpu(), rtol=0, atol=0)

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

        context = 16
        for batch in (1, 256, 259, 513):
            with self.subTest(batch=batch):
                pitch = ((batch + 127) // 128) * 128
                options = dict(vocab_size=32000, oe_vocab_sizes=(997, 991, 983, 977))
                gen = torch.Generator().manual_seed(27)
                table = torch.randint(32000, (batch, context), dtype=torch.int32, generator=gen).to("npu")
                with torch.inference_mode(), TemporaryDirectory() as build:
                    def forward(ids, req, pos, previous):
                        hashed = welmv4_oe_hash_decode_4way_npu(ids, table, req, pos, **options)
                        draft_args = (ids, hashed[0], previous) if batch == 256 else (
                            hashed[0], previous, ids
                        )
                        return hashed, welmv4_oe_hash_explicit_history_4way_npu(
                            *draft_args, **options)

                    examples = []
                    for step in range(3):
                        values = [torch.randint(32000, (batch,), generator=gen, dtype=torch.int64),
                                  torch.randperm(batch, generator=gen, dtype=torch.int64),
                                  torch.arange(batch, dtype=torch.int32).roll(step) % context,
                                  torch.randint(32000, (batch,), generator=gen, dtype=torch.int64)]
                        operands = tuple(t.to("npu") for t in values)
                        eager = forward(*operands)
                        self.assertTrue(all(t.is_contiguous() for t in eager))
                        expected = tuple(t.cpu() for t in eager)
                        examples.append((operands, expected))
                    fixed = tuple(t.clone() for t in examples[0][0])
                    install_welm_tile_layouts()
                    with decode_tile_scope():
                        graph = capture_fixed(lambda: forward(*fixed), build, tile_chains=True)
                    try:
                        self.assertEqual(graph.host_boundary_count, 0, graph.execution_summary)
                        self.assertEqual(graph.scope_count, 1)
                        self.assertEqual(graph.plans[0].tile_dependencies, [(0, 1)] if batch > 128 else [])
                        for operands, expected in examples:
                            for dst, src in zip(fixed, operands):
                                dst.copy_(src)
                            for result, eager in zip(graph.replay(), expected):
                                self.assertEqual(result.shape, (4, batch))
                                self.assertEqual(result.stride(), (pitch, 1))
                                if pitch != batch:
                                    full = result.as_strided((4, pitch), (pitch, 1))
                                    padding = full[:, batch:].cpu()
                                    self.assertTrue(torch.equal(padding, torch.zeros_like(padding)))
                                torch.testing.assert_close(result.cpu(), eager, rtol=0, atol=0)
                    finally:
                        graph.reset()


if __name__ == "__main__":
    unittest.main()
