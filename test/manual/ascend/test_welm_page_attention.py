"""A5 regression for WeLM physical-page attention and its pre-write guards."""
import math
import os
import tempfile
import unittest
from unittest.mock import patch


class _A5Test(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            import torch_npu
        except ModuleNotFoundError as error:
            if error.name not in {"torch", "torch_npu"}:
                raise
            self.skipTest(f"A5 test requires {error.name}")
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires an Ascend A5 NPU")
        self.torch = torch
        self.torch_npu = torch_npu


class WelmPageAttentionTest(_A5Test):
    def _check_graph(self, plan, fine, gated=False, batch=None):
        locations = {}
        for scope_id, scope in enumerate(plan.plans):
            for kernel_id, name in enumerate(scope.kernel_names):
                locations.setdefault(name, []).append((scope_id, kernel_id))

        self.assertNotIn("_read_flash_pages", locations)
        for name in ("_write_kv_pages", "_page_attention_partials", "_merge_page_attention"):
            self.assertEqual(len(locations.get(name, [])), 1, (name, locations, plan.execution_summary))

        (write_scope, writer), = locations["_write_kv_pages"]
        (partial_scope, partial), = locations["_page_attention_partials"]
        (merge_scope, merge), = locations["_merge_page_attention"]
        self.assertEqual(write_scope, partial_scope, locations)
        scope = plan.plans[partial_scope]
        self.assertIn(writer, scope.dependencies[partial], locations)
        self.assertEqual((writer, partial) in scope.tile_dependencies, fine, locations)
        self.assertLessEqual(partial_scope, merge_scope, locations)
        if partial_scope == merge_scope:
            self.assertIn(partial, scope.dependencies[merge], locations)
            self.assertNotIn((partial, merge), scope.tile_dependencies, locations)
        if gated:
            (gate_scope, gate), = locations["sigmoid_mul_kernel_npu"]
            self.assertEqual(gate_scope, merge_scope, locations)
            scope = plan.plans[merge_scope]
            self.assertIn(merge, scope.dependencies[gate], scope.dependencies)
            edge = (merge, gate)
            self.assertEqual(edge in scope.tile_dependencies, fine, locations)
            self.assertEqual(scope.tile_dependency_ratios.get(edge), (1, 3) if fine else None)
            # These are the recorded child launch grids for our pinned shape.
            # The selected fine edge above separately verifies MK could use
            # the matching launch widths for this scope on the test device.
            for kernel, grid in ((merge, batch * 2), (gate, batch * 6)):
                task = plan.tasks[scope.task_indices[kernel]]
                self.assertEqual(task["blocks"], grid, (scope.kernel_names[kernel], task["blocks"], grid))

    def _dense_reference(self, query, cache, table, lengths, sinks, window):
        # Enumerate logical tokens in order. Multiple references to a physical
        # page count separately. Every 12 consecutive query heads share a KV head.
        torch = self.torch
        grouped_query = query.cpu().double().view(-1, 2, 12, 256)
        keys, values = (tensor.cpu().double() for tensor in cache)
        result = torch.zeros(query.shape, dtype=torch.float64)
        for row, length in enumerate(lengths.tolist()):
            if length == 0:
                continue
            start = max(0, length - 1 - window) if window >= 0 else 0
            positions = torch.arange(start, length)
            pages = table[row, positions // 64].long()
            offsets = positions % 64
            scores = torch.einsum("gmd,tgd->gmt", grouped_query[row], keys[pages, offsets]) / 16
            if sinks is not None:
                sink_scores = sinks.cpu().reshape(2, 12, 1).double()
                scores = torch.cat([scores, sink_scores], dim=-1)
            # A sink contributes to the denominator and has a zero value vector.
            weights = scores.softmax(dim=-1)[..., :len(positions)]
            output = torch.einsum("gmt,tgd->gmd", weights, values[pages, offsets])
            result[row] = output.reshape(24, 256)
        return result.float()

    def _compare_output(self, actual, reference, active_rows, label):
        torch = self.torch
        got = actual[:active_rows].float()
        ref = reference[:active_rows].float()
        self.assertTrue(torch.isfinite(got).all(), f"{label}: nonfinite candidate")
        self.assertTrue(torch.isfinite(ref).all(), f"{label}: nonfinite reference")
        if not active_rows:
            return

        # CANN's tiled softmax and the FP32 partial reduction round to BF16
        # in different orders. Check individual elements and relative RMS.
        torch.testing.assert_close(got, ref, atol=1 / 128, rtol=1 / 64, msg=lambda msg: f"{label}: {msg}")
        relative_rms = float((got - ref).norm() / ref.norm().clamp_min(1e-12))
        self.assertLessEqual(relative_rms, 0.02, (label, relative_rms))

    def _page_inputs(self, batch, active_rows, span, step):
        torch = self.torch
        lengths = torch.tensor(
            [span[row % 2] if row < active_rows else 0 for row in range(batch)], dtype=torch.int32
        )
        slots = torch.zeros(batch, dtype=torch.int32)
        table = torch.zeros(batch, 4, dtype=torch.int32)
        for row, length in enumerate(lengths[:active_rows].tolist()):
            count = (length + 63) // 64
            for column in range(count):
                table[row, column] = 1 + (step * 5 + 3 * column) % 9
            if step == 1 and count >= 3:
                table[row, 1] = table[row, 0]
            table[row, count - 1] = 12 + (row + step) % 8
            slots[row] = int(table[row, count - 1]) * 64 + (length - 1) % 64
            if step == 3:
                # These independent write destinations are absent from the page table.
                slots[row] = 20 * 64 + row
        return lengths, slots, table

    def _poison_partials(self, workspace):
        buffers = [
            value for key, value in workspace.items()
            if isinstance(key, tuple) and key and key[0] == "page_attention"
        ]
        self.assertEqual(len(buffers), 1, "expected one fixed partial-attention workspace")
        partials = buffers[0]
        self.assertTrue(self.torch.is_tensor(partials) and partials.is_floating_point())
        partials.fill_(float("nan"))

    def test_paged_partial_attention(self):
        for batch in (1, 2, 4, 8):
            for window in (-1, 31):
                for fine in (False, True):
                    with self.subTest(batch=batch, window=window, fine=fine):
                        self._exercise(batch, window, fine)

    @patch.dict(os.environ, SGLANG_NPU_WELMV4_MK_PAGE_GATE_TILES="1")
    def test_gated_page_attention_replay(self):
        for batch in (1, 2, 4, 8):
            for window in (-1, 31):
                for fine in (False, True):
                    with self.subTest(batch=batch, window=window, fine=fine):
                        self._exercise(batch, window, fine, gated=True)

    def _exercise(self, batch, window, fine, gated=False):
        import cann_ops_transformer as cann
        from megakernel._fixed import capture_fixed
        from sglang.srt.hardware_backend.npu.graph_runner.welm_page_attention import try_page_attention
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import (
            decode_tile_scope, install_welm_tile_layouts, welm_decode_tile_workspace,
        )
        from sglang.srt.layers.welmv4_npu_op import inplace_sigmoid_mul_npu

        torch = self.torch
        rng = torch.Generator().manual_seed(29017)

        def random_bf16(*shape):
            return torch.randn(shape, dtype=torch.bfloat16, generator=rng).npu()

        num_pages = 21
        seed_cache = tuple(
            random_bf16(num_pages * 64 * 2 * 256).view(num_pages, 64, 2, 256) for _ in range(2)
        )

        def new_cache():
            return tuple(tensor.flatten().clone().view_as(tensor) for tensor in seed_cache)

        cache = new_cache()
        native_cache = new_cache()
        eager_page_cache = new_cache() if gated else None
        query = random_bf16(batch, 24, 256)
        key = random_bf16(batch, 2, 256)
        value = random_bf16(batch, 4, 256)[:, 2:]
        slot_buffer = torch.zeros(batch, dtype=torch.int32, device="npu")
        length_buffer = torch.ones_like(slot_buffer)
        table_buffer = torch.ones((batch, 4), dtype=torch.int32, device="npu")
        # Cover the no-sink specialization in coarse SWA; the other cases
        # carry an FP32 sink for each query head.
        sinks = torch.zeros(24, dtype=torch.float32, device="npu") if gated or fine or window < 0 else None
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()
        query_offsets = torch.arange(batch + 1, device="npu", dtype=torch.int32)
        gate = torch.zeros((batch, 24, 1), dtype=torch.bfloat16, device="npu") if gated else None

        def page_attention(pool, scratch):
            output = try_page_attention(
                query, key, value, slot_buffer, *pool, table_buffer, length_buffer,
                sinks, window, 1 / 16, scratch,
            )
            self.assertIsNotNone(output, "supported A5 decode bucket fell back")
            return output

        def forward():
            output = page_attention(cache, workspace)
            if gated:
                # The backend flattens the result and the model restores its
                # head axis. Exercise the unmodified public gate call on that view.
                output = output.view(batch, -1).view(batch, 24, 256)
                inplace_sigmoid_mul_npu(gate, output)
            return output

        def gate_reference(attention, factors):
            return (attention.to(torch.bfloat16).float() * factors.float()).to(torch.bfloat16)

        def native_attention():
            options = dict(
                cu_seqlens_q=query_offsets, cu_seqlens_kv=None,
                seqused_q=(length_buffer > 0).int(), seqused_kv=length_buffer,
                max_seqlen_q=1, max_seqlen_kv=-1,
                mask_mode=4 if window >= 0 else 3, win_left=window, win_right=0 if window >= 0 else -1,
                layout_q="TND", layout_kv="PA_BBND", layout_out="TND",
            )
            schedule = cann.flash_attn_metadata(24, 2, 256, **options)
            output, _ = cann.flash_attn(
                query, *native_cache, block_table=table_buffer, sinks=sinks, attn_mask=mask,
                metadata=schedule, softmax_scale=1 / 16, return_softmax_lse=False, **options,
            )
            return output.reshape_as(query).cpu()

        padded_active_rows = {1: 0, 2: 1, 4: 3, 8: 5}[batch]
        cases = [(batch, (1, 64)), (batch, (193, 130)), (padded_active_rows, (65, 129)), (batch, (192, 129))]
        if gated:
            install_welm_tile_layouts()
        with torch.inference_mode(), decode_tile_scope(), tempfile.TemporaryDirectory() as build:
            workspace = welm_decode_tile_workspace() if gated else {}
            with patch.dict(os.environ, MK_NATIVE_TILE_DEPENDENCIES=str(int(fine))):
                plan = capture_fixed(forward, build, tile_chains=True)
            try:
                self._check_graph(plan, fine, gated, batch)
                for step, (active_rows, span) in enumerate(cases):
                    lengths, slots, table = self._page_inputs(batch, active_rows, span, step)
                    query.copy_(random_bf16(*query.shape))
                    key.copy_(random_bf16(*key.shape))
                    value.copy_(random_bf16(*value.shape))
                    if active_rows < batch:
                        # Dummy writes share slot zero, so their order cannot alter the oracle.
                        key[active_rows:].zero_()
                        value[active_rows:].zero_()
                    if sinks is not None:
                        sinks.copy_(torch.linspace(-1.0, 4.0 + step, 24).npu())
                    slot_buffer.copy_(slots)
                    length_buffer.copy_(lengths)
                    table_buffer.copy_(table)
                    for candidate, native, seed in zip(cache, native_cache, seed_cache):
                        candidate.copy_(seed)
                        native.copy_(seed)

                    self.torch_npu.npu_scatter_pa_kv_cache(
                        key.contiguous(), value.contiguous(), *native_cache, slot_buffer, cache_mode="Norm"
                    )
                    reference = native_attention()
                    dense = self._dense_reference(query, native_cache, table, lengths, sinks, window)
                    if gated:
                        # Match sigmoid_mul_kernel_npu's independent BF16
                        # rounding of the sigmoid factor before multiplication.
                        logits = torch.tensor([-8, -4, -2, -1, 0, 1, 2, 4, 8], dtype=torch.bfloat16)
                        index = (torch.arange(batch * 24) + 3 * step) % len(logits)
                        gate_values = logits[index].view(batch, 24, 1)
                        gate.copy_(gate_values)
                        factors = gate_values.float().sigmoid().to(torch.bfloat16)
                        # Eager PAGE also writes KV. Give it the same initial
                        # data while preserving the native scatter result as
                        # an independent whole-cache oracle.
                        for eager_buffer, seed in zip(eager_page_cache, seed_cache):
                            eager_buffer.copy_(seed)
                        with decode_tile_scope():
                            eager = page_attention(eager_page_cache, welm_decode_tile_workspace()).cpu()
                        exact = gate_reference(eager, factors)
                        reference = gate_reference(reference, factors)
                        dense = gate_reference(dense, factors)
                    self._poison_partials(workspace)
                    plan.outputs.fill_(float("nan"))
                    actual = plan.replay().reshape_as(query).cpu()
                    label = f"batch={batch} window={window} fine={fine} step={step}"
                    if gated:
                        torch.testing.assert_close(actual, exact, rtol=0, atol=0, msg=label)
                    self._compare_output(actual, reference, active_rows, label + " vs CANN")
                    self._compare_output(actual, dense, active_rows, label + " vs CPU float64")
                    self.assertEqual(torch.count_nonzero(actual[active_rows:]).item(), 0, label)
                    for candidate, native in zip(cache, native_cache):
                        torch.testing.assert_close(candidate.cpu(), native.cpu(), atol=0, rtol=0)
            finally:
                plan.reset()


class WelmPageAttentionGuards(_A5Test):
    def setUp(self):
        super().setUp()
        torch = self.torch
        self.batch = 2
        self.num_pages = 5

        def filled_bf16(value, *shape):
            return torch.full((math.prod(shape),), value, dtype=torch.bfloat16, device="npu").view(shape)

        self.query = filled_bf16(1, self.batch, 24, 256)
        self.key = filled_bf16(3, self.batch, 2, 256)
        self.value = filled_bf16(5, self.batch, 2, 256)
        self.cache = tuple(filled_bf16(value, self.num_pages, 64, 2, 256) for value in (-7, -11))
        self.before = tuple(tensor.cpu() for tensor in self.cache)
        self.slots = torch.tensor([64, 128], dtype=torch.int32, device="npu")
        self.table = torch.tensor([[1], [2]], dtype=torch.int32, device="npu")
        self.lengths = torch.ones_like(self.slots)
        self.sinks = torch.zeros(24, dtype=torch.float32, device="npu")
        self.marker = object()
        self.workspace = {"unrelated_owner": self.marker}

    def _call(self, *, key=None, sinks=None):
        from sglang.srt.hardware_backend.npu.graph_runner.welm_page_attention import try_page_attention
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import decode_tile_scope

        with self.torch.inference_mode(), decode_tile_scope():
            return try_page_attention(
                self.query, self.key if key is None else key, self.value, self.slots, *self.cache,
                self.table, self.lengths, self.sinks if sinks is None else sinks, -1, 1 / 16, self.workspace,
            )

    def _assert_untouched(self):
        self.torch.npu.synchronize()
        for actual, original in zip(self.cache, self.before):
            self.torch.testing.assert_close(actual.cpu(), original, rtol=0, atol=0)
        self.assertEqual(self.workspace, {"unrelated_owner": self.marker})

    @patch.dict(os.environ, SGLANG_NPU_WELMV4_MK_PAGE_GATE_TILES="1")
    def test_gate_dispatch_tracks_the_complete_page_output_in_its_scope(self):
        from sglang.srt.hardware_backend.npu.graph_runner.welm_page_attention import try_page_attention
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import (
            decode_tile_scope, welm_decode_tile_workspace,
        )
        from sglang.srt.layers import welmv4_npu_op

        def page_output():
            result = try_page_attention(
                self.query, self.key, self.value, self.slots, *self.cache,
                self.table, self.lengths, self.sinks, -1, 1 / 16, welm_decode_tile_workspace(),
            )
            self.assertIsNotNone(result)
            return result

        def assert_dispatch(y, expected_block, **options):
            gates = self.query.new_zeros((y.shape[0], 24, 1))
            with patch.object(welmv4_npu_op, "sigmoid_mul_kernel_npu") as kernel:
                welmv4_npu_op.inplace_sigmoid_mul_npu(gates, y, **options)
                launch = kernel.__getitem__.return_value
                launch.assert_called_once()
                self.assertEqual(launch.call_args.kwargs["BLOCK_SIZE"], expected_block)
                logical_heads = y.shape[0] * 24
                kernel.__getitem__.assert_called_once_with(((logical_heads + expected_block - 1) // expected_block,))

        with self.torch.inference_mode():
            with patch.dict(os.environ):
                os.environ.pop("SGLANG_NPU_WELMV4_MK_PAGE_GATE_TILES", None)
                with decode_tile_scope():
                    assert_dispatch(page_output(), 128)
            with decode_tile_scope():
                unrelated = self.query.clone()
                assert_dispatch(unrelated, 128)
                # Shape and dtype also match on a clone of the genuine result.
                assert_dispatch(page_output().clone(), 128)
                # These views share the allocation. Each omits an entire row.
                for row in (0, 1):
                    source = page_output()
                    assert_dispatch(source[row:row + 1], 128)
                explicit = page_output()
                assert_dispatch(explicit.view_as(explicit), 128, BLOCK_SIZE=128)
                explicit = page_output()
                assert_dispatch(explicit, 16, BLOCK_SIZE=16)
                source = page_output()
                model_view = source.view(self.batch, -1).view(self.batch, 24, 256)
                assert_dispatch(model_view, 4)
                assert_dispatch(model_view, 128)
                retained = page_output()
            assert_dispatch(retained, 128)
            with decode_tile_scope():
                assert_dispatch(retained, 128)

    def test_unsupported_operands_return_before_writing(self):
        strided_sinks = self.torch.zeros(48, device="npu", dtype=self.torch.float32)[::2]
        self.assertFalse(strided_sinks.is_contiguous())
        cases = (("K rows disagree", {"key": self.key[:1]}), ("strided per-head sink", {"sinks": strided_sinks}))
        for reason, arguments in cases:
            with self.subTest(reason=reason):
                self.assertIsNone(self._call(**arguments))
                self._assert_untouched()

    def test_partial_allocation_failure_preserves_cache_and_workspace(self):
        torch = self.torch
        # This pool has one group per page. Only fail the FP32 partial tensor,
        # whether allocated with its graph shape or via flat ND storage.
        partial_shape = (self.num_pages, self.batch, 24, 272)
        flat_shape = (math.prod(partial_shape),)
        real_empty = torch.empty
        injected = []

        def fail_partials(*args, **kwargs):
            size = args[0] if args else kwargs.get("size")
            shape = (size,) if isinstance(size, int) else tuple(size or ())
            if (
                shape in (partial_shape, flat_shape)
                and kwargs.get("dtype") == torch.float32
                and kwargs.get("device") == self.query.device
            ):
                injected.append(shape)
                raise torch.OutOfMemoryError("test: physical page attention FP32 partials")
            return real_empty(*args, **kwargs)

        with patch.object(torch, "empty", side_effect=fail_partials):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "FP32 partials"):
                self._call()
        self.assertEqual(len(injected), 1, "scratch allocation did not reach the OOM injector exactly once")
        self._assert_untouched()


if __name__ == "__main__":
    unittest.main()
