"""Replay grouped KV writes and CANN Full/SWA Flash on the same persistent cache."""
import tempfile
import unittest


class WelmFlashDirectReplayTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            import torch_npu
        except ImportError as exc:
            self.skipTest(str(exc))
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires an A5 NPU")
        self.torch, self.torch_npu = torch, torch_npu

    def _forward_schedule_fields(self, schedule):
        # A5 flash_attn_tiling_data.h: a 16-word header followed by per-core
        # FA and FD sections. Ignore alignment, reserved words and grad data.
        torch = self.torch
        self.assertEqual(schedule.dtype, torch.int32, "unknown CANN metadata dtype")
        self.assertEqual(schedule.ndim, 1, "unknown CANN metadata layout")
        words = schedule.cpu()
        stride = 16
        self.assertGreaterEqual(words.numel(), stride)
        header = words[[0, 1, 4, 5]]
        sections, has_fd, cubes, vectors = header.tolist()
        self.assertTrue(sections > 0 and has_fd in (0, 1) and cubes > 0 and vectors > 0,
                        f"unrecognized CANN metadata header {words[:stride].tolist()}")
        self.assertEqual(vectors % cubes, 0, header)
        end_fa = stride * (1 + sections * cubes)
        end_fd = end_fa + stride * sections * vectors
        self.assertLessEqual(end_fd, words.numel(), f"truncated CANN descriptor {header.tolist()}")
        fa = words[stride:end_fa].view(-1, stride)
        # DN/ND read start/end BN, M, S2, and the first FD workspace index.
        fields = [header, fa[:, :7].reshape(-1)]
        if has_fd:
            fd = words[end_fa:end_fd].view(-1, stride)
            # mLen == 0 disables a Vector record. Only live records' other
            # words are consumed by GetFDSectionInfo in flash_attn_kernel_dn.h.
            m_len = fd[:, 5]
            self.assertTrue((m_len >= 0).all())
            self.assertTrue((m_len > 0).any(), header)
            fields.extend([m_len, fd[m_len > 0, :5].reshape(-1)])
        return torch.cat(fields)

    def test_direct_cache_and_flash_replay(self):
        for batch in (1, 2, 4, 8):
            for window in (-1, 511):
                with self.subTest(batch=batch, window=window):
                    self._check(batch, window)

    def _check(self, batch, window):
        import cann_ops_transformer as cann
        from megakernel._fixed import capture_fixed
        from sglang.srt.hardware_backend.npu.graph_runner.welm_paged_kv import stage_flash_kv
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import decode_tile_scope

        torch, torch_npu = self.torch, self.torch_npu
        rng = torch.Generator().manual_seed(313)
        pages, columns = 32, 20
        capacity = pages * 64 * 2 * 256

        def bf16(*shape):
            return torch.randn(shape, generator=rng, dtype=torch.bfloat16).npu()

        # The native and MK writes operate on distinct flat ND allocations.
        seed = tuple(bf16(capacity).view(pages, 64, 2, 256) for _ in range(2))
        def new_pool():
            return tuple(t.flatten().clone().view_as(t) for t in seed)

        cache, native_cache = new_pool(), new_pool()
        k, v = bf16(batch, 2, 256), bf16(batch, 4, 256)[:, 2:]
        query = bf16(batch, 24, 256)
        slots = torch.zeros(batch, dtype=torch.int32, device="npu")
        table = torch.zeros((batch, columns), dtype=torch.int32, device="npu")
        lengths = torch.ones_like(slots)
        q_used = torch.ones_like(slots)
        cu = torch.arange(batch + 1, dtype=torch.int32, device="npu")
        sinks = torch.linspace(-1, 1, 24, device="npu", dtype=torch.float32)
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()
        fixed_output = torch.empty_like(query)
        workspace = {}
        common = dict(cu_seqlens_q=cu, cu_seqlens_kv=None, seqused_q=q_used,
                      seqused_kv=lengths, max_seqlen_q=1, max_seqlen_kv=-1,
                      mask_mode=4 if window >= 0 else 3, win_left=window,
                      win_right=0 if window >= 0 else -1, layout_q="TND",
                      layout_kv="PA_BBND", layout_out="TND")

        def metadata():
            return cann.flash_attn_metadata(24, 2, 256, **common)

        def flash(pool, schedule):
            return cann.flash_attn(
                query, *pool, block_table=table, sinks=sinks, attn_mask=mask,
                metadata=schedule, softmax_scale=1 / 16, return_softmax_lse=False, **common,
            )[0]

        def write():
            return stage_flash_kv(k, v, slots, *cache, table, lengths, workspace, copy_pages=False)

        def forward():
            persistent = write()
            schedule = metadata()
            fixed_output.copy_(flash(persistent, schedule).view_as(fixed_output))
            return fixed_output, schedule

        # Long, short and partially active batches use the same storage. The
        # last valid page in each row contains the key just written this step.
        active = {1: 1, 2: 1, 4: 3, 8: 5}[batch]
        cases = [(batch, (1, 64)), (batch, (1025, 513)), (active, (65, 193)), (batch, (511, 512))]
        with torch.inference_mode(), decode_tile_scope(), tempfile.TemporaryDirectory() as build:
            persistent = write()
            self.assertIsNotNone(persistent)
            self.assertEqual([t.data_ptr() for t in persistent], [t.data_ptr() for t in cache])
            self.assertFalse(workspace)
            plan = capture_fixed(forward, build, tile_chains=True)
            try:
                kernels = [name for scope in plan.plans for name in scope.kernel_names]
                self.assertIn("_write_kv_pages", kernels)
                self.assertNotIn("_read_flash_pages", kernels)
                # A runtime may keep CANN metadata as a Torch call or observe
                # its native launches. A one-row schedule can stay identical
                # across lengths, so also verify its producer survived capture.
                schedule_op, = (call for call in plan.graph.calls if
                                str(call.function) == "cann_ops_transformer.flash_attn_metadata.default")
                host_steps = [s["host_call"] for s in plan.steps if isinstance(s, dict) and "host_call" in s]
                if not any(host.call is schedule_op for host in host_steps):
                    launches = range(*schedule_op.observation)
                    self.assertTrue(launches, "metadata has no replay producer")
                    scoped = {i for scope in plan.plans for i in scope.task_indices}
                    native = {id(s) for s in plan.steps if isinstance(s, dict) and "native" in s}
                    self.assertTrue(all(i in scoped or id(plan.tasks[i]) in native for i in launches))
                for step, (raw, span) in enumerate(cases):
                    used = [span[i % 2] if i < raw else 0 for i in range(batch)]
                    block_table = torch.zeros(batch, columns, dtype=torch.int32)
                    loc = torch.zeros(batch, dtype=torch.int32)
                    for row, length in enumerate(used[:raw]):
                        count = (length + 63) // 64
                        for col in range(count):
                            block_table[row, col] = 1 + (3 * step + 7 * col) % 20
                        # Distinct writable pages, shared history, and changed page order.
                        block_table[row, count - 1] = 21 + (step + row) % 8
                        loc[row] = block_table[row, count - 1] * 64 + (length - 1) % 64
                    slots.copy_(loc)
                    lengths.copy_(torch.tensor(used))
                    q_used.copy_(torch.tensor([1] * raw + [0] * (batch - raw)))
                    table.copy_(block_table)
                    query.copy_(bf16(*query.shape))
                    k.copy_(bf16(*k.shape))
                    v.copy_(bf16(*v.shape))
                    # Every dummy row targets slot 0, whose write order is unspecified.
                    if raw < batch:
                        k[raw:].zero_()
                        v[raw:].zero_()
                    for dest, ref, initial in zip(cache, native_cache, seed):
                        dest.copy_(initial)
                        ref.copy_(initial)
                    torch_npu.npu_scatter_pa_kv_cache(
                        k.contiguous(), v.contiguous(), *native_cache, slots, cache_mode="Norm",
                    )
                    expected_schedule = metadata()
                    expected_out = flash(native_cache, expected_schedule).cpu().view(query.shape)
                    expected = self._forward_schedule_fields(expected_schedule)
                    # Confirm the installed eager operator agrees on all fields
                    # used by this ABI before interpreting fixed replay results.
                    torch.testing.assert_close(self._forward_schedule_fields(metadata()),
                                               expected, atol=0, rtol=0)
                    fixed_output.fill_(float("nan"))
                    actual, captured_schedule = plan.replay()
                    self.assertEqual(actual.data_ptr(), fixed_output.data_ptr())
                    torch.testing.assert_close(actual.cpu()[:raw], expected_out[:raw], atol=0, rtol=0)
                    self.assertTrue(torch.isfinite(expected_out[:raw]).all())
                    for got, ref in zip(cache, native_cache):
                        torch.testing.assert_close(got.cpu(), ref.cpu(), atol=0, rtol=0)
                    observed = self._forward_schedule_fields(captured_schedule)
                    torch.testing.assert_close(observed, expected, atol=0, rtol=0)
                self.assertFalse(workspace)
            finally:
                plan.reset()


if __name__ == "__main__":
    unittest.main()
