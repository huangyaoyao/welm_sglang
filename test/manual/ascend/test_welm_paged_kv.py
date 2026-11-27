"""Exercise paged writes, the native dependency and the actual CANN Flash reader."""
import os
import tempfile
import unittest
from unittest.mock import patch


class WelmPagedKVTest(unittest.TestCase):
    def test_flash_metadata_replay(self):
        for batch in (1, 2, 4, 8):
            with self.subTest(batch=batch):
                self._flash_metadata_replay(batch)

    def test_ep_flash_page_tables(self):
        for batch in (1, 8, 12, 16):
            with self.subTest(batch=batch):
                self._flash_metadata_replay(batch, page_size=128, columns=2048)

    def _flash_metadata_replay(self, batch, page_size=64, columns=4):
        try:
            import torch
            import torch_npu
        except ImportError as exc:
            self.skipTest(str(exc))
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires an A5 NPU")
        from megakernel._fixed import capture_fixed
        from sglang.srt.hardware_backend.npu.graph_runner.welm_paged_kv import prepare_flash_pages
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import decode_tile_scope

        # A strided request map with a short final logical page. Mapping[-1] is the SWA sentinel.
        req_map = torch.zeros(4, 160, dtype=torch.int32, device="npu")[:, :145]
        mapping = torch.arange(513, dtype=torch.int64, device="npu") * 3
        mapping[-1] = -1
        ids, lengths, slots = [torch.zeros(batch, dtype=torch.int32, device="npu") for _ in range(3)]
        def table():
            return torch.zeros((batch + 1, columns), dtype=torch.int32, device="npu")[1:]
        full, swa = table(), table()
        written, qs, ks = (torch.zeros_like(slots) for _ in range(3))
        with torch.inference_mode(), decode_tile_scope(), tempfile.TemporaryDirectory() as build:
            for hybrid in (False, True):
                def forward():
                    enabled = prepare_flash_pages(req_map, ids, lengths, slots, full, qs, ks,
                                                  mapping if hybrid else None,
                                                  swa if hybrid else None, written if hybrid else None,
                                                  page_size=page_size)
                    self.assertTrue(enabled)
                    return (full, qs, ks, swa, written) if hybrid else (full, qs, ks)

                plan = capture_fixed(forward, build, tile_chains=True)
                try:
                    for req, used, loc in (([1, 2, 3, 1, 2, 3, 2, 1], [129, 64, 1, 0, 65, 128, 129, 64],
                                            [257, 67, 128, -1, 64, 256, 0, 65]),
                                           ([2, 3, 2, 3, 1, 2, 3, 2], [70, 145, 1, 64, 145, 65, 129, 1],
                                            [65, -1, 66, 67, 256, 64, 65, 257]),
                                           ([999, 1, 2, 999, 1, 999, 3, 999], [0, 1, 130, 0, 64, 0, 128, 0],
                                            [-1, 65, 67, -1, 128, -1, 65, -1])):
                        req, used, loc = (req * 2)[:batch], (used * 2)[:batch], (loc * 2)[:batch]
                        ids.copy_(torch.tensor(req))
                        lengths.copy_(torch.tensor(used))
                        slots.copy_(torch.tensor(loc))
                        ref_map = torch.zeros((4, 145), dtype=torch.int32)
                        ref_map[1, ::64] = torch.tensor([64, 192, 256])
                        ref_map[2, ::64] = torch.tensor([320, -1, 0])
                        ref_map[3, ::64] = torch.tensor([384, 64, 128])
                        req_map.copy_(ref_map)
                        results = plan.replay()
                        expected_full = torch.zeros((batch, columns), dtype=torch.int32)
                        expected_swa = expected_full.clone()
                        map_cpu = mapping.cpu()
                        for row in range(batch):
                            for column in range((used[row] + page_size - 1) // page_size):
                                physical = int(ref_map[req[row], page_size * column])
                                expected_full[row, column] = physical // page_size
                                expected_swa[row, column] = map_cpu[physical] // page_size
                        expected = [expected_full, (lengths.cpu() > 0).int(), lengths.cpu()]
                        if hybrid:
                            dest = [map_cpu[slot].item() if n else 0 for slot, n in zip(loc, used)]
                            expected.extend([expected_swa, torch.tensor(dest, dtype=written.dtype)])
                        for actual, ref in zip(results, expected):
                            torch.testing.assert_close(actual.cpu(), ref, rtol=0, atol=0)
                finally:
                    plan.reset()

    def test_cross_page_replay(self):
        for batch in (1, 2, 4, 8):
            with self.subTest(batch=batch):
                self._cross_page_replay(batch)

    def _cross_page_replay(self, batch):
        try:
            import torch
            import torch_npu
        except ImportError as exc:
            self.skipTest(str(exc))
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires an A5 NPU")
        import cann_ops_transformer as cann

        from megakernel._fixed import capture_fixed
        from sglang.srt.hardware_backend.npu.graph_runner.welm_paged_kv import stage_flash_kv
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import decode_tile_scope

        rng = torch.Generator().manual_seed(241)
        def bf16(*shape):
            return torch.randn(shape, generator=rng, dtype=torch.bfloat16).npu()

        # Match the pool's flat ND storage, keeping a nonzero layer offset.
        page_elements = 64 * 2 * 256
        keys, values = (bf16(10 * page_elements)[page_elements:].view(9, 64, 2, 256)
                        for _ in range(2))
        for cache in (keys, values):
            self.assertIn(torch_npu.get_npu_format(cache), (0, 2))
        k, v = bf16(batch, 2, 256), bf16(batch, 4, 256)[:, 2:]
        slots = torch.zeros(batch, device="npu", dtype=torch.int64)
        table = torch.zeros((batch, 4), device="npu", dtype=torch.int32)
        lengths = torch.zeros(batch, device="npu", dtype=torch.int32)
        workspace = {}
        def stage():
            return stage_flash_kv(k, v, slots, keys, values, table, lengths, workspace)

        query = bf16(batch, 24, 256)
        sinks = torch.randn(24, generator=rng).npu()
        cu = torch.arange(batch + 1, device="npu", dtype=torch.int32)
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()
        def attention(cache, window):
            common = dict(cu_seqlens_q=cu, cu_seqlens_kv=None, seqused_q=(lengths > 0).int(),
                          seqused_kv=lengths, max_seqlen_q=1, max_seqlen_kv=-1,
                          mask_mode=4 if window >= 0 else 3, win_left=window,
                          win_right=0 if window >= 0 else -1, layout_q="TND",
                          layout_kv="PA_BBND", layout_out="TND")
            schedule = cann.flash_attn_metadata(24, 2, 256, **common)
            return cann.flash_attn(query, *cache, block_table=table, sinks=sinks,
                                   attn_mask=mask, metadata=schedule, softmax_scale=1 / 16,
                                   return_softmax_lse=False, **common)[0].cpu()

        cases = [
            ([80, 193], [[5, 1, 7, 0], [5, 3, 4, 7]], [81, 130]),
            ([257, 66], [[4, 6, 1, 0], [5, 4, 7, 0]], [132, 66]),
            ([255, 254], [[3, 5, 1, 7], [5, 3, 3, 7]], [64, 128]),
            ([-1, 0], [[7, 0, 1, 3], [3, 7, 7, 7]], [65, 0]),
        ]
        replay_cases = []
        for loc, pages, used in cases:
            # Extra rows update distinct offsets, including another slot on a
            # shared page. Their page order and valid prefix vary per replay.
            loc = (loc + [384, 385, 450, 209, 515, 514])[:batch]
            extra_pages = [[6, 7, 1, 0], [6, 8, 4, 0], [7, 2, 0, 0],
                           [3, 1, 8, 0], [8, 2, 0, 0], [8, 4, 6, 0]]
            if used[0] % 2 == 0:
                extra_pages.reverse()
            pages = (pages + extra_pages)[:batch]
            used = (used + [130, 65, 0, 1, 128, 3])[:batch]
            replay_cases.append((loc, pages, used, None))

        # Match a padded DP rank. Slots point to the last valid token in each
        # request's physical page table; padding rows all write scratch slot 0.
        if batch in (4, 8):
            raw = 3 if batch == 4 else 5
            used = [65, 128, 129, 1, 66][:raw]
            pages = [[1, 3, 0, 0], [4, 3, 0, 0], [2, 1, 7, 0],
                     [2, 0, 0, 0], [6, 5, 0, 0]][:raw]
            loc = [row[(n - 1) // 64] * 64 + (n - 1) % 64 for row, n in zip(pages, used)]
            pad = batch - raw
            replay_cases.append((loc + [0] * pad, pages + [[0] * 4 for _ in range(pad)],
                                 used + [0] * pad, raw))
        self.assertIsNone(stage())
        with torch.inference_mode(), decode_tile_scope(), tempfile.TemporaryDirectory() as build:
            self.assertIsNone(stage_flash_kv(k[:-1], v, slots, keys, values, table, lengths, workspace))
            if batch == 2:
                # TorchNPU versions may label this slice NCDHW or canonicalize it.
                other = bf16(1, 9, 64, 2, 256)[0]
                if torch_npu.get_npu_format(other) not in (0, 2):
                    self.assertIsNone(stage_flash_kv(k, v, slots, other, values, table,
                                                     lengths, workspace))
            self.assertIsNotNone(stage())
            for fine in (False, True):
                with patch.dict(os.environ, MK_NATIVE_TILE_DEPENDENCIES=str(int(fine))):
                    plan = capture_fixed(stage, build, tile_chains=True)
                try:
                    scope, = plan.plans
                    self.assertEqual(plan.host_boundary_count, 0)
                    self.assertEqual(list(scope.tile_dependencies), [(0, 1)] if fine else [])
                    for loc, pages, used, raw in replay_cases:
                        slots.copy_(torch.tensor(loc))
                        table.copy_(torch.tensor(pages))
                        lengths.copy_(torch.tensor(used))
                        k.copy_(bf16(batch, 2, 256))
                        v.copy_(bf16(batch, 2, 256))
                        flash_reference = (keys, values)
                        if raw is not None:
                            # Scatter can process duplicate dummy slots in any order.
                            k[raw:].zero_()
                            v[raw:].zero_()
                            flash_reference = tuple(t.flatten().clone().view_as(t) for t in (keys, values))
                            torch_npu.npu_scatter_pa_kv_cache(
                                k.contiguous(), v.contiguous(), *flash_reference,
                                slots.to(torch.int32).contiguous(), cache_mode="Norm",
                            )
                        gold = [t.cpu() for t in (keys, values)]
                        for row, slot in enumerate(loc):
                            if slot >= 0:
                                for ref, src in zip(gold, (k, v)):
                                    ref[slot // 64, slot % 64].copy_(src[row].cpu())
                        for t in plan.outputs:
                            t.fill_(1234)
                        staged = plan.replay()
                        live = {page for row, length in zip(pages, used)
                                for page in row[:(length + 63) // 64]}
                        for out, cache, ref in zip(staged, (keys, values), gold):
                            torch.testing.assert_close(cache.cpu(), ref, atol=0, rtol=0)
                            actual = out.cpu()
                            for page in range(len(ref)):
                                if page in live:
                                    torch.testing.assert_close(actual[page], ref[page], atol=0, rtol=0)
                                else:
                                    self.assertTrue((actual[page] == 1234).all())
                        if raw is not None:
                            for actual, native_cache in zip((keys, values), flash_reference):
                                # Includes every unwritten slot and the shared scratch slot.
                                torch.testing.assert_close(actual.cpu(), native_cache.cpu(), atol=0, rtol=0)
                        for window in (-1, 31):
                            actual = attention(staged, window)
                            expected = attention(flash_reference, window)
                            if raw is not None:
                                actual, expected = actual[:raw], expected[:raw]
                            self.assertTrue(torch.isfinite(expected).all())
                            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                finally:
                    plan.reset()


if __name__ == "__main__":
    unittest.main()
