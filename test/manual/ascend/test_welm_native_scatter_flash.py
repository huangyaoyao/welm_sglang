"""A5 check of physical KV page aliases with the intact CANN scatter and Flash operators."""
import unittest

from inspect_welm_kv_flash import scatter_input_rows


class NativeScatterFlashTest(unittest.TestCase):
    def test_changed_shared_physical_pages(self):
        try:
            import torch
            import torch_npu
            import cann_ops_transformer as cann
        except ImportError as exc:
            self.skipTest(str(exc))
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires A5")
        n, pages, width = 4, 12, 64
        def zeros(*sizes):
            return torch.zeros(sizes, dtype=torch.bfloat16, device="npu")
        def indices(values):
            return torch.tensor(values, dtype=torch.int32, device="npu")
        q, key_rows, value_rows = zeros(n, 24, 256), zeros(n, 2, 256), zeros(n, 2, 256)
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()
        lengths = [65, 65, 1, 0]
        for window in (-1, 511):
            common = dict(cu_seqlens_q=indices(range(n + 1)), cu_seqlens_kv=None,
                          seqused_q=indices([1, 1, 1, 0]), seqused_kv=indices(lengths),
                          max_seqlen_q=1, max_seqlen_kv=-1, mask_mode=3 if window < 0 else 4,
                          win_left=window, win_right=-1 if window < 0 else 0,
                          layout_q="TND", layout_kv="PA_BBND", layout_out="TND")
            schedule = cann.flash_attn_metadata(24, 2, 256, **common)
            for shared in (4, 7):
                # Row 0 appends token 65 at the start of its second page, which is
                # also row 1's prefix. Row 3 is padding and still writes slot 0.
                table = [[2, shared], [shared, 9], [10, 0], [0, 0]]
                slots = [shared * width, 9 * width, 10 * width, 0]
                inputs = scatter_input_rows(slots, table, lengths, pages, width)
                self.assertEqual(inputs, [[0], [0, 1], [2], []])
                device_table = indices(table)
                for writer in range(n):
                    with self.subTest(window=window, shared_page=shared, writer=writer):
                        keys, values = (zeros(pages, width, 2, 256) for _ in range(2))
                        value_rows.zero_()
                        value_rows[writer].fill_(64)
                        torch_npu.npu_scatter_pa_kv_cache(
                            key_rows, value_rows, keys, values, indices(slots), cache_mode="Norm")
                        expected_cache = torch.zeros_like(values)
                        expected_cache.view(-1, 2, 256)[slots[writer]].fill_(64)
                        torch.testing.assert_close(values.cpu(), expected_cache.cpu(), atol=0, rtol=0)
                        out = cann.flash_attn(q, keys, values, block_table=device_table,
                                              attn_mask=mask, sinks=None, metadata=schedule,
                                              softmax_scale=1/16, return_softmax_lse=False, **common)[0]
                        changed = out.reshape(n, -1)[:3].abs().amax(dim=1).cpu().tolist()
                        self.assertEqual([x > 0 for x in changed], [writer in row for row in inputs[:3]])


if __name__ == "__main__":
    unittest.main()
