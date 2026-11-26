"""Fixed decode Flash parity and dynamic page-table access on one Ascend 950."""
import os
import tempfile
from unittest.mock import patch

from test_welm_page_attention import _A5Test


class WelmPageCubeAttentionTest(_A5Test):
    @patch.dict(os.environ, SGLANG_NPU_WELMV4_MK_MODEL_ROW_TILES="1",
                SGLANG_NPU_WELMV4_MK_CUBE_PAGE_ATTENTION="1",
                MK_NATIVE_TILE_DEPENDENCIES="1")
    def test_native_flash_and_unreferenced_cache(self):
        import cann_ops_transformer as cann
        from megakernel._fixed import capture_fixed
        from sglang.srt.hardware_backend.npu.graph_runner.welm_page_attention import try_page_attention
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import decode_tile_scope

        torch = self.torch
        generator = torch.Generator().manual_seed(8364)
        with torch.inference_mode():
            for batch in (1, 2, 4, 8):
                with self.subTest(batch=batch), decode_tile_scope(), tempfile.TemporaryDirectory() as build:
                    # A fixed table has capacity for nine pages per request. Page zero is unreferenced.
                    columns = 9
                    pages = 1 + columns * batch
                    def random(shape):
                        return torch.randn(shape, dtype=torch.float32, generator=generator).bfloat16()
                    q = random((batch, 24, 256)).npu()
                    source = [random((pages, 64, 2, 256)) for _ in range(2)]
                    pool = [t.npu() for t in source]
                    table = torch.arange(1, pages, dtype=torch.int32).reshape(batch, columns).npu()
                    lengths = torch.empty(batch, dtype=torch.int32, device="npu")
                    slots = torch.empty_like(lengths)
                    k, v = [random((batch, 2, 256)).npu() for _ in range(2)]
                    sinks = torch.linspace(-3., 5., 24, dtype=torch.float32).npu()
                    opts = dict(cu_seqlens_q=torch.arange(batch+1, dtype=torch.int32, device="npu"),
                                cu_seqlens_kv=None, seqused_q=torch.ones_like(lengths), seqused_kv=lengths,
                                max_seqlen_q=1, max_seqlen_kv=-1, mask_mode=3, win_left=-1, win_right=-1,
                                layout_q="TND", layout_kv="PA_BBND", layout_out="TND")
                    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()
                    scratch = {}
                    def forward():
                        return try_page_attention(q, k, v, slots, *pool, table, lengths, sinks, -1, 1/16, scratch)
                    def setup(length):
                        lengths.fill_(length)
                        p = table[:, (length-1)//64]
                        slots.copy_(p*64 + (length-1)%64)
                        for cache, src, update in zip(pool, source, (k, v)):
                            cache.copy_(src)
                            cache[p.long(), (length-1)%64] = update
                    setup(65)
                    plan = capture_fixed(forward, build, tile_chains=True)
                    try:
                        names = [n for scope in plan.plans for n in scope.kernel_names]
                        self.assertCountEqual(names, ["_write_kv_pages", "_welm_paged_qk_cube", "_welm_paged_softmax",
                                                     "_welm_paged_value_pack", "_welm_paged_pv_cube", "_welm_paged_normalize"])
                        for length in (5, 129, 511):
                            setup(length)
                            metadata = cann.flash_attn_metadata(24, 2, 256, **opts)
                            reference, _ = cann.flash_attn(q, *pool, block_table=table, sinks=sinks, attn_mask=mask,
                                                           metadata=metadata, softmax_scale=1/16,
                                                           return_softmax_lse=False, **opts)
                            for cache in pool:
                                cache[0].fill_(float("nan"))
                                for pg in table[:, (length-1)//64].cpu().tolist():
                                    cache[pg, (length-1)%64+1:].fill_(float("nan"))
                            for buffer in scratch[next(k for k in scratch if k[0]=="page_cube_blocks")]:
                                buffer.fill_(float("nan"))
                            output = plan.replay()
                            torch.testing.assert_close(output.cpu().reshape_as(reference), reference.cpu(), atol=0, rtol=0)
                    finally:
                        plan.reset()
