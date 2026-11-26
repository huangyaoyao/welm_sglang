"""Single-plan A5 probe of production decode boundaries. Run once per --batch."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torch_npu

from megakernel._fixed import capture_fixed, prepare_static
from megakernel.triton import prepare_bindings
from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend, ForwardMetadata
from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import (
    decode_tile_scope, install_welm_tile_layouts, welm_native_region_active,
)
from sglang.srt.layers.welmv4_npu_op import welmv4_oe_concat_4way


def run(batch, directory):
    import cann_ops_transformer as cann

    torch.npu.set_device(0)
    assert torch.npu.get_device_name().startswith("Ascend950")
    install_welm_tile_layouts()
    generator = torch.Generator().manual_seed(131)
    device = torch.device("npu")
    def bf16(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.bfloat16).to(device)

    def i32(*shape):
        return torch.zeros(shape, dtype=torch.int32, device=device)

    # Identity Full pages and a different, in-bounds SWA permutation. Execute
    # the original int64 -> int32 SWA conversion as in the real allocator.
    context, pages = 512, 24
    req_map = torch.zeros((4, context), dtype=torch.int32)
    for req in (1, 2, 3):
        req_map[req, ::64] = 64 * ((torch.arange(8) + 3 * req) % pages)
    mapping = ((torch.arange(pages * 64) + 64) % (pages * 64)).long().to(device)
    req_map = req_map.to(device)
    starts = torch.arange(8, device=device, dtype=torch.int64) * 64
    lengths = torch.ones(batch, dtype=torch.int64, device=device)
    reqs = torch.ones_like(lengths)
    batch_view = SimpleNamespace(seq_lens=lengths, req_pool_indices=reqs)
    query, key, value = bf16(batch, 24, 256), bf16(pages, 64, 2, 256), bf16(pages, 64, 2, 256)
    banks = [bf16(23, 512) for _ in range(4)]
    ids = torch.zeros(4, batch, dtype=torch.int64, device=device)
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).to(device)
    sinks = torch.linspace(-1, 1, 24, device=device, dtype=torch.float32)
    layers = [SimpleNamespace(layer_id=i, tp_q_head_num=24, tp_k_head_num=2, tp_v_head_num=2,
                              qk_head_dim=256, v_head_dim=256, scaling=1 / 16, sliding_window_size=w)
              for i, w in enumerate((-1, 127))]

    def backend():
        b = object.__new__(AscendAttnBackend)
        b.graph_mode = b.is_hybrid_swa = b.use_welm_flash_attn = True
        b._is_swa_layer = b._has_layerwise_sliding_window = lambda layer: layer.sliding_window_size >= 0
        b.page_size, b.req_to_token, b.full_to_swa_index_mapping = 64, req_map, mapping
        b.welm_flash_attn_mask, b._welm_flash_attn, b._welm_flash_attn_metadata = mask, cann.flash_attn, cann.flash_attn_metadata
        b.graph_metadata = dict(welm_decode_page_starts=starts, welm_decode_page_in_bounds=starts < context)
        b.forward_metadata = ForwardMetadata(
            welm_decode_metadata_in_graph=True, welm_flash_seqused_q=i32(batch),
            welm_flash_seqused_kv=i32(batch), block_tables=i32(batch, 8), block_tables_swa=i32(batch, 8),
            welm_flash_cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32, device=device),
            welm_flash_max_seqlen_q=1,
        )
        return b

    eager, candidate = backend(), backend()

    def forward(b):
        b.init_forward_metadata_in_graph(batch_view)
        concatenated = welmv4_oe_concat_4way([F.embedding(ids[i], bank) for i, bank in enumerate(banks)])
        outputs = [b._forward_welm_flash_attention(query, key, value, layer, sinks) for layer in layers]
        m = b.forward_metadata
        return (concatenated, m.welm_flash_seqused_q, m.welm_flash_seqused_kv, m.block_tables,
                m.block_tables_swa, *outputs)

    with torch.inference_mode():
        with decode_tile_scope(native_ops_only=True, batch=batch):
            regions = {r for r in ("metadata", "pages", "oe") if welm_native_region_active(r)}
            fn = lambda: forward(candidate)
            for _ in range(2):
                fn()
            torch.npu.synchronize()
            with prepare_bindings():
                prepare_static(fn, directory)
            torch.npu.synchronize()
            plan = capture_fixed(fn, directory, prepared=True, tile_chains=True, retain_intermediates=True)

        try:
            originals = [str(call.function) for call in plan.graph.calls]
            assert originals.count("aten.embedding.default") == 4, originals
            print("regions", sorted(regions), "physical_scopes", plan.scope_count, flush=True)
            print("execution", plan.execution_summary, flush=True)
            names = [n for p in plan.plans for n in p.kernel_names]
            if "oe" in regions:
                assert any("_welmv4_oe_concat_4way_kernel" in name for name in names), names
            if "metadata" in regions:
                recorded = {r["operation"]: r["calls"] for r in plan.execution_summary["recorded_native_steps"]}
                assert recorded.get("sglang.welm_decode_flash_metadata.default") == 1, recorded
                assert recorded.get("cann_ops_transformer.flash_attn_metadata.default") == 1, recorded
                assert plan.host_boundary_count == 0, plan.execution_summary
                order = [str(step["host_call"].call.function)
                         if isinstance(step, dict) and "host_call" in step else "device-plan"
                         for step in plan.steps]
                assert order[:2] == ["sglang.welm_decode_flash_metadata.default",
                                     "cann_ops_transformer.flash_attn_metadata.default"], order
            for step in range(4):
                n = [[1, 64], [65, 511], [129, 0], [17, 255]][step]
                live = [n[i % len(n)] for i in range(batch)]
                lengths.copy_(torch.tensor(live, dtype=torch.int64))
                reqs.copy_(torch.tensor([1 + (step + i) % 3 if length else 777
                                         for i, length in enumerate(live)]))
                ids.copy_((torch.arange(4 * batch).view(4, batch) + step * 3) % 23)
                reference = [x.cpu() for x in forward(eager)]
                result = plan.replay()
                for actual, expected in zip(result, reference, strict=True):
                    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
            print("PASS", batch, flush=True)
        finally:
            plan.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    args.build.mkdir(parents=True, exist_ok=True)
    run(args.batch, args.build)
