"""Decode attention whose KV readers share the cache writer's physical page partition."""
import math

import torch
import triton
import triton.language as tl

from megakernel.triton import native
from sglang.srt.utils import get_bool_env_var

from .welm_paged_kv import _validated_page_group, _write_grouped_kv
from .welm_tile_layouts import record_page_attention_output, welm_decode_tiles_active, welm_model_row_tiles_active


def _partial_layout(q, keys, values, table, lengths, partials, rows, pages, group,
                    columns, window, scale):
    page_bytes = 64 * 2 * 256 * keys.element_size()
    return dict(count=pages, block=group, reads=[(keys, page_bytes), (values, page_bytes)],
                writes=[], shared_reads=[q, table, lengths], whole_writes=[partials],
                allow_subviews=True)


@native(enabled=welm_decode_tiles_active, tile_layout=_partial_layout)
@triton.jit
def _page_attention_partials(q, keys, values, table, lengths, partials, rows: tl.constexpr,
                             pages: tl.constexpr, group: tl.constexpr, columns: tl.constexpr,
                             window: tl.constexpr, scale: tl.constexpr):
    owner = tl.program_id(0)
    first = owner * group
    dim = tl.arange(0, 256)
    token = tl.arange(0, 8)
    stat = tl.arange(0, 16)
    for task in range(rows * 24):
        row = task // 24
        kv_head = (task % 24) // 12
        query = tl.load(q + task * 256 + dim).to(tl.float32)
        length = tl.load(lengths + row).to(tl.int32)
        begin = 0
        if window >= 0:
            begin = tl.maximum(0, length - 1 - window)
        maximum = -float("inf")
        mass = 0.0
        numerator = tl.full((256,), 0, tl.float32)
        for logical in range(begin // 64, tl.minimum(tl.cdiv(length, 64), columns)):
            page = tl.load(table + row * columns + logical)
            owned = (page >= first) & (page < first + group) & (page < pages)
            if owned:
                # Even a predicated address belongs to this owner's cache region.
                physical = tl.where(owned, page, first)
                for chunk in range(8):
                    pos = logical * 64 + chunk * 8 + token
                    valid = (pos >= begin) & (pos < length)
                    slot = physical * 64 + chunk * 8 + token
                    offsets = (slot[:, None] * 2 + kv_head) * 256 + dim[None, :]
                    k = tl.load(keys + offsets, mask=valid[:, None], other=0).to(tl.float32)
                    v = tl.load(values + offsets, mask=valid[:, None], other=0).to(tl.float32)
                    logits = tl.sum(k * query[None, :], 1) * scale
                    logits = tl.where(valid, logits, -float("inf"))
                    next_max = tl.maximum(maximum, tl.max(logits, 0))
                    safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
                    alpha = tl.exp(maximum - safe_max)
                    weights = tl.exp(logits - safe_max)
                    numerator = numerator * alpha + tl.sum(weights[:, None] * v, 0)
                    mass = mass * alpha + tl.sum(weights, 0)
                    maximum = next_max
        # Store complete aligned vectors. The last 16 FP32 lanes hold m and l.
        dest = partials + (owner * rows * 24 + task) * 272
        tl.store(dest + dim, numerator)
        tl.store(dest + 256 + stat, tl.where(stat < 8, maximum, mass))


def _merge_layout(partials, sinks, out, rows, groups, has_sinks):
    return dict(count=rows * 2, block=1, reads=[], writes=[(out, 12 * 256 * out.element_size())],
                shared_reads=[partials, sinks] if has_sinks else [partials],
                allow_subviews=True)


@native(enabled=welm_decode_tiles_active, tile_layout=_merge_layout)
@triton.jit
def _merge_page_attention(partials, sinks, out, rows: tl.constexpr, groups: tl.constexpr,
                          has_sinks: tl.constexpr):
    # One program owns twelve query heads (one KV head). Every partial is ready.
    first_head = tl.program_id(0) * 12
    dim = tl.arange(0, 256)
    for head in range(12):
        task = first_head + head
        maximum = -float("inf")
        mass = 0.0
        numerator = tl.full((256,), 0, tl.float32)
        for owner in range(groups):
            data = partials + (owner * rows * 24 + task) * 272
            local_max = tl.load(data + 256)
            local_mass = tl.load(data + 264)
            next_max = tl.maximum(maximum, local_max)
            safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            alpha = tl.exp(maximum - safe_max)
            beta = tl.exp(local_max - safe_max)
            numerator = numerator * alpha + tl.load(data + dim) * beta
            mass = mass * alpha + local_mass * beta
            maximum = next_max
        empty = mass == 0.0
        if has_sinks:
            # Add the zero-value sink once, after combining all actual tokens.
            sink = tl.load(sinks + task % 24).to(tl.float32)
            combined = tl.maximum(maximum, sink)
            safe_max = tl.where(combined == -float("inf"), 0.0, combined)
            weight = tl.exp(maximum - safe_max)
            sink_weight = tl.where(sink == float("inf"), 1.0, tl.exp(sink - safe_max))
            numerator *= weight
            mass = mass * weight + sink_weight
        denom = tl.where(empty, 1.0, mass)
        result = tl.where(empty, 0.0, numerator / denom)
        tl.store(out + task * 256 + dim, result.to(tl.bfloat16))


def try_page_attention(q, k, v, slots, keys, values, table, lengths, sinks, window, scale,
                       workspace):
    """Run ordinary BF16 GQA on supported pools, or decline before any KV write."""
    if not welm_decode_tiles_active() or workspace is None or q.ndim not in (2, 3):
        return None
    rows = q.shape[0]
    if (q.shape[1:] not in ((6144,), (24, 256)) or rows not in (1, 2, 4, 8)
            or q.dtype != torch.bfloat16 or q.device.type != "npu" or q.device != keys.device
            or not q.is_contiguous() or k.ndim != 3 or k.shape[0] != rows
            or not isinstance(window, int) or window < -1
            or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0):
        return None
    import torch_npu

    if torch_npu.get_npu_format(q) not in (0, 2):
        return None
    if sinks is not None and (sinks.shape != (24,) or sinks.dtype != torch.float32
                              or sinks.device != q.device or not sinks.is_contiguous()
                              or torch_npu.get_npu_format(sinks) not in (0, 2)):
        return None

    group = _validated_page_group(k, v, slots, keys, values, table, lengths)
    if group is None:
        return None
    if welm_model_row_tiles_active() and get_bool_env_var("SGLANG_NPU_WELMV4_MK_CUBE_PAGE_ATTENTION"):
        from .welm_page_cube_attention import run_page_cube_attention

        return run_page_cube_attention(q, k, v, slots, keys, values, table, lengths, sinks, window,
                                       scale, workspace, group)

    pages = keys.shape[0]
    groups = triton.cdiv(pages, group)
    # All allocations precede mutation of persistent KV. Allocate flat ND
    # storage so the physical order agrees with the kernels' pointer arithmetic.
    out = q.new_empty((rows * 6144,)).view(rows, 24, 256)
    cache_key = ("page_attention", q.device, rows, groups)
    if cache_key not in workspace:
        workspace[cache_key] = torch.empty(groups * rows * 24 * 272, device=q.device,
                                           dtype=torch.float32).view(groups, rows, 24, 272)
    partials = workspace[cache_key]
    _write_grouped_kv(k, v, slots, keys, values, group)
    _page_attention_partials[(groups,)](q, keys, values, table, lengths, partials, rows, pages,
                                        group, table.shape[1], window, float(scale))
    _merge_page_attention[(rows * 2,)](partials, sinks, out, rows, groups, sinks is not None)
    record_page_attention_output(out)
    return out
