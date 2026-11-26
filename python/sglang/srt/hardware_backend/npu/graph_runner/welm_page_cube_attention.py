'BF16 Cube QK/PV for fixed WeLM row and page tasks on Ascend 950.'
import torch
import triton
import triton.language as tl
from megakernel.triton import native
from .welm_paged_kv import _write_grouped_kv
from .welm_tile_layouts import record_page_attention_output, welm_decode_tiles_active


@triton.jit
def _cann_round_bf16(value):
    # Flash CAST_ROUND rounds BF16 midpoints away from zero.
    bits = value.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x8000) & 0xFFFF0000
    rounded = tl.where(value != value, (bits | 0x00400000) & 0xFFFF0000, rounded)
    return rounded.to(tl.float32, bitcast=True)


def _qk_access(q, keys, scores, rows, pages):
    return dict(
        count=pages,
        block=1,
        reads=[(keys, 64 * 512 * 2)],
        writes=[(scores, rows * 2 * 16 * 64 * 4)],
        shared_reads=[q],
        keep_cube_width=True,
        allow_subviews=True,
    )


@native(core='cube', enabled=welm_decode_tiles_active, tile_layout=_qk_access)
@triton.jit
def _welm_paged_qk_cube(q, keys, scores, rows: tl.constexpr, pages: tl.constexpr):
    # The whole program reads one physical page, including all rows sharing it.
    page = tl.program_id(0)
    for row_head in range(rows * 2):
        row = row_head // 2
        kv_head = row_head % 2
        task = page * rows * 2 + row_head
        lhs = tl.make_block_ptr(q + (row * 2 + kv_head) * 12 * 256, (12, 256), (256, 1), (0, 0), (16, 128), (1, 0))
        rhs = tl.make_block_ptr(keys + (page * 64 * 2 + kv_head) * 256, (64, 256), (512, 1), (0, 0), (64, 128), (1, 0))
        acc = tl.full((16, 64), 0, tl.float32)
        for k in range(2):
            acc = tl.dot(tl.load(lhs, boundary_check=(0,), padding_option='zero'), tl.trans(tl.load(rhs)), acc)
            lhs = tl.advance(lhs, (0, 128))
            rhs = tl.advance(rhs, (0, 128))
        out = tl.make_block_ptr(scores + task * 16 * 64, (16, 64), (64, 1), (0, 0), (16, 64), (1, 0))
        tl.store(out, acc)


def _prob_access(
    scores, table, lengths, sinks, probabilities, stats, rows, pages, columns, blocks, window,
    scale,
):
    return dict(
        count=rows * 2,
        block=1,
        reads=[],
        writes=[(probabilities, blocks * 16 * 256 * 2), (stats, blocks * 16 * 16 * 4)],
        shared_reads=[scores, table, lengths] + ([sinks] if sinks is not None else []),
        allow_subviews=True,
    )


@native(enabled=welm_decode_tiles_active, tile_layout=_prob_access)
@triton.jit
def _welm_paged_softmax(
    scores, table, lengths, sinks, probabilities, stats, rows: tl.constexpr,
    pages: tl.constexpr, columns: tl.constexpr, blocks: tl.constexpr, window: tl.constexpr,
    scale: tl.constexpr,
):
    row_head = tl.program_id(0)
    row = row_head // 2
    kv_head = row_head % 2
    lane = tl.arange(0, 64)
    stat = tl.arange(0, 16)
    length = tl.load(lengths + row)
    start = 0
    if window >= 0:
        start = tl.maximum(0, length - 1 - window)
    # The native decode shape uses S2=256 and carries the running maximum.
    # P x V takes BF16 weights while the denominator retains FP32 weights.
    for head in range(16):
        sink = -float('inf')
        if sinks is not None:
            sink = tl.load(sinks + kv_head * 12 + head, mask=head < 12, other=-float('inf'))
        maximum = sink
        for block in range(blocks):
            task = row_head * blocks + block
            for page_lane in tl.static_range(4):
                column = block * 4 + page_lane
                page = tl.load(table + row * columns + column, mask=column < columns, other=-1)
                physical = tl.where((page >= 0) & (page < pages), page, 0)
                position = column * 64 + lane
                valid = (page >= 0) & (page < pages) & (position >= start) & (position < length)
                logits = tl.where(valid, tl.load(scores + ((physical * rows * 2 + row_head) * 16 + head) * 64 + lane) * scale, -float('inf'))
                maximum = tl.maximum(maximum, tl.max(logits, 0))
            safe_max = tl.where(maximum == -float('inf'), 0.0, maximum)
            weights0 = tl.full((64,), 0, tl.float32)
            weights1 = tl.full((64,), 0, tl.float32)
            weights2 = tl.full((64,), 0, tl.float32)
            weights3 = tl.full((64,), 0, tl.float32)
            for page_lane in tl.static_range(4):
                column = block * 4 + page_lane
                page = tl.load(table + row * columns + column, mask=column < columns, other=-1)
                physical = tl.where((page >= 0) & (page < pages), page, 0)
                position = column * 64 + lane
                valid = (page >= 0) & (page < pages) & (position >= start) & (position < length)
                logits = tl.where(valid, tl.load(scores + ((physical * rows * 2 + row_head) * 16 + head) * 64 + lane) * scale, -float('inf'))
                weights = tl.exp(logits - safe_max)
                if page_lane == 0:
                    weights0 = weights
                elif page_lane == 1:
                    weights1 = weights
                elif page_lane == 2:
                    weights2 = weights
                else:
                    weights3 = weights
                tl.store(probabilities + (task * 16 + head) * 256 + page_lane * 64 + lane, _cann_round_bf16(weights).to(tl.bfloat16))
            # The CANN VF combines even and odd exponents before reducing its
            # 64 FP32 lanes. A separate sum for each physical page changes ties.
            if length - block * 256 <= 64:
                mass = tl.sum(weights0, 0)
            else:
                even0, odd0 = tl.split(tl.reshape(weights0, (32, 2)))
                even1, odd1 = tl.split(tl.reshape(weights1, (32, 2)))
                even2, odd2 = tl.split(tl.reshape(weights2, (32, 2)))
                even3, odd3 = tl.split(tl.reshape(weights3, (32, 2)))
                low = (even0 + odd0) + (even2 + odd2)
                high = (even1 + odd1) + (even3 + odd3)
                lanes = tl.reshape(tl.trans(tl.join(low, high)), (64,))
                mass = tl.sum(lanes, 0)
            if sinks is not None:
                if block == 0:
                    mass += tl.where(sink == float('inf'), 1.0, tl.exp(sink - safe_max))
            tl.store(stats + (task * 16 + head) * 16 + stat, tl.where(stat < 8, maximum, mass))


def _pack_access(values, table, lengths, packed, rows, pages, columns, blocks, window):
    return dict(
        count=rows * 2 * blocks,
        block=1,
        reads=[],
        writes=[(packed, 256 * 256 * 2)],
        shared_reads=[values, table, lengths],
        allow_subviews=True,
    )


@native(enabled=welm_decode_tiles_active, tile_layout=_pack_access)
@triton.jit
def _welm_paged_value_pack(
    values, table, lengths, packed, rows: tl.constexpr, pages: tl.constexpr,
    columns: tl.constexpr, blocks: tl.constexpr, window: tl.constexpr,
):
    task = tl.program_id(0)
    row = task // blocks // 2
    kv_head = task // blocks % 2
    block = task % blocks
    length = tl.load(lengths + row)
    start = 0
    if window >= 0:
        start = tl.maximum(0, length - 1 - window)
    channel = tl.arange(0, 256)
    lane = tl.arange(0, 8)
    # Mask before PV so a NaN in an unreferenced slot cannot enter a Cube dot.
    for half in range(4):
        column = block * 4 + half
        page_lane = tl.load(table + row * columns + column, mask=column < columns, other=-1)
        valid = (page_lane >= 0) & (page_lane < pages)
        physical = tl.where(valid, page_lane, 0)
        for chunk in range(8):
            token = chunk * 8 + lane
            position = column * 64 + token
            mask = valid & (position >= start) & (position < length)
            src = values + ((physical * 64 + token[:, None]) * 2 + kv_head) * 256 + channel[None, :]
            val = tl.load(src, mask=mask[:, None], other=0)
            out = packed + (task * 256 + 64 * half + token[:, None]) * 256 + channel[None, :]
            tl.store(out, val)


def _pv_access(probabilities, packed, partials, rows, blocks):
    return dict(
        count=rows * 2 * blocks,
        block=1,
        reads=[(probabilities, 16 * 256 * 2), (packed, 256 * 256 * 2)],
        writes=[(partials, 16 * 256 * 4)],
        keep_cube_width=True,
        allow_subviews=True,
    )


@native(core='cube', enabled=welm_decode_tiles_active, tile_layout=_pv_access)
@triton.jit
def _welm_paged_pv_cube(probabilities, packed, partials, rows: tl.constexpr, blocks: tl.constexpr):
    task = tl.program_id(0)
    lhs = tl.make_block_ptr(probabilities + task * 16 * 256, (16, 256), (256, 1), (0, 0), (16, 16), (1, 0))
    rhs = tl.make_block_ptr(packed + task * 256 * 256, (256, 256), (256, 1), (0, 0), (16, 256), (1, 0))
    acc = tl.full((16, 256), 0, tl.float32)
    for i in range(16):
        acc = tl.dot(tl.load(lhs), tl.load(rhs), acc)
        lhs = tl.advance(lhs, (0, 16))
        rhs = tl.advance(rhs, (16, 0))
    out = tl.make_block_ptr(partials + task * 16 * 256, (16, 256), (256, 1), (0, 0), (16, 256), (1, 0))
    tl.store(out, acc)


def _finish_access(partials, stats, out, rows, blocks):
    return dict(
        count=rows * 2,
        block=1,
        reads=[(partials, blocks * 16 * 256 * 4), (stats, blocks * 16 * 16 * 4)],
        writes=[(out, 12 * 256 * 2)],
        allow_subviews=True,
    )


@native(enabled=welm_decode_tiles_active, tile_layout=_finish_access)
@triton.jit
def _welm_paged_normalize(partials, stats, out, rows: tl.constexpr, blocks: tl.constexpr):
    row_head = tl.program_id(0)
    channel = tl.arange(0, 256)
    for head in range(12):
        maximum = -float('inf')
        mass = 0.0
        total = tl.full((256,), 0, tl.float32)
        for block in range(blocks):
            task = row_head * blocks + block
            local = tl.load(stats + (task * 16 + head) * 16)
            next_max = tl.maximum(maximum, local)
            safe_max = tl.where(next_max == -float('inf'), 0.0, next_max)
            alpha = tl.where((maximum == float('inf')) & (safe_max == float('inf')), 1.0, tl.exp(maximum - safe_max))
            beta = tl.where((local == float('inf')) & (safe_max == float('inf')), 1.0, tl.exp(local - safe_max))
            total = total * alpha + tl.load(partials + (task * 16 + head) * 256 + channel) * beta
            mass = mass * alpha + tl.load(stats + (task * 16 + head) * 16 + 8) * beta
            maximum = next_max
        denom = tl.where(mass > 0, mass, 1.0)
        tl.store(out + (row_head * 12 + head) * 256 + channel, _cann_round_bf16(total / denom).to(tl.bfloat16))


def run_page_cube_attention(
    q, k, v, slots, keys, values, table, lengths, sinks, window, scale, workspace, group,
):
    """Run after the caller has validated all shape, format and indexing contracts."""
    rows = q.shape[0]
    pages = keys.shape[0]
    columns = table.shape[1]
    blocks = triton.cdiv(columns, 4)
    out = q.new_empty((rows * 6144,)).view(rows, 24, 256)
    key = ('page_cube_blocks', q.device, rows, pages, columns)
    # Allocate every scratch tensor before either publishing it or mutating the KV cache.
    if key not in workspace:
        tasks = rows * 2 * blocks
        workspace[key] = (torch.empty(pages * rows * 2 * 16 * 64, device=q.device, dtype=torch.float32), torch.empty(tasks * 16 * 256, device=q.device, dtype=torch.bfloat16), torch.empty(tasks * 16 * 16, device=q.device, dtype=torch.float32), torch.empty(tasks * 16 * 256, device=q.device, dtype=torch.float32), torch.empty(tasks * 256 * 256, device=q.device, dtype=torch.bfloat16))
    logits, probabilities, stats, partials, packed = workspace[key]
    _write_grouped_kv(k, v, slots, keys, values, group)
    _welm_paged_qk_cube[pages,](q, keys, logits, rows, pages)
    _welm_paged_softmax[rows * 2,](logits, table, lengths, sinks, probabilities, stats, rows, pages, columns, blocks, window, float(scale))
    _welm_paged_value_pack[rows * 2 * blocks,](values, table, lengths, packed, rows, pages, columns, blocks, window)
    _welm_paged_pv_cube[rows * 2 * blocks,](probabilities, packed, partials, rows, blocks)
    _welm_paged_normalize[rows * 2,](partials, stats, out, rows, blocks)
    record_page_attention_output(out)
    return out
