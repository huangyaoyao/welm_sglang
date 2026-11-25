"""Physically partitioned KV writes and paged reads for fixed WeLM decode buckets."""
import torch
import triton
import triton.language as tl

from megakernel.triton import native

from .welm_tile_layouts import welm_decode_tiles_active


_DECODE_BATCH_SIZES = (1, 2, 4, 8)


def _write_layout(k, v, slots, keys, values, k_stride, v_stride, pages, group, rows):
    page_bytes = 64 * 512 * keys.element_size()
    return dict(count=pages, block=group, reads=[], shared_reads=[k, v, slots],
                writes=[(keys, page_bytes), (values, page_bytes)], allow_subviews=True)


@native(enabled=welm_decode_tiles_active, tile_layout=_write_layout)
@triton.jit
def _write_kv_pages(k, v, slots, keys, values, k_stride: tl.constexpr,
                    v_stride: tl.constexpr, pages: tl.constexpr, group: tl.constexpr,
                    rows: tl.constexpr):
    first = tl.program_id(0) * group
    cols = tl.arange(0, 512)
    for row in tl.static_range(rows):
        slot = tl.load(slots + row).to(tl.int64)
        if (slot >= first * 64) & (slot < (first + group) * 64) & (slot < pages * 64):
            tl.store(keys + slot * 512 + cols, tl.load(k + row * k_stride + cols))
            tl.store(values + slot * 512 + cols, tl.load(v + row * v_stride + cols))


def _read_layout(keys, values, table, lengths, flash_k, flash_v, pages, group, columns, rows):
    page_bytes = 64 * 512 * keys.element_size()
    return dict(count=pages, block=group, reads=[(keys, page_bytes), (values, page_bytes)],
                writes=[(flash_k, page_bytes), (flash_v, page_bytes)],
                shared_reads=[table, lengths], allow_subviews=True)


@native(enabled=welm_decode_tiles_active, tile_layout=_read_layout)
@triton.jit
def _read_flash_pages(keys, values, table, lengths, flash_k, flash_v,
                      pages: tl.constexpr, group: tl.constexpr, columns: tl.constexpr,
                      rows: tl.constexpr):
    first = tl.program_id(0) * group
    offsets = tl.arange(0, 2048)
    for row in tl.static_range(rows):
        length = tl.load(lengths + row)
        for logical in range(tl.minimum(tl.cdiv(length, 64), columns)):
            page = tl.load(table + row * columns + logical)
            if (page >= first) & (page < first + group) & (page < pages):
                for j in tl.static_range(16):
                    addr = page * 32768 + j * 2048 + offsets
                    tl.store(flash_k + addr, tl.load(keys + addr))
                    tl.store(flash_v + addr, tl.load(values + addr))


def _validated_page_group(k, v, slots, keys, values, table, lengths):
    """Check all cache/slot operands without allocating or submitting device writes."""
    import torch_npu

    if not welm_decode_tiles_active() or slots is None or table is None or lengths is None:
        return None
    if slots.ndim != 1 or slots.shape[0] not in _DECODE_BATCH_SIZES:
        return None
    rows = slots.shape[0]
    if (lengths.shape != (rows,) or table.ndim != 2
            or table.shape[0] != rows or not 1 <= table.shape[1] <= 64
            or keys.ndim != 4 or keys.shape[1:] != (64, 2, 256)
            or values.shape != keys.shape or not 2 <= keys.shape[0] <= 128
            or slots.dtype not in (torch.int32, torch.int64)
            or table.dtype != torch.int32 or lengths.dtype != torch.int32):
        return None
    device = keys.device
    if (device.type != "npu" or not torch.npu.get_device_name(device).startswith("Ascend950")
            or any(t.device != device for t in (k, v, values, table, lengths, slots))
            or any(t.dtype != torch.bfloat16 for t in (keys, values, k, v))
            or any(not t.is_contiguous() for t in (keys, values, table, lengths, slots))
            or any(t.shape != (rows, 2, 256) or t.stride()[1:] != (256, 1)
                   for t in (k, v))
            or any(torch_npu.get_npu_format(t) not in (0, 2) for t in (keys, values, k, v))):
        return None

    return triton.cdiv(keys.shape[0], 16)


def _write_grouped_kv(k, v, slots, keys, values, group):
    pages = keys.shape[0]
    grid = (triton.cdiv(pages, group),)
    _write_kv_pages[grid](k, v, slots, keys, values, k.stride(0), v.stride(0),
                          pages, group, slots.shape[0])


def stage_flash_kv(k, v, slots, keys, values, table, lengths, workspace, *, copy_pages=True):
    """Write KV and return the staged or persistent Flash operands when supported."""
    group = _validated_page_group(k, v, slots, keys, values, table, lengths)
    if group is None:
        return None
    pages = keys.shape[0]
    if copy_pages:
        # The graph bucket owns this scratch. Equal geometries reuse it across
        # layers; Flash consumes it before the next layer can overwrite it.
        cache_key = (keys.device, pages)
        if cache_key not in workspace:
            workspace[cache_key] = (torch.zeros_like(keys), torch.zeros_like(values))
        flash_k, flash_v = workspace[cache_key]
    _write_grouped_kv(k, v, slots, keys, values, group)
    if not copy_pages:
        return keys, values
    columns = table.shape[1]
    grid = (triton.cdiv(pages, group),)
    _read_flash_pages[grid](keys, values, table, lengths, flash_k, flash_v,
                            pages, group, columns, slots.shape[0])
    return flash_k, flash_v


def _metadata_layout(req_map, requests, lengths, slots, mapping, full_table, swa_table,
                     swa_slots, q_used, kv_used, req_stride, context, map_size, columns,
                     lanes, hybrid):
    reads = [(t, t.element_size()) for t in (requests, lengths, slots)]
    writes = [(q_used, q_used.element_size()), (kv_used, kv_used.element_size()),
              (full_table, columns * full_table.element_size())]
    shared = [req_map]
    if hybrid:
        shared.append(mapping)
        writes.extend([(swa_table, columns * swa_table.element_size()),
                       (swa_slots, swa_slots.element_size())])
    return dict(count=requests.numel(), block=1, reads=reads, writes=writes, shared_reads=shared,
                allow_subviews=True)


@native(enabled=welm_decode_tiles_active, tile_layout=_metadata_layout)
@triton.jit
def _prepare_flash_pages(req_map, requests, lengths, slots, mapping, full_table,
                         swa_table, swa_slots, q_used, kv_used, req_stride: tl.constexpr,
                         context: tl.constexpr, map_size: tl.constexpr, columns: tl.constexpr,
                         lanes: tl.constexpr, hybrid: tl.constexpr):
    row = tl.program_id(0)
    logical = tl.arange(0, lanes)
    pos = logical * 64
    n = tl.load(lengths + row)
    req = tl.load(requests + row).to(tl.int64)
    live = (logical < columns) & (pos < n) & (pos < context)
    # Give masked lanes a valid address even on SIMD targets which load before selecting.
    off = tl.where(live, req * req_stride + pos, 0)
    full = tl.where(live, tl.load(req_map + off), 0).to(tl.int64)
    tl.store(full_table + row * columns + logical, full >> 6, mask=logical < columns)
    if hybrid:
        # The allocator appends a sentinel so tensor index -1 maps to its last element.
        idx = tl.where(full < 0, full + map_size, full)
        swa = tl.load(mapping + idx).to(tl.int64)
        tl.store(swa_table + row * columns + logical, tl.where(live, swa >> 6, 0),
                 mask=logical < columns)
        slot = tl.where(n > 0, tl.load(slots + row), 0).to(tl.int64)
        mapped = tl.load(mapping + tl.where(slot < 0, slot + map_size, slot))
        tl.store(swa_slots + row, tl.where(n > 0, mapped, 0))
    tl.store(q_used + row, (n > 0).to(tl.int32))
    tl.store(kv_used + row, tl.maximum(n, 0))


def prepare_flash_pages(req_map, requests, lengths, slots, full_table, q_used, kv_used,
                        mapping=None, swa_table=None, swa_slots=None):
    if not welm_decode_tiles_active() or req_map.device.type != "npu":
        return False
    hybrid = mapping is not None
    if hybrid != (swa_table is not None) or hybrid != (swa_slots is not None):
        return False
    if requests is None or requests.ndim != 1 or requests.shape[0] not in _DECODE_BATCH_SIZES:
        return False
    rows = requests.shape[0]
    vectors = (requests, lengths, slots, q_used, kv_used)
    if (any(t is None or t.shape != (rows,) for t in vectors)
            or req_map.ndim != 2 or req_map.stride(1) != 1
            or full_table.ndim != 2 or full_table.shape[0] != rows
            or not 1 <= full_table.shape[1] <= 64
            or not torch.npu.get_device_name(req_map.device).startswith("Ascend950")):
        return False
    buffers = (*vectors, full_table)
    if hybrid:
        if mapping.ndim != 1 or swa_table.shape != full_table.shape or swa_slots.shape != (rows,):
            return False
        buffers += (mapping, swa_table, swa_slots)
    if (req_map.dtype not in (torch.int32, torch.int64)
            or any(t.device != req_map.device or not t.is_contiguous()
                   or t.dtype not in (torch.int32, torch.int64) for t in buffers)):
        return False
    columns = full_table.shape[1]
    _prepare_flash_pages[(rows,)](
        req_map, requests, lengths, slots, mapping, full_table, swa_table, swa_slots,
        q_used, kv_used, req_map.stride(0), req_map.shape[1], mapping.numel() if hybrid else 0,
        columns, triton.next_power_of_2(columns), hybrid,
    )
    return True
