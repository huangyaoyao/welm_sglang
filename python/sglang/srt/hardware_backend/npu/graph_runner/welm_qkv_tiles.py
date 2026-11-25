"""BF16 QKV projection for fixed WeLM MK decode buckets."""
import torch
import triton
import triton.language as tl

from megakernel.triton import native

from .welm_tile_layouts import welm_decode_tiles_active


# Each row spans fourteen 512-column tiles. Group contiguous tiles so an
# A5 launch needs at most 28 Cube programs for these decode buckets.
_TILES_PER_PROGRAM = {1: 1, 2: 1, 4: 2, 8: 7}


def _projection_layout(x, weight, packed, tiles_per_program):
    if (x.ndim != 2 or x.shape[1] != 2048 or weight.shape != (7168, 2048)
            or packed.shape != (x.shape[0], 7168)
            or _TILES_PER_PROGRAM.get(x.shape[0]) != tiles_per_program):
        return None
    return dict(count=packed.numel(), block=512 * tiles_per_program, reads=[],
                writes=[(packed, packed.element_size())], shared_reads=[x, weight],
                allow_subviews=True)


@native(core="cube", enabled=welm_decode_tiles_active, tile_layout=_projection_layout)
@triton.jit
def _welm_bf16_qkv_kernel(x, weight, packed, tiles_per_program: tl.constexpr):
    pid = tl.program_id(0)
    programs_per_row: tl.constexpr = 14 // tiles_per_program
    row = pid // programs_per_row
    first_tile = (pid % programs_per_row) * tiles_per_program
    for tile in tl.static_range(tiles_per_program):
        col = (first_tile + tile) * 512
        a = tl.make_block_ptr(x + row * 2048, (1, 2048), (2048, 1), (0, 0), (1, 128), (1, 0))
        b = tl.make_block_ptr(weight + col * 2048, (512, 2048), (2048, 1), (0, 0), (512, 128), (1, 0))
        acc = tl.full((1, 512), 0, tl.float32)
        for _ in range(16):
            acc = tl.dot(tl.load(a), tl.trans(tl.load(b)), acc)
            a = tl.advance(a, (0, 128))
            b = tl.advance(b, (0, 128))
        dst = tl.make_block_ptr(packed + row * 7168 + col, (1, 512), (512, 1), (0, 0), (1, 512), (1, 0))
        tl.store(dst, acc.to(tl.bfloat16))


def _query_layout(packed, query):
    if (packed.ndim != 2 or packed.shape[0] not in _TILES_PER_PROGRAM
            or packed.shape[1] != 7168 or query.shape != (packed.shape[0], 6144)):
        return None
    return dict(count=packed.shape[0], block=1, reads=[(packed, 7168 * packed.element_size())],
                writes=[(query, 6144 * query.element_size())], allow_subviews=True)


@native(enabled=welm_decode_tiles_active, tile_layout=_query_layout)
@triton.jit
def _welm_pack_query_kernel(packed, query):
    row = tl.program_id(0)
    offs = tl.arange(0, 2048)
    for segment in tl.static_range(3):
        cols = segment * 2048 + offs
        tl.store(query + row * 6144 + cols, tl.load(packed + row * 7168 + cols))


def try_project_qkv(x, weight):
    import torch_npu

    if (not welm_decode_tiles_active() or x.ndim != 2 or x.shape[0] not in _TILES_PER_PROGRAM
            or x.shape[1] != 2048 or weight.shape != (7168, 2048)
            or x.device.type != "npu" or weight.device != x.device
            or x.dtype != torch.bfloat16 or weight.dtype != x.dtype
            or not x.is_contiguous() or not weight.is_contiguous()
            or x.data_ptr() % 32 or weight.data_ptr() % 32
            or torch_npu.get_npu_format(x) != 2 or torch_npu.get_npu_format(weight) != 2
            or not torch.npu.get_device_name(x.device).startswith("Ascend950")):
        return None

    rows = x.shape[0]
    tiles = _TILES_PER_PROGRAM[rows]
    packed = x.new_empty((rows, 7168))
    query = x.new_empty((rows, 6144))
    _welm_bf16_qkv_kernel[(rows * (14 // tiles),)](x, weight, packed, tiles)
    _welm_pack_query_kernel[(rows,)](packed, query)
    return query, packed[:, 6144:6656], packed[:, 6656:]
