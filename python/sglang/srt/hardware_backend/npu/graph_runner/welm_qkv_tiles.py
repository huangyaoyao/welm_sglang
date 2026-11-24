"""BF16 QKV projection for the fixed two-token WeLM MK decode bucket."""
import torch
import triton
import triton.language as tl

from megakernel.triton import native

from .welm_tile_layouts import welm_decode_tiles_active


def _projection_layout(x, weight, packed):
    if x.shape != (2, 2048) or weight.shape != (7168, 2048) or packed.shape != (2, 7168):
        return None
    return dict(count=packed.numel(), block=512, reads=[],
                writes=[(packed, packed.element_size())], shared_reads=[x, weight],
                allow_subviews=True)


@native(core="cube", enabled=welm_decode_tiles_active, tile_layout=_projection_layout)
@triton.jit
def _welm_bf16_qkv_kernel(x, weight, packed):
    pid = tl.program_id(0)
    row = pid // 14
    col = (pid % 14) * 512
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
    if packed.shape != (2, 7168) or query.shape != (2, 6144):
        return None
    return dict(count=2, block=1, reads=[(packed, 7168 * packed.element_size())],
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

    if (not welm_decode_tiles_active() or x.shape != (2, 2048) or weight.shape != (7168, 2048)
            or x.device.type != "npu" or weight.device != x.device
            or x.dtype != torch.bfloat16 or weight.dtype != x.dtype
            or not x.is_contiguous() or not weight.is_contiguous()
            or x.data_ptr() % 32 or weight.data_ptr() % 32
            or torch_npu.get_npu_format(x) != 2 or torch_npu.get_npu_format(weight) != 2
            or not torch.npu.get_device_name(x.device).startswith("Ascend950")):
        return None


    packed = x.new_empty((2, 7168))
    query = x.new_empty((2, 6144))
    _welm_bf16_qkv_kernel[(28,)](x, weight, packed)
    _welm_pack_query_kernel[(2,)](packed, query)
    return query, packed[:, 6144:6656], packed[:, 6656:]
