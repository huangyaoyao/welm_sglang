"""Row-owned BF16 projections in fixed ordinary WeLM decode."""
import torch
import torch_npu
import triton
import triton.language as tl

from megakernel.triton import native

from .welm_tile_layouts import welm_decode_tile_workspace, welm_decode_tiles_active


_SHAPES = {(6144, 2048), (2048, 2048), (2048, 1024), (512, 2048), (2048, 6144), (2048, 8192)}


def _projection_layout(x, weight, out, K, N):
    # All output columns of a row require every K element of that input row.
    if ((K, N) not in _SHAPES or x.ndim != 2 or x.shape[1] != K
            or weight.shape != (N, K) or out.shape != (x.shape[0], N)):
        return None
    return dict(count=x.shape[0], block=1, reads=[(x, K * x.element_size())],
                writes=[(out, N * out.element_size())], shared_reads=[weight], allow_subviews=True, keep_cube_width=True)


@native(core="cube", enabled=welm_decode_tiles_active, tile_layout=_projection_layout)
@triton.jit
def _welm_row_projection(x, weight, out, K: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0)
    for stripe in range(N // 256):
        col = stripe * 256
        a = tl.make_block_ptr(x + row * K, (1, K), (K, 1), (0, 0), (1, 128), (1, 0))
        b = tl.make_block_ptr(weight + col * K, (256, K), (K, 1), (0, 0), (256, 128), (1, 0))
        acc = tl.full((1, 256), 0, tl.float32)
        for k in range(K // 128):
            acc = tl.dot(tl.load(a), tl.trans(tl.load(b)), acc)
            a = tl.advance(a, (0, 128))
            b = tl.advance(b, (0, 128))
        dst = tl.make_block_ptr(out + row * N + col, (1, 256), (256, 1), (0, 0), (1, 256), (1, 0))
        tl.store(dst, acc.to(tl.bfloat16))


def row_projection_eligible(x, weight):
    return (
        welm_decode_tiles_active() and x.ndim == weight.ndim == 2
        and x.shape[0] in (1, 2, 4, 8) and (x.shape[1], weight.shape[0]) in _SHAPES
        and weight.shape[1] == x.shape[1] and x.device.type == "npu"
        and x.device == weight.device and x.dtype == weight.dtype == torch.bfloat16
        and x.is_contiguous() and weight.is_contiguous()
        and x.data_ptr() % 32 == 0 and weight.data_ptr() % 32 == 0
        and torch_npu.get_npu_format(x) in (0, 2) and torch_npu.get_npu_format(weight) in (0, 2)
        and torch.npu.get_device_name(x.device).startswith("Ascend950")
    )


def try_row_projection(x, weight):
    if not row_projection_eligible(x, weight):
        return None
    out = x.new_empty((x.shape[0], weight.shape[0]))
    _welm_row_projection[(x.shape[0],)](x, weight, out, x.shape[1], weight.shape[0])
    return out


def plain_linear(layer, kind):
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    return (type(layer) is kind and type(layer.quant_method) is UnquantizedLinearMethod
            and layer.bias is None and layer.quant_config is None and "forward" not in layer.__dict__
            and not layer._forward_pre_hooks and not layer._forward_hooks
            and not torch.nn.modules.module._global_forward_pre_hooks
            and not torch.nn.modules.module._global_forward_hooks)


def _embedding_mean_layout(base, oe, out, WIDTH):
    width = WIDTH * base.element_size()
    return dict(count=oe.shape[0], block=1, reads=[(base, width), (oe, width)],
                writes=[(out, width)], allow_subviews=True)


@native(enabled=welm_decode_tiles_active, tile_layout=_embedding_mean_layout)
@triton.jit
def _welm_embedding_mean(base, oe, out, WIDTH: tl.constexpr):
    offsets = tl.program_id(0) * WIDTH + tl.arange(0, WIDTH)
    # The original BF16 addition rounds before the division by two.
    summed = (tl.load(base + offsets).to(tl.float32) + tl.load(oe + offsets).to(tl.float32)).to(tl.bfloat16)
    tl.store(out + offsets, (summed.to(tl.float32) * 0.5).to(tl.bfloat16))


def try_oe_embedding(base, concat, weight):
    if (base.shape != (concat.shape[0], 2048) or base.device != concat.device or base.dtype != torch.bfloat16
            or not base.is_contiguous() or base.data_ptr() % 32 or base.device.type != "npu"
            or torch_npu.get_npu_format(base) not in (0, 2)):
        return None
    projected = try_row_projection(concat, weight)
    if projected is None:
        return None
    out = torch.empty_like(projected)
    _welm_embedding_mean[(concat.shape[0],)](base, projected, out, 2048)
    return out


def _base_lookup_layout(ids, weight, out, vocab):
    return dict(count=ids.numel(), block=1, reads=[(ids, ids.element_size())],
                writes=[(out, 2048 * out.element_size())], shared_reads=[weight], allow_subviews=True)


@native(enabled=welm_decode_tiles_active, tile_layout=_base_lookup_layout)
@triton.jit
def _welm_base_lookup(ids, weight, out, vocab: tl.constexpr):
    row = tl.program_id(0)
    token = tl.load(ids + row).to(tl.int64)
    # Only the checked row indices are consumed. Bound the device address as well.
    safe = tl.maximum(0, tl.minimum(token, vocab - 1))
    cols = tl.arange(0, 2048)
    tl.store(out + row * 2048 + cols, tl.load(weight + safe * 2048 + cols))


def try_base_embedding(ids, embedding):
    from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from sglang.srt.layers.welmv4_npu_op import _welm_plain_embedding_weight
    from sglang.srt.utils.async_probe import maybe_detect_oob

    workspace = welm_decode_tile_workspace()
    if (workspace is None or type(embedding) is not VocabParallelEmbedding or embedding.tp_size != 1
            or type(embedding.quant_method) is not UnquantizedEmbeddingMethod
            or embedding._forward_hooks or embedding._forward_pre_hooks or "forward" in embedding.__dict__
            or torch.nn.modules.module._global_forward_hooks or torch.nn.modules.module._global_forward_pre_hooks
            or ids.ndim != 1 or ids.numel() not in (1, 2, 4, 8) or not ids.is_contiguous()
            or ids.dtype not in (torch.int32, torch.int64) or ids.device.type != "npu"):
        return None
    weight = embedding.weight
    if (weight.device != ids.device or weight.dtype != torch.bfloat16 or not weight.is_contiguous()
            or weight.ndim != 2 or weight.shape[1] != 2048 or weight.shape[0] == 0
            or not torch.npu.get_device_name(ids.device).startswith("Ascend950")
            or not _welm_plain_embedding_weight(embedding, weight)):
        return None
    maybe_detect_oob(ids, 0, embedding.num_embeddings, "VocabParallelEmbedding input id")
    # Use the original embedding operator to resolve indices on the device.
    # Checking a scalar with aten::_assert_async would synchronize on A5.
    key = ("base_embedding_rows", ids.device, weight.shape[0])
    if key not in workspace:
        workspace[key] = torch.arange(weight.shape[0], dtype=torch.int32, device=ids.device).view(-1, 1)
    checked_ids = torch.nn.functional.embedding(ids.long(), workspace[key]).view(-1)
    out = torch.empty((ids.numel(), 2048), dtype=weight.dtype, device=ids.device)
    _welm_base_lookup[(ids.numel(),)](checked_ids, weight, out, weight.shape[0])
    return out
