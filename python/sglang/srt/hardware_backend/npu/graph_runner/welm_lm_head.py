"""Ordinary single-rank BF16 WeLM decoding with vocabulary-striped logits."""
from contextvars import ContextVar

import torch
import torch_npu
import triton
import triton.language as tl
from megakernel._runtime import _runtime
from megakernel.triton import native

from sglang.srt.layers.logits_processor import LogitsProcessor, should_apply_lm_head_quant_method
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

from .welm_tile_layouts import welm_decode_tiles_active


_VOCAB = 155648
_HIDDEN = 2048
_STRIPES = {1: 16, 2: 16, 4: 8, 8: 4}
_LOCAL_HEAD = ContextVar("welm_local_lm_head", default=None)


def _fanout_layout(x, copies, stripes):
    return dict(count=x.shape[0], block=1,
                reads=[(x, _HIDDEN * x.element_size())],
                writes=[(copies, stripes * _HIDDEN * x.element_size())], allow_subviews=True)


@native(enabled=welm_decode_tiles_active, tile_layout=_fanout_layout)
@triton.jit
def _welm_head_fanout(x, copies, stripes: tl.constexpr):
    row = tl.program_id(0)
    dim = tl.arange(0, 2048)
    hidden = tl.load(x + row * 2048 + dim)
    for stripe in tl.static_range(stripes):
        tl.store(copies + (row * stripes + stripe) * 2048 + dim, hidden)


def _projection_layout(copies, weight, bf16_logits, stripes, width):
    return dict(count=copies.shape[0] * stripes, block=1,
                reads=[(copies, _HIDDEN * copies.element_size())],
                writes=[(bf16_logits, width * bf16_logits.element_size())],
                shared_reads=[weight], allow_subviews=True)


@native(core="cube", enabled=welm_decode_tiles_active, tile_layout=_projection_layout)
@triton.jit
def _welm_head_projection(copies, weight, bf16_logits, stripes: tl.constexpr, width: tl.constexpr):
    owner = tl.program_id(0)
    col = (owner % stripes) * width
    for tile in range(width // 512):
        a = tl.make_block_ptr(copies + owner * 2048, (1, 2048), (2048, 1),
                              (0, 0), (1, 128), (1, 0))
        b = tl.make_block_ptr(weight + (col + tile * 512) * 2048, (512, 2048), (2048, 1),
                              (0, 0), (512, 128), (1, 0))
        acc = tl.full((1, 512), 0, tl.float32)
        for _ in range(16):
            acc = tl.dot(tl.load(a), tl.trans(tl.load(b)), acc)
            a = tl.advance(a, (0, 128))
            b = tl.advance(b, (0, 128))
        dst = tl.make_block_ptr(bf16_logits + owner * width + tile * 512,
                                (1, 512), (512, 1), (0, 0), (1, 512), (1, 0))
        tl.store(dst, acc.to(tl.bfloat16))


def _cast_layout(bf16_logits, logits, width):
    return dict(count=bf16_logits.numel() // width, block=1,
                reads=[(bf16_logits, width * bf16_logits.element_size())],
                writes=[(logits, width * logits.element_size())], allow_subviews=True)


@native(enabled=welm_decode_tiles_active, tile_layout=_cast_layout)
@triton.jit
def _welm_head_cast(bf16_logits, logits, width: tl.constexpr):
    start = tl.program_id(0) * width
    lane = tl.arange(0, 512)
    for tile in range(width // 512):
        offset = start + tile * 512 + lane
        tl.store(logits + offset, tl.load(bf16_logits + offset).to(tl.float32))


def _dense_nd(tensor, device, dtype):
    return (tensor.dtype == dtype and tensor.device == device and tensor.is_contiguous()
            and tensor.data_ptr() % 32 == 0 and torch_npu.get_npu_format(tensor) in (0, 2))


def try_welm_final_norm(norm, x):
    if (not welm_decode_tiles_active() or x.ndim != 2 or x.shape[0] not in _STRIPES
            or x.shape[1] != _HIDDEN or x.device.type != "npu"
            or not torch.npu.get_device_name(x.device).startswith("Ascend950")
            or not getattr(_runtime(), "_capture_cube_tile_inputs", False)
            or norm.weight.shape != (_HIDDEN,) or not _dense_nd(x, x.device, torch.bfloat16)
            or not _dense_nd(norm.weight, x.device, torch.bfloat16)):
        return None
    # Use the existing row-owned WeLM RMS kernel after the original BF16
    # pre-normalization cast; return the same pre-norm value to hidden capture.
    return norm(x)[0]


def _welm_projection_stripes(rows, tp_size):
    stripes = _STRIPES.get(rows)
    if stripes is None or tp_size not in (1, 4, 8):
        return None
    # Every local vocabulary stripe contains complete 512-column Cube tiles.
    return min(stripes, 16 // tp_size)


def try_welm_bf16_projection(x, weight, tp_size):
    rows = x.shape[0] if x.ndim == 2 else 0
    stripes = _welm_projection_stripes(rows, tp_size)
    if (not welm_decode_tiles_active() or stripes is None or x.shape != (rows, _HIDDEN)
            or weight.shape != (_VOCAB // tp_size, _HIDDEN) or x.device.type != "npu"
            or not torch.npu.get_device_name(x.device).startswith("Ascend950")
            or not _dense_nd(x, x.device, torch.bfloat16)
            or not _dense_nd(weight, x.device, torch.bfloat16)
            or not getattr(_runtime(), "_capture_cube_tile_inputs", False)):
        return None
    local_vocab = weight.shape[0]
    width = local_vocab // stripes
    copies = x.new_empty(rows * stripes * _HIDDEN).view(rows, stripes, _HIDDEN)
    bf16_logits = x.new_empty((rows, local_vocab))
    _welm_head_fanout[(rows,)](x, copies, stripes)
    _welm_head_projection[(rows * stripes,)](copies, weight, bf16_logits, stripes, width)
    return bf16_logits


def try_welm_logits(x, weight, out):
    if x.ndim != 2 or x.device.type != "npu":
        return None
    rows = x.shape[0]
    if out is not None and (out.shape != (rows, _VOCAB) or not _dense_nd(out, x.device, torch.float32)):
        return None
    bf16_logits = try_welm_bf16_projection(x, weight, 1)
    if bf16_logits is None:
        return None
    if out is None:
        out = x.new_empty((rows, _VOCAB), dtype=torch.float32)
    stripes = _STRIPES[rows]
    _welm_head_cast[(rows * stripes,)](bf16_logits, out, _VOCAB // stripes)
    return out


class WelmDecodeLogitsProcessor(LogitsProcessor):
    def supports_welm_head_tiles(self):
        return (self.vocab_size == _VOCAB and not self.use_fp32_lm_head
                and self.rl_on_policy_target is None and self.logit_scale is None
                and not self.final_logit_softcapping and not self.do_tensor_parallel_all_gather
                and not self.do_tensor_parallel_all_gather_dp_attn)

    def _get_logits(self, hidden_states, lm_head, logits_metadata, embedding_bias=None,
                    use_logits_buffer=True):
        if (not torch.compiler.is_compiling() and logits_metadata.forward_mode.is_decode()
                and welm_decode_tiles_active() and type(lm_head) is ParallelLMHead
                and getattr(lm_head, "quant_config", None) is None
                and not should_apply_lm_head_quant_method(lm_head, getattr(lm_head, "quant_method", None))
                and self.vocab_size == _VOCAB and not self.use_fp32_lm_head
                and self.rl_on_policy_target is None and embedding_bias is None):
            if lm_head.tp_size == 1 and self.supports_welm_head_tiles():
                out = logits_metadata.next_token_logits_buffer if use_logits_buffer else None
                logits = try_welm_logits(hidden_states, lm_head.weight, out)
                if logits is not None:
                    return logits
            elif lm_head.tp_size in (4, 8):
                token = _LOCAL_HEAD.set(lm_head)
                try:
                    # Parent owns DP hidden gather, TP vocab gather, DP scatter,
                    # scaling, FP32 output storage, and optional softcapping.
                    return super()._get_logits(hidden_states, lm_head, logits_metadata, embedding_bias,
                                               use_logits_buffer)
                finally:
                    _LOCAL_HEAD.reset(token)
        return super()._get_logits(hidden_states, lm_head, logits_metadata, embedding_bias,
                                   use_logits_buffer)

    def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None):
        if _LOCAL_HEAD.get() is lm_head and embedding_bias is None:
            local = try_welm_bf16_projection(hidden_states, lm_head.weight, lm_head.tp_size)
            if local is not None:
                return local
        return super()._compute_lm_head(hidden_states, lm_head, embedding_bias)
