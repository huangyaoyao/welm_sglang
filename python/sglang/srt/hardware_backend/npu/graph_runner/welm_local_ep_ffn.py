"""Compute the local BF16 FFN on already-dispatched, compact expert rows."""
from contextvars import ContextVar
import os

import torch
import torch_npu
import triton
import triton.language as tl
from megakernel.triton import native

from .welm_ffn_tiles import _plain_module, _welm_expert_slot_mm, _welm_ffn_swiglu
from .welm_tile_layouts import welm_decode_tiles_active


_local_ffn = ContextVar("welm_local_ffn", default=False)
# One program owns consecutive complete rows, including inactive tail slots.
_SORTED_ROWS = {10: 1, 20: 1, 40: 2, 80: 5, 160: 8, 320: 16, 640: 20}


def run_with_local_ffn(experts, hidden, topk):
    token = _local_ffn.set(True)
    try:
        return experts(hidden, topk)
    finally:
        _local_ffn.reset(token)


@native(enabled=welm_decode_tiles_active)
@triton.jit
def _welm_sorted_expert_ids(counts, owners, experts: tl.constexpr):
    # The sorted buffer concatenates local expert groups in weight-bank order. Each program
    # reads their complete count table; empty groups advance the owner too.
    rows = tl.program_id(0) * 8 + tl.arange(0, 8)
    end = 0
    selected = tl.full((8,), 0, tl.int32)
    for expert in range(experts):
        end += tl.load(counts + expert).to(tl.int32)
        selected += (rows >= end).to(tl.int32)
    tl.store(owners + rows, tl.where(selected == experts, -1, selected))


def _clear_tail_layout(x, owners, out, slots):
    return dict(count=x.shape[0], block=slots, reads=[(x, 2048 * x.element_size())],
                shared_reads=[owners], writes=[(out, 2048 * out.element_size())], allow_subviews=True)


@native(enabled=welm_decode_tiles_active, tile_layout=_clear_tail_layout)
@triton.jit
def _welm_clear_sorted_tail(x, owners, out, slots: tl.constexpr):
    cols = tl.arange(0, 2048)
    for offset in range(slots):
        row = tl.program_id(0) * slots + offset
        value = tl.load(x + row * 2048 + cols)
        tl.store(out + row * 2048 + cols, tl.where(tl.load(owners + row) >= 0, value, 0.))


def try_sorted_experts(inputs, quant, layer):
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUUnquantMoEMethod

    if (not _local_ffn.get() or not welm_decode_tiles_active() or layer._dwdp_bound
            or layer._use_ascend_fuseep or not _plain_module(layer)
            or layer.moe_ep_size not in (4, 8) or layer.moe_tp_size != 1
            or layer.num_local_experts != 512 // layer.moe_ep_size
            or type(layer.w13_kernel) is not NPUUnquantMoEMethod
            or type(layer.w2_kernel) is not NPUUnquantMoEMethod or layer.quant_config is not None
            or layer.moe_runner_config.activation != "silu" or layer.moe_runner_config.gemm1_clamp_limit
            or getattr(quant, "w13_weight_bias", None) is not None
            or getattr(quant, "w2_weight_bias", None) is not None
            or inputs.group_list_type != 1 or os.getenv("MOE_EXPERT_TOKEN_NUMS_TYPE", "1") != "1"):
        return None
    experts = layer.num_local_experts
    x, counts, up, down = inputs.hidden_states, inputs.expert_tokens, quant.w13_weight, quant.w2_weight
    if (x.ndim != 2 or x.shape[1] != 2048 or x.shape[0] not in _SORTED_ROWS
            or x.shape[0] > layer.moe_ep_size * 8 * 10
            or x.device.type != "npu" or not torch.npu.get_device_name(x.device).startswith("Ascend950")
            or counts.shape != (experts,) or counts.dtype not in (torch.int32, torch.int64)
            or up.shape != (experts, 1024, 2048) or down.shape != (experts, 2048, 512)
            or any(t.dtype != torch.bfloat16 for t in (x, up, down))
            or any(t.device != x.device or not t.is_contiguous() or t.data_ptr() % 32
                   or torch_npu.get_npu_format(t) not in (0, 2) for t in (x, counts, up, down))):
        return None
    rows = x.shape[0]
    slots = _SORTED_ROWS[rows]
    grid = (rows // slots,)
    owner_blocks = triton.cdiv(rows, 8)
    owners = torch.empty((owner_blocks * 8,), device=x.device, dtype=torch.int32)
    packed = x.new_empty((rows, 1024))
    activation = x.new_empty((rows, 512))
    projected, out = torch.empty_like(x), torch.empty_like(x)
    _welm_sorted_expert_ids[(owner_blocks,)](counts, owners, experts)
    _welm_expert_slot_mm[grid](x, up, owners, packed, 2048, 1024, experts, False, slots)
    _welm_ffn_swiglu[grid](packed, activation, slots)
    _welm_expert_slot_mm[grid](activation, down, owners, projected, 512, 2048, experts, False, slots)
    _welm_clear_sorted_tail[grid](projected, owners, out, slots)
    return out
