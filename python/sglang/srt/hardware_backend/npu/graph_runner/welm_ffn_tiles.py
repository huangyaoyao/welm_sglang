"""Single-rank BF16 WeLM experts. The CANN router owns expert selection and tie order."""
import torch
import triton
import triton.language as tl

from megakernel.triton import native
from .welm_tile_layouts import welm_decode_tiles_active


# Group whole consecutive slots; a program never crosses a token boundary.
_SLOTS = {1: 1, 2: 1, 4: 2, 8: 5}


def _swiglu_layout(packed, out, slots):
    return dict(count=out.shape[0], block=slots,
                reads=[(packed, 1024 * packed.element_size())],
                writes=[(out, 512 * out.element_size())], allow_subviews=True)


@native(enabled=welm_decode_tiles_active, tile_layout=_swiglu_layout)
@triton.jit
def _welm_ffn_swiglu(packed, out, slots: tl.constexpr):
    cols = tl.arange(0, 512)
    for local in tl.static_range(slots):
        row = tl.program_id(0) * slots + local
        gate = tl.load(packed + row * 1024 + cols).to(tl.float32)
        up = tl.load(packed + row * 1024 + 512 + cols).to(tl.float32)
        tl.store(out + row * 512 + cols, gate * tl.sigmoid(gate) * up)


def _slot_mm_layout(x, weights, ids, out, k, n, experts, expand, slots):
    reads = [] if expand else [(x, k * x.element_size())]
    shared = [weights, ids]
    if expand:
        shared.append(x)  # Ten slots fan out from one original token.
    # Each Cube program owns a fixed group of slots and has no cross-program sync.
    return dict(count=out.shape[0], block=slots, reads=reads, shared_reads=shared,
                writes=[(out, n * out.element_size())], keep_cube_width=True, allow_subviews=True)


@native(core="cube", enabled=welm_decode_tiles_active, tile_layout=_slot_mm_layout)
@triton.jit
def _welm_expert_slot_mm(x, weights, ids, out, k: tl.constexpr, n: tl.constexpr,
                         experts: tl.constexpr, expand: tl.constexpr, slots: tl.constexpr):
    # Keep the five-slot BS8 path a loop so it fits in the child image.
    for local in range(slots):
        slot = tl.program_id(0) * slots + local
        row = slot // 10 if expand else slot
        expert = tl.load(ids + slot)
        # Padding can carry -1. Even a speculative load must stay inside W.
        owner = tl.minimum(tl.maximum(expert, 0), experts - 1).to(tl.int64)
        base = weights + owner * n * k
        for tile in range(n // 512):
            a = tl.make_block_ptr(x + row * k, (1, k), (k, 1), (0, 0), (1, 128), (1, 0))
            b = tl.make_block_ptr(base + tile * 512 * k, (512, k), (k, 1),
                                  (0, 0), (512, 128), (1, 0))
            acc = tl.full((1, 512), 0, tl.float32)
            for _ in range(k // 128):
                acc = tl.dot(tl.load(a), tl.trans(tl.load(b)), acc)
                a = tl.advance(a, (0, 128))
                b = tl.advance(b, (0, 128))
            dst = tl.make_block_ptr(out + slot * n + tile * 512, (1, 512), (512, 1),
                                    (0, 0), (1, 512), (1, 0))
            tl.store(dst, acc.to(tl.bfloat16))


def _combine_layout(expert_out, ids, weights, out, experts):
    return dict(count=out.shape[0], block=1,
                reads=[(expert_out, 10 * 2048 * expert_out.element_size())],
                shared_reads=[ids, weights], writes=[(out, 2048 * out.element_size())],
                allow_subviews=True)


@native(enabled=welm_decode_tiles_active, tile_layout=_combine_layout)
@triton.jit
def _welm_combine_expert_slots(expert_out, ids, weights, out, experts: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, 2048)
    total = tl.full((2048,), 0, tl.float32)
    for index in tl.static_range(10):
        slot = row * 10 + index
        owner = tl.load(ids + slot)
        value = tl.load(expert_out + slot * 2048 + cols).to(tl.float32)
        weight = tl.load(weights + slot)
        total += tl.where((owner >= 0) & (owner < experts), value * weight, 0.)
    tl.store(out + row * 2048 + cols, total.to(tl.bfloat16))


def _plain_module(layer):
    return (not layer._forward_hooks and not layer._forward_pre_hooks
            and "forward" not in layer.__dict__
            and not torch.nn.modules.module._global_forward_hooks
            and not torch.nn.modules.module._global_forward_pre_hooks)


def plain_router(layer):
    from sglang.srt.layers.linear import ReplicatedLinear
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    return (type(layer) is ReplicatedLinear and type(layer.quant_method) is UnquantizedLinearMethod
            and layer.quant_config is None and layer.bias is None and _plain_module(layer))


def try_shared_expert(x, layer):
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.models.welmv4 import Qwen2MoeMLP
    from .welm_row_projection import try_row_projection

    if type(layer) is not Qwen2MoeMLP or layer.swiglu_clamp_limit or not _plain_module(layer):
        return None
    up, down = layer.gate_up_proj, layer.down_proj
    if (type(up) is not MergedColumnParallelLinear or type(down) is not RowParallelLinear
            or type(layer.act_fn) is not SiluAndMul or not _plain_module(layer.act_fn)
            or up.gather_output or not down.input_is_parallel or down.use_decode_attn_tp
            or any(type(m.quant_method) is not UnquantizedLinearMethod or m.tp_size != 1
                   or m.bias is not None or not _plain_module(m) for m in (up, down))
            or up.weight.shape != (1024, 2048) or down.weight.shape != (2048, 512)):
        return None
    packed = try_row_projection(x, up.weight)
    if packed is None:
        return None
    activation = x.new_empty((x.shape[0], 512))
    _welm_ffn_swiglu[(x.shape[0],)](packed, activation, 1)
    return try_row_projection(activation, down.weight)


def try_expert_slots(x, ids, scores, w13, w2):
    import torch_npu

    if (not welm_decode_tiles_active() or x.device.type != "npu"
            or not torch.npu.get_device_name(x.device).startswith("Ascend950")
            or x.ndim != 2 or x.shape[0] not in _SLOTS or x.shape[1] != 2048
            or w13.ndim != 3 or w13.shape[1:] != (1024, 2048)
            or not 1 <= w13.shape[0] <= 512 or w2.shape != (w13.shape[0], 2048, 512)
            or ids.shape != (x.shape[0], 10) or scores.shape != ids.shape
            or ids.dtype != torch.int32 or scores.dtype != torch.float32
            or any(t.dtype != torch.bfloat16 for t in (x, w13, w2))
            or any(t.device != x.device or not t.is_contiguous() or t.data_ptr() % 32 or
                   torch_npu.get_npu_format(t) not in (0, 2) for t in (x, ids, scores, w13, w2))):
        return None
    rows, experts = x.shape[0], w13.shape[0]
    slots = _SLOTS[rows]
    grid = (rows * 10 // slots,)
    packed = x.new_empty((rows * 10, 1024))
    activation = x.new_empty((rows * 10, 512))
    down = x.new_empty((rows * 10, 2048))
    out = torch.empty_like(x)
    _welm_expert_slot_mm[grid](x, w13, ids, packed, 2048, 1024, experts, True, slots)
    _welm_ffn_swiglu[grid](packed, activation, slots)
    _welm_expert_slot_mm[grid](activation, w2, ids, down, 512, 2048, experts, False, slots)
    _welm_combine_expert_slots[(rows,)](down, ids, scores, out, experts)
    return out


def try_routed_expert(x, topk, layer):
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUUnquantMoEMethod
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    if (type(layer) is not FusedMoE or not _plain_module(layer) or type(topk) is not StandardTopKOutput
            or layer._dwdp_bound or layer._use_ascend_fuseep or layer.moe_tp_size != 1
            or layer.moe_ep_size != 1 or layer.num_experts != layer.num_local_experts
            or layer.quant_config is not None or layer.num_fused_shared_experts
            or layer.moe_runner_config.no_combine or layer.moe_runner_config.apply_router_weight_on_input
            or not layer.moe_runner_config.is_gated
            or type(layer.dispatcher) is not AscendTPDispatcher
            or layer.dispatcher._original_dispatch_func or layer.dispatcher._original_combine_func
            or type(layer.w13_kernel) is not NPUUnquantMoEMethod
            or type(layer.w2_kernel) is not NPUUnquantMoEMethod
            or layer.moe_runner_config.activation != "silu" or layer.moe_runner_config.gemm1_clamp_limit
            or getattr(layer, "w13_weight_bias", None) is not None
            or getattr(layer, "w2_weight_bias", None) is not None):
        return None
    return try_expert_slots(x, topk.topk_ids, topk.topk_weights, layer.w13_weight, layer.w2_weight)
