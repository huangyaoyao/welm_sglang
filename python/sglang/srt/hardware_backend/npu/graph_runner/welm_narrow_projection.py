"""Small-batch WeLM decode projections with a Cube program per output-column tile."""
import torch
import torch_npu
import triton
import triton.language as tl
from megakernel.triton import native

from .welm_tile_layouts import welm_single_pk_decode_active


@native(core="cube", enabled=welm_single_pk_decode_active)
@triton.jit
def _welm_narrow_projection_kernel(
    x, weight, output, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, KN: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr, FP32_OUT: tl.constexpr,
):
    col = tl.program_id(0) * BN
    a = tl.make_block_ptr(x, (M, K), (K, 1), (0, 0), (M, BK), (1, 0))
    if KN:
        b = tl.make_block_ptr(weight, (K, N), (N, 1), (0, col), (BK, BN), (1, 0))
    else:
        b = tl.make_block_ptr(weight, (N, K), (K, 1), (col, 0), (BN, BK), (1, 0))
    acc = tl.full((M, BN), 0, tl.float32)
    for _ in range(K // BK):
        lhs = tl.load(a)
        if KN:
            rhs = tl.load(b, boundary_check=(1,), padding_option="zero")
            b = tl.advance(b, (BK, 0))
        else:
            rhs = tl.trans(tl.load(b, boundary_check=(0,), padding_option="zero"))
            b = tl.advance(b, (0, BK))
        acc = tl.dot(lhs, rhs, acc)
        a = tl.advance(a, (0, BK))
    out = tl.make_block_ptr(output, (M, N), (N, 1), (0, col), (M, BN), (1, 0))
    if FP32_OUT:
        tl.store(out, acc, boundary_check=(1,))
    else:
        tl.store(out, acc.to(tl.bfloat16), boundary_check=(1,))


def narrow_projection(x, weight, *, router):
    """Return the small-batch projection or None when the original linear must run.

    The gate stores [N,K] weights and returns BF16. The router takes [K,N]
    weights (including a transpose view) and returns unrounded FP32 logits.
    """
    if (not welm_single_pk_decode_active() or x.device.type != "npu"
            or weight.device != x.device or x.ndim != 2 or x.shape[1] != 2048
            or x.shape[0] not in (1, 2, 4, 8)
            or x.dtype != torch.bfloat16 or weight.dtype != x.dtype
            or weight.ndim != 2 or not x.is_contiguous()
            or x.data_ptr() % 32 or weight.data_ptr() % 32
            or not torch.npu.get_device_name(x.device).startswith("Ascend950")):
        return None
    m, k = x.shape
    n = weight.shape[1] if router else weight.shape[0]
    if (router and (weight.shape[0] != k or n != 512)
            or not router and (weight.shape[1] != k or n != 24)):
        return None
    # Check dense storage before computing strides in the device program.
    if any(torch_npu.get_npu_format(t) not in (0, 2) for t in (x, weight)):
        return None
    if weight.is_contiguous():
        kn = router
    elif router and tuple(weight.stride()) == (1, weight.shape[0]):
        kn = not router
    else:
        return None
    out = torch.empty((m, n), device=x.device, dtype=torch.float32 if router else x.dtype)
    # A column tile covers all input rows, so the router still uses 32 Cube
    # programs at BS8. Keep all 24 gate columns in one program when rows
    # share a 32-byte store boundary (BF16 gate rows are 48 bytes apart).
    bn = 32 if m > 1 and not router else 16
    _welm_narrow_projection_kernel[(triton.cdiv(n, bn),)](
        x, weight, out, m, n, k, kn, bn, 128, router,
    )
    return out
