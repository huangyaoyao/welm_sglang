"""MK-recorded current decode lengths and the original CANN Flash metadata."""

import torch
import torch_npu


def can_record_decode_metadata(seq_lens, cu_q, used_q, used_kv):
    tensors = (seq_lens, cu_q, used_q, used_kv)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return False
    batch = seq_lens.numel()
    if (batch not in (1, 2, 4, 8, 12, 16) or seq_lens.device.type != "npu"
            or seq_lens.dtype not in (torch.int32, torch.int64)
            or any(t.dtype != torch.int32 for t in tensors[1:])):
        return False
    spans = []
    for tensor, size in zip(tensors, (batch, batch + 1, batch, batch)):
        # A one-element slice with stride > 1 is still contiguous to Torch.
        # The MK fixed descriptor requires exactly the dense 1-D stride.
        if (tensor.shape != (size,) or tensor.stride() != (1,)
                or tensor.layout != torch.strided or tensor.device != seq_lens.device
                or tensor.is_conj() or tensor.is_neg()
                or torch_npu.get_npu_format(tensor) not in (0, 2)):
            return False
        storage = tensor.untyped_storage()
        begin = tensor.data_ptr()
        end = begin + size * tensor.element_size()
        if not storage.data_ptr() <= begin < end <= storage.data_ptr() + storage.nbytes():
            return False
        spans.append((begin, end))
    spans.sort()
    return all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))


_LIBRARY = torch.library.Library("sglang", "FRAGMENT")
_LIBRARY.define(
    "welm_decode_flash_metadata(Tensor seq_lens, Tensor cu_seqlens_q, Tensor(a!) seqused_q, "
    "Tensor(b!) seqused_kv, int q_heads, int kv_heads, int head_dim, int max_seqlen_q, "
    "int max_seqlen_kv, int mask_mode, int win_left, int win_right) -> Tensor"
)


def _decode_flash_metadata(seq_lens, cu_seqlens_q, seqused_q, seqused_kv,
                           q_heads, kv_heads, head_dim, max_seqlen_q, max_seqlen_kv,
                           mask_mode, win_left, win_right):
    import cann_ops_transformer as cann

    # These are the original in-graph length producers. Enclosing them with
    # their first AICPU reader lets MK retain a single replay-stream service.
    valid = seq_lens > 0
    seqused_q.copy_(valid)
    seqused_kv.copy_(torch.where(valid, seq_lens, 0))
    return cann.flash_attn_metadata(
        q_heads, kv_heads, head_dim, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=None,
        seqused_q=seqused_q, seqused_kv=seqused_kv, max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv, mask_mode=mask_mode, win_left=win_left,
        win_right=win_right, layout_q="TND", layout_kv="PA_BBND", layout_out="TND",
    )


_LIBRARY.impl("welm_decode_flash_metadata", _decode_flash_metadata, "PrivateUse1")
