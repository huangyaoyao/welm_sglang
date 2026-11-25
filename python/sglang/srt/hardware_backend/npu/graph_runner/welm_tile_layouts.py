"""Native tile contracts for the original WeLM decode Triton programs."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

_decode_active = ContextVar("welm_native_decode_tiles", default=None)


@contextmanager
def decode_tile_scope(*, single_pk=False):
    from sglang.srt.utils import get_bool_env_var

    workspace = {"single_pk": True} if single_pk else {}
    if get_bool_env_var("SGLANG_NPU_WELMV4_MK_MODEL_ROW_TILES"):
        workspace["model_row_tiles"] = True
    if get_bool_env_var("SGLANG_NPU_WELMV4_MK_PAGE_GATE_TILES"):
        workspace["page_attention_output"] = None
    token = _decode_active.set(workspace)
    try:
        yield
    finally:
        _decode_active.reset(token)


def welm_decode_tiles_active():
    return _decode_active.get() is not None


def welm_single_pk_decode_active():
    workspace = _decode_active.get()
    return workspace is not None and workspace.get("single_pk") is True


def welm_decode_tile_workspace():
    """Scratch shared by layers of this capture, isolated from other plans."""
    return _decode_active.get()


def welm_model_row_tiles_active():
    workspace = _decode_active.get()
    return workspace is not None and workspace.get("model_row_tiles", False)


def welm_page_gate_tiles_active():
    workspace = _decode_active.get()
    return workspace is not None and "page_attention_output" in workspace


def record_page_attention_output(output):
    # Keep only the latest merge alive until its gate uses the result. A bare
    # address could match an unrelated allocation after the output is freed.
    workspace = _decode_active.get()
    if "page_attention_output" in workspace:
        workspace["page_attention_output"] = output


def take_page_attention_output(output):
    """Recognize the full contiguous view passed through RadixAttention."""
    workspace = _decode_active.get()
    merged = workspace.get("page_attention_output") if workspace is not None else None
    if (merged is None or output.device != merged.device or output.dtype != merged.dtype
            or output.numel() != merged.numel() or not output.is_contiguous()
            or output.data_ptr() != merged.data_ptr()):
        return False
    workspace["page_attention_output"] = None
    return True


def _decode_native(*, tile_layout, **options):
    from megakernel.triton import native

    def decode_layout(*args, **kwargs):
        layout = tile_layout(*args, **kwargs)
        if layout is None:
            return None
        # SGLang slices fixed bucket buffers. The capture layer checks the
        # declared byte extent against both the view and its owning storage.
        return dict(layout, allow_subviews=True)

    return native(enabled=welm_decode_tiles_active, tile_layout=decode_layout, **options)


def _k_norm_layout(
    x, gamma, output, tokens, kv_heads, head_dim, x_stride, output_stride,
    eps, kv_heads_block, x_column=0,
):
    row = kv_heads * head_dim
    packed = x.shape == (tokens, x_stride) and 0 <= x_column <= x_stride - row
    dense = x.shape == (tokens, kv_heads, head_dim) and x_stride == row and x_column == 0
    if (not (packed or dense) or output.shape != (tokens, kv_heads, head_dim)
            or gamma.numel() != head_dim or output_stride != row):
        return None
    return dict(
        count=tokens,
        block=1,
        reads=[(x, x_stride * x.element_size())],
        writes=[(output, row * output.element_size())],
        shared_reads=[gamma],
    )


def _rope_layout(
    q, k, positions, table, last_index, tokens, last_query_rows, q_stride, k_stride,
    head_dim, rope_dim, stages, q_heads, k_heads, q_heads_block, k_heads_block,
):
    # Generic equal-row Q/K path. Indirect positions only read the shared table.
    if (
        last_index is not None or last_query_rows != 0
        or not 0 < rope_dim <= head_dim or rope_dim % 2
        or q.shape != (tokens, q_heads, head_dim)
        or k.shape != (tokens, k_heads, head_dim)
        or q_stride != q_heads * head_dim or k_stride != k_heads * head_dim
        or positions.shape != (tokens,)
    ):
        return None
    rows = [(q, q_stride * q.element_size()), (k, k_stride * k.element_size())]
    return dict(count=tokens, block=1, reads=[*rows, (positions, positions.element_size())],
                writes=rows, shared_reads=[table])


@lru_cache(maxsize=1)
def install_welm_tile_layouts():
    """Register Triton children once in the WeLM native worker."""
    import torch

    from sglang.srt.layers import welmv4_npu_op, welmv4_op
    from .welm_attention_tiles import install_attention_tiles
    from .welm_oe_tiles import install_oe_tiles

    if torch.npu.get_device_name().startswith("Ascend950"):
        from megakernel.triton import native
        from sglang.srt.hardware_backend.npu.attention import sink_full_attention

        sink_full_attention.paged_decode_kernel = native(
            child=sink_full_attention._paged_decode_child,
            core="mix",
            enabled=welm_decode_tiles_active,
        )(sink_full_attention.paged_decode_kernel)
    install_attention_tiles(_decode_native)
    install_oe_tiles(_decode_native)

    welmv4_op.mmq_style_k_rms_norm_kernel = _decode_native(tile_layout=_k_norm_layout)(
        welmv4_op.mmq_style_k_rms_norm_kernel
    )
    welmv4_npu_op._welmv4_inplace_rope_kernel_npu = _decode_native(tile_layout=_rope_layout)(
        welmv4_npu_op._welmv4_inplace_rope_kernel_npu
    )


def refresh_welm_moe_tiles():
    # Which optional extension modules exist depends on the actual forward.
    from .welm_moe_tiles import install_moe_tiles

    install_moe_tiles(_decode_native)
