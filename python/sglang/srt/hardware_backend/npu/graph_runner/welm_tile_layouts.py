"""Native tile contracts for the original WeLM decode Triton programs."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

_decode_active = ContextVar("welm_native_decode_tiles", default=False)


@contextmanager
def decode_tile_scope():
    token = _decode_active.set(True)
    try:
        yield
    finally:
        _decode_active.reset(token)


def welm_decode_tiles_active():
    return _decode_active.get()


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
    """Bind A5 implicit-program-ID children once in the opted-in worker."""
    from sglang.srt.layers import welmv4_npu_op, welmv4_op
    from .welm_attention_tiles import install_attention_tiles
    from .welm_oe_tiles import install_oe_tiles

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
