"""Tile footprints of the original Triton gates and norms used by WeLM decode."""

from importlib import import_module


def _norm_rows(rows, cols, block_rows, inputs, outputs, weights):
    """The kernel's reduction over cols stays within each owned row."""
    if min(rows, cols, block_rows) <= 0:
        return None
    inputs = [tensor for tensor in inputs if tensor is not None]
    outputs = [tensor for tensor in outputs if tensor is not None]
    if any(t.numel() != rows * cols or t.shape[-1] != cols for t in inputs + outputs):
        return None
    if any(t.numel() != cols for t in weights):
        return None
    return dict(
        count=rows,
        block=block_rows,
        reads=[(t, cols * t.element_size()) for t in inputs],
        writes=[(t, cols * t.element_size()) for t in outputs],
        shared_reads=list(weights),
    )


def _specialized_norm(x, residual, gamma, out, auxiliary, rows, cols, eps,
                      BLOCK_SIZE, BLOCK_ROWS=1):
    # These kernels load BLOCK_SIZE columns without a column mask. In both
    # specializations the fifth pointer is a separate output; True/True
    # writes post-norm FP32 there and False/False writes pre-norm residual.
    if BLOCK_SIZE != cols or out is None or auxiliary is None:
        return None
    return _norm_rows(rows, cols, BLOCK_ROWS, (x, residual), (out, auxiliary), (gamma,))


def _generic_norm(x, residual, gamma, out, out_residual, out_copy, rows, cols, eps,
                  x_row_stride, heads, x_batch_stride, residual_row_stride,
                  residual_after_layernorm, BLOCK_SIZE):
    # For 3D Q the source indexes row % heads and row // heads separately.
    # Both terms must describe the same contiguous linear row numbering.
    if (heads <= 0 or rows % heads
            or (residual is not None and residual_row_stride != cols)
            or (residual_after_layernorm and out_residual is None)
            or (not residual_after_layernorm and residual is None and out_residual is not None)
            or BLOCK_SIZE < cols):
        return None
    outputs = (out, out_residual, out_copy)
    if x_row_stride == cols and (rows == heads or x_batch_stride == heads * cols):
        return _norm_rows(rows, cols, 1, (x, residual), outputs, (gamma,))
    # In the QKV split branch Q can be a 3D view with gaps between tokens.
    # Preserve the entire source allocation as a read, while the distinct,
    # tightly packed norm outputs can still feed a tiled consumer.
    if (len(x.shape) != 3 or x.shape != (rows // heads, heads, cols)
            or x.stride() != (x_batch_stride, x_row_stride, 1)
            or x_row_stride < cols or x_batch_stride < heads * x_row_stride):
        return None
    layout = _norm_rows(rows, cols, 1, (residual,), outputs, (gamma,))
    if layout is not None:
        layout["shared_reads"].append(x)
    return layout


def _post_attention_norm(x, residual, onorm_weight, rnorm_weight, out, out_residual,
                         fp32_out, rows, cols, eps, BLOCK_SIZE, WRITE_FP32_OUT):
    if BLOCK_SIZE < cols:
        return None
    # The launcher reuses out_residual as the unused FP32 pointer.
    if not WRITE_FP32_OUT and fp32_out is not out_residual:
        return None
    outputs = (out, out_residual, fp32_out) if WRITE_FP32_OUT else (out, out_residual)
    return _norm_rows(rows, cols, 1, (x, residual), outputs, (onorm_weight, rnorm_weight))


def _sigmoid_gate(x, y, rows, cols, x_row_stride, y_row_stride, y_col_stride, BLOCK_SIZE):
    # A logical row is one (token, head) pair. One program owns BLOCK_SIZE
    # such rows, even when a program boundary falls inside a token.
    if (min(rows, cols, BLOCK_SIZE) <= 0 or x.shape != (rows, 1)
            or y.shape != (rows, cols) or x_row_stride != 1
            or y_row_stride != cols or y_col_stride != 1):
        return None
    values = (y, cols * y.element_size())
    return dict(count=rows, block=BLOCK_SIZE, reads=[(x, x.element_size()), values],
                writes=[values])


def _blocked_rope(q, k, positions, table, num_blocks, rows, q_stride, k_stride, head_dim,
                  rope_dim, token_block, masked, stages, q_heads, k_heads, *, multibuffer=True):
    # Despite its upstream name the launcher selects this JIT by shape,
    # including large ordinary decode/verify buckets with arbitrary positions.
    if (min(rows, token_block) <= 0 or (q_heads, k_heads) not in ((6, 1), (12, 1), (24, 2))
            or num_blocks != (rows + token_block - 1) // token_block
            or (not masked and rows % token_block)
            or not 0 < rope_dim <= head_dim or rope_dim % 2
            or q.shape != (rows, q_heads, head_dim) or q_stride != q_heads * head_dim
            or k.shape != (rows, k_heads, head_dim) or k_stride != k_heads * head_dim
            or positions.shape != (rows,) or table.shape[-1] != rope_dim):
        return None
    data = [(q, q_stride * q.element_size()), (k, k_stride * k.element_size())]
    return dict(count=rows, block=token_block,
                reads=[*data, (positions, positions.element_size())], writes=data,
                shared_reads=[table])


KERNEL_LAYOUTS = {
    "welmv4_op": (
        ("rms_norm_kernel", _generic_norm),
        ("mmq_style_norm_after_attn_kernel", _post_attention_norm),
    ),
    "welmv4_npu_op": (
        ("_welmv4_fused_rms_norm_true_true_kernel_npu", _specialized_norm),
        ("_welmv4_fused_rms_norm_true_true_multirow_kernel_npu", _specialized_norm),
        ("_welmv4_fused_rms_norm_false_false_kernel_npu", _specialized_norm),
        ("_welmv4_fused_rms_norm_false_false_multirow_kernel_npu", _specialized_norm),
        ("sigmoid_mul_kernel_npu", _sigmoid_gate),
        ("_welmv4_inplace_rope_prefill_kernel_npu", _blocked_rope),
    ),
}


def install_attention_tiles(native):
    for module_name, bindings in KERNEL_LAYOUTS.items():
        module = import_module(f"sglang.srt.layers.{module_name}")
        for name, layout in bindings:
            setattr(module, name, native(tile_layout=layout)(getattr(module, name)))
