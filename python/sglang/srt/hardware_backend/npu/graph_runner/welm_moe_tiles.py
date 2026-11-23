"""Audited block footprints for WeLM routing, expert activation and logits copies."""

import logging
import sys
from functools import lru_cache
from hashlib import sha256
from math import prod

logger = logging.getLogger(__name__)


def _routing_l1_layout(input_ptr, output_ptr, batch_size, hidden_size, NUM_CORES):
    if (
        min(batch_size, hidden_size, NUM_CORES) <= 0
        or input_ptr.shape != (batch_size, hidden_size)
        or output_ptr.shape != input_ptr.shape
    ):
        return None
    block = (batch_size + NUM_CORES - 1) // NUM_CORES
    if (batch_size + block - 1) // block != NUM_CORES:
        return None  # This launch includes idle programs beyond the final row.
    return dict(
        count=batch_size,
        block=block,
        reads=[(input_ptr, hidden_size * input_ptr.element_size())],
        writes=[(output_ptr, hidden_size * output_ptr.element_size())],
        allow_subviews=True,
    )


def _expert_swiglu_layout(
    x_ptr, group_list_ptr, out_ptr, scale_ptr, *, TOTAL_COLS, HALF_COLS,
    NUM_EXPERTS, GROUP_LIST_TYPE, NUM_CORES, SCALE, **options,
):
    # With exactly NUM_CORES rows of capacity, every legal nonempty dispatch
    # has ceil(sum(expert_counts) / NUM_CORES) == 1. Only an active prefix
    # writes. For a larger allocation the same pid can change rows at replay.
    if (
        GROUP_LIST_TYPE != 1 or NUM_CORES <= 0 or NUM_EXPERTS <= 0
        or HALF_COLS <= 0 or TOTAL_COLS != 2 * HALF_COLS
        or x_ptr.shape != (NUM_CORES, TOTAL_COLS)
        or out_ptr.shape != (NUM_CORES, HALF_COLS)
        or scale_ptr.shape != (NUM_CORES,)
        or group_list_ptr.shape != (NUM_EXPERTS,)
        or not group_list_ptr.is_contiguous()
    ):
        return None
    writes = [(out_ptr, HALF_COLS * out_ptr.element_size())]
    shared_reads = [group_list_ptr]
    if SCALE:
        writes.append((scale_ptr, scale_ptr.element_size()))
    else:
        # This launch ABI still passes the allocated scale argument. Keep a
        # conservative whole-buffer dependency on the unused pointer.
        shared_reads.append(scale_ptr)
    return dict(
        count=NUM_CORES,
        block=1,
        reads=[(x_ptr, TOTAL_COLS * x_ptr.element_size())],
        writes=writes,
        shared_reads=shared_reads,
        allow_subviews=True,
    )


def _dp_copy_layout(
    dst_ptr, src_ptr, offset_ptr, sz_ptr, offset_src, chunk_size, BLOCK_SIZE
):
    if (
        type(offset_src) is not bool or min(chunk_size, BLOCK_SIZE) <= 0
        or offset_ptr.numel() != 1 or sz_ptr.numel() != 1
        or dst_ptr.shape[1:] != src_ptr.shape[1:]
        or chunk_size != prod(dst_ptr.shape[1:])
    ):
        return None
    # The device offset and length change on replay. Only the buffer whose
    # addresses start at pid * BLOCK_SIZE can participate in a tiled edge.
    local, arena = (dst_ptr, src_ptr) if offset_src else (src_ptr, dst_ptr)
    if local.numel() == 0 or local.numel() > arena.numel():
        return None
    span = [(local, local.element_size())]
    if offset_src:
        return dict(
            count=local.numel(), block=BLOCK_SIZE, reads=[], writes=span,
            shared_reads=[src_ptr, offset_ptr, sz_ptr], allow_subviews=True,
        )
    return dict(
        count=local.numel(), block=BLOCK_SIZE, reads=span, writes=[],
        shared_reads=[offset_ptr, sz_ptr], whole_writes=[dst_ptr], allow_subviews=True,
    )


# The two external contracts cover the original allgather-deepep kernels from
# a9b33b7. An independently installed package can keep the same call signature
# but change its program-ID mapping. Only bind the source that was audited.
_KERNELS = (
    ("sgl_kernel_npu.norm.l1_norm", "l1_norm_kernel", _routing_l1_layout,
     "2410b71d9f96db64331f9a17c56eb32456f47bb47d761065000aec038539d45c"),
    ("sgl_kernel_npu.activation.swiglu_quant", "_swiglu_quant_kernel", _expert_swiglu_layout,
     "76b9d5086ea0d915af8406520ef862b64c0f8929a9bbc76d99e0ec1448f627c3"),
    ("sglang.kernels.ops.memory.memcpy_triton", "memcpy_triton_kernel", _dp_copy_layout, None),
)


@lru_cache(maxsize=None)
def _install(module, name, layout, source_hash, native):
    kernel = getattr(module, name, None)
    if kernel is None:
        logger.warning("No WeLM tile contract for missing kernel %s.%s", module.__name__, name)
        return
    if source_hash is not None:
        src = getattr(kernel, "src", "")
        if not isinstance(src, str) or sha256(src.strip().encode()).hexdigest() != source_hash:
            logger.warning("No WeLM tile contract for changed kernel %s.%s", module.__name__, name)
            return
    setattr(module, name, native(tile_layout=layout)(kernel))


def install_moe_tiles(native):
    """Inspect the process's loaded routing modules after ordinary warmup."""
    for path, name, layout, source_hash in _KERNELS:
        module = sys.modules.get(path)
        if module is not None:
            _install(module, name, layout, source_hash, native)
