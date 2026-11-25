"""Single-card A5 small-batch OE lookup replay, including CPU-mapped expert-independent vocab tables."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torch_npu
import triton
import triton.language as tl

from megakernel._fixed import capture_fixed
from megakernel.triton import prepare_bindings
from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import (
    decode_tile_scope,
    install_welm_tile_layouts,
)
from sglang.srt.hardware_backend.npu.host_mapped_embedding import (
    allocate_host_mapped_npu_tensor,
)
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.welmv4_npu_op import (
    _welmv4_oe_row_offset,
    try_welmv4_oe_lookup_4way_npu,
)


@triton.jit
def _check_row_offsets(ids, offsets):
    i = tl.arange(0, 4)
    tl.store(offsets + i, _welmv4_oe_row_offset(tl.load(ids + i), 512))


def run(build, large_table):
    torch.npu.set_device(0)
    assert torch.npu.get_device_name().startswith("Ascend950")
    install_welm_tile_layouts()
    # Verify int32 hash inputs above the 2**31-element and 2**32-element thresholds
    # on device, using only four result elements.
    row_ids = torch.tensor([0, 1 << 22, 1 << 23, 16_000_063], dtype=torch.int32)
    offsets = torch.empty(4, device="npu", dtype=torch.int64)
    _check_row_offsets[(1,)](row_ids.npu(), offsets)
    torch.testing.assert_close(offsets.cpu(), row_ids.long() * 512, rtol=0, atol=0)

    # --large-table also reads a real mapped row beyond 4 GiB. Only the selected
    # rows are initialized, and only the first of the four banks is that large.
    sizes = [((1 << 22) + 2) if large_table else 11, 13, 15, 17]
    choices = [[0] * 4, [size - 1 for size in sizes]]
    choices.extend([[row, row + 1, row + 2, row + 3] for row in range(1, 7)])
    tables = []
    plan = None
    try:
        branches = []
        pattern = torch.arange(512, dtype=torch.float32, device="cpu")
        for branch, size in enumerate(sizes):
            allocation = allocate_host_mapped_npu_tensor((size, 512), torch.bfloat16, 0)
            tables.append(allocation)
            weight = torch.nn.Parameter(allocation.npu_view, requires_grad=False)
            weight._host_mapped_npu_allocation = allocation
            assert weight.data_ptr() == allocation.dev_ptr
            for row in {case[branch] for case in choices}:
                allocation.cpu_view[row].copy_(((pattern + row % 67 + 31 * branch) % 127) - 63)
            branches.append(SimpleNamespace(
                weight=weight, tp_size=1, quant_method=UnquantizedEmbeddingMethod(),
                _host_mapped_npu_allocations={"weight": allocation},
            ))
        # The decode hash returns four planes separated by a padded pitch of 128.
        for batch in (1, 2, 4, 8):
            padded = torch.zeros((4, 128), dtype=torch.int32, device="npu")
            ids = padded[:, :batch]

            def forward():
                return try_welmv4_oe_lookup_4way_npu(ids, branches, sizes)

            assert forward() is None
            with decode_tile_scope():
                assert forward() is None
            with decode_tile_scope(single_pk=True):
                # Interleaved token IDs are outside the dense row contract.
                assert try_welmv4_oe_lookup_4way_npu(padded[:, :2 * batch:2], branches, sizes) is None
                with prepare_bindings():
                    assert forward() is not None
                torch.npu.synchronize()
                plan = capture_fixed(forward, build / f"bs{batch}", tile_chains=False)
            assert (plan.scope_count, plan.operator_count, plan.fused_operator_count, plan.host_boundary_count) == (1, 1, 1, 0), plan.execution_summary
            assert "oe_lookup_4way" in plan.plans[0].kernel_names[0]
            for shift in (0, 1, 3, None):
                # Each request reads different rows; the last replay repeats the same row.
                rows = [choices[0 if shift is None else (i + shift) % len(choices)]
                        for i in range(batch)]
                indices = torch.tensor(rows, dtype=torch.long)
                ids.copy_(indices.T)
                expected = torch.cat([
                    F.embedding(indices[:, branch], table.cpu_view)
                    for branch, table in enumerate(tables)
                ], dim=-1)
                torch.testing.assert_close(plan.replay().cpu(), expected, rtol=0, atol=0)
            plan.reset()
            plan = None
            print(f"PASS: four mapped OE lookups in one MK child; exact BS{batch} replays", flush=True)
    finally:
        if plan is not None:
            plan.reset()
        for allocation in tables:
            allocation.close(synchronize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--large-table", action="store_true", help="test an actual high row using 4 GiB of pinned RAM")
    args = parser.parse_args()
    with torch.inference_mode():
        run(args.build_dir, args.large_table)
