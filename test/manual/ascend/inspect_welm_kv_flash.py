"""Inspect one replay's native A5 scatter -> Flash input hazards from a JSON snapshot.

Use the actual scatter launch's tiling fields, cache data_ptrs from both calls,
and the slot mapping, page table and lengths passed in that replay. Full and
SWA must each use their own translated table, slots and cache. The returned
indices identify scatter AIV blocks, before intersecting with Flash's FA/FD
schedule. Keep every valid page, including masked offsets and shared prefixes.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def scatter_inputs_by_row(*, tiling_key, block_dim, dtype, layout, tiling, key_shape, value_shape, scatter_cache_ptrs,
                          flash_cache_ptrs, slot_mapping, block_table, kv_lengths):
    # A5 scatter_pa_kv_cache_apt.cpp dispatches these Norm keys to the dual
    # fully_load/not_fully_load variants. Both index slot_mapping by
    # blockIdx * blockFactor plus the source row within that block.
    if tiling_key not in (141, 181, 100) or dtype != "bfloat16" or layout != "PA_BBND":
        raise ValueError("requires a BF16 dual K/V Norm scatter and PA_BBND Flash")
    groups, block, tail = (tiling[k] for k in ("usedCoreNum", "blockFactor", "tailBlockFactor"))
    pages, width = tiling["numBlocks"], tiling["blockSize"]
    # Norm does not populate the raw numTokens field in the A5 tiling packet.
    # Use the actual compiled tensor descriptors and the rows assigned to AIVs.
    tokens = len(slot_mapping)
    if list(key_shape) != [tokens, 2, 256] or list(value_shape) != [tokens, 2, 256]:
        raise ValueError("unrecognized scatter input shapes")
    if not (groups == block_dim and groups > 0 and 0 < tail <= block and pages > 0
            and (groups - 1) * block + tail == tokens
            and width == 64 and len(block_table) == len(kv_lengths) == tokens):
        raise ValueError("inconsistent scatter launch geometry or decode row count")
    if (tiling["numHead"], tiling["kHeadSize"], tiling["vHeadSize"],
            tiling["kHandleNumPerCore"], tiling["vHandleNumPerCore"]) != (2, 256, 256, 512, 512):
        raise ValueError("unrecognized per-row K/V geometry")
    for cache in ("key", "value"):
        if tuple(tiling[f"{cache}CacheStride{i}"] for i in range(4)) != (64 * 512, 512, 256, 1):
            raise ValueError("requires contiguous paired K/V caches")
    if (len(scatter_cache_ptrs) != 2 or scatter_cache_ptrs != flash_cache_ptrs
            or min(scatter_cache_ptrs) <= 0
            or abs(scatter_cache_ptrs[0] - scatter_cache_ptrs[1]) < pages * width * 512 * 2):
        raise ValueError("Flash must read the same distinct K/V planes updated by scatter")

    rows = scatter_input_rows(slot_mapping, block_table, kv_lengths, pages, width)
    return [sorted({row // block for row in reads}) for reads in rows]


def scatter_input_rows(slot_mapping, block_table, kv_lengths, pages, width):
    """Conservative scatter source rows for the actual Full or translated SWA page table."""
    if width <= 0 or pages <= 0 or len(slot_mapping) != len(block_table) or len(block_table) != len(kv_lengths):
        raise ValueError("inconsistent page table or decode batch")
    writers = defaultdict(set)
    for row, slot in enumerate(slot_mapping):
        if slot == -1:
            continue
        if not 0 <= slot < width * pages:
            raise ValueError("scatter slot outside the captured cache")
        # A padded row may still write physical slot zero.
        writers[slot // width].add(row)

    inputs = []
    for table, length in zip(block_table, kv_lengths):
        count = (length + width - 1) // width
        if length < 0 or count > len(table):
            raise ValueError("length outside supplied page table")
        parents = set()
        for page in table[:count]:
            if not 0 <= page < pages:
                raise ValueError("invalid physical page in active KV prefix")
            parents.update(writers.get(page, ()))
        inputs.append(sorted(parents))
    return inputs


def scatter_inputs_by_flash_group(rows, fa_group_owners):
    """Join two maps from one replay's unsplit, fully active FA schedule.

    Each value is a set of potential RAW hazards. Flash's full-cohort
    initialization and unrelated hazards can still require an all-input wait.
    """
    if set(fa_group_owners) != {(row, kv_head) for row in range(len(rows)) for kv_head in (0, 1)}:
        raise ValueError("incomplete per-row FA group map")
    parents = defaultdict(set)
    for (row, _), group in fa_group_owners.items():
        parents[group].update(rows[row])
    return {group: sorted(writers) for group, writers in sorted(parents.items())}


def frozen_scatter_flash_waits(scatter, owners, physical_groups):
    """Check the data dependencies of the fixed B8 A5 scatter/Flash fixture.

    Pass owners from short_decode_final_groups for the same immutable FA
    operands and tiling. Static queue admission for the physical participants
    is a separate device runtime condition.
    """
    fields = scatter["tiling"]
    batch = 8
    if (physical_groups != 32 or scatter["tiling_key"] != 141 or scatter["block_dim"] != batch
            or len(scatter["slot_mapping"]) != batch or fields["blockFactor"] != 1
            or fields["tailBlockFactor"] != 1 or fields["usedCoreNum"] != batch):
        raise ValueError("expected the audited B8 row-per-AIV scatter and full-width MIX")
    expected = {(row, head): 2 * row + head for row in range(batch) for head in (0, 1)}
    if owners != expected:
        raise ValueError("unexpected FA assignment for the fixed input mapping")
    rows = scatter_inputs_by_row(**scatter)
    for row, length in enumerate(scatter["kv_lengths"]):
        if not 1 <= length <= 256:
            raise ValueError("expected eight active single-S2 rows")
        page, offset = divmod(length - 1, fields["blockSize"])
        last_slot = scatter["block_table"][row][page] * fields["blockSize"] + offset
        if scatter["slot_mapping"][row] != last_slot:
            raise ValueError("scatter slot does not append to the same physical row used by Flash")
    hazards = scatter_inputs_by_flash_group(rows, owners)
    waits = []
    for group in range(physical_groups):
        if group in hazards:
            writer = group // 2
            if hazards[group] != [writer]:
                raise ValueError("Flash group can read another scatter writer's physical page")
            waits.append([writer])
        else:
            # Inactive Flash groups still enter CANN's initialization barrier.
            waits.append(list(range(batch)))
    return waits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    try:
        snapshot = json.loads(args.snapshot.read_text())
        flash = snapshot.pop("flash_schedule", None)
        rows = scatter_inputs_by_row(**snapshot)
        result = {"scatter_aiv_blocks_by_flash_row": rows}
        if flash is not None:
            from welm_flash_schedule import MixCohort, decode_schedule, short_decode_final_groups

            owners = short_decode_final_groups(
                decode_schedule(flash["words"]), snapshot["kv_lengths"], query_lengths=flash["query_lengths"],
                window=flash["window"], need_init_output=flash["need_init_output"],
                cohort=MixCohort(**flash["cohort"]) if "cohort" in flash else None,
            )
            result["scatter_data_hazards_by_fa_group"] = scatter_inputs_by_flash_group(rows, owners)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(1, f"Cannot map scatter writes: {exc}\n")
    print(json.dumps(result))
