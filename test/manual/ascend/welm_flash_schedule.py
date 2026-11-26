"""Inspect A5 CANN FlashAttnMetadata without treating a schedule as a completion signal.

The header and FA/FD records follow flash_attn_metadata.h. FA ranges use the
lexicographic half-open (BN, M, S2) iteration in flash_attn_kernel_dn.h.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Schedule:
    header: tuple
    fa_ranges: tuple
    fd_writers: tuple

    def as_dict(self):
        return asdict(self)


def decode_schedule(words):
    """Return live FA/FD records with (section, physical group/lane) labels."""
    if len(words) < 16:
        raise ValueError("truncated CANN Flash schedule header")
    sections, fd, mbase, s2base, cubes, vectors, mode = words[:7]
    if (min(sections, mbase, s2base, cubes) <= 0 or fd not in (0, 1)
            or vectors != 2 * cubes or mode not in (0, 1)):
        raise ValueError("unsupported CANN Flash schedule header")
    fd_start = 16 * (1 + sections * cubes)
    if len(words) < fd_start + 16 * sections * vectors:
        raise ValueError("truncated FA/FD records")
    fa, writers = [], []
    for section in range(sections):
        for group in range(cubes):
            start = 16 * (1 + section * cubes + group)
            fields = tuple(words[start:start + 7])
            if min(fields) < 0 or fields[:3] > fields[3:6]:
                raise ValueError("invalid FA half-open range")
            if fields[:3] < fields[3:6]:
                fa.append((section, group, *fields))
        if fd:
            for lane in range(vectors):
                start = fd_start + 16 * (section * vectors + lane)
                fields = tuple(words[start:start + 6])
                if fields[5] == 0:
                    continue
                if min(fields) < 0 or fields[3] == 0:
                    raise ValueError("invalid live FD writer")
                writers.append((section, lane, *fields))
    if fd and not writers:
        raise ValueError("FD enabled with no live FD final writer")
    return Schedule(tuple(words[:7]), tuple(fa), tuple(writers))


def validate_decode_ranges(schedule, lengths, query_lengths):
    """Bounds-check live records against this replay's TND 24:2 one-Q geometry."""
    if (len(lengths) not in (1, 2, 4, 8) or len(query_lengths) != len(lengths)
            or min(lengths) < 0 or any(q not in (0, 1) for q in query_lengths)):
        raise ValueError("unsupported one-query decode geometry")
    _, _, mbase, s2base, _, _, mode = schedule.header
    bn_per_row, m_per_query = (2, 12) if mode == 0 else (24, 1)
    bn_count = len(lengths) * bn_per_row
    for _, _, bn, m, s2, end_bn, end_m, end_s2, _ in schedule.fa_ranges:
        for b, i, j in ((bn, m, s2), (end_bn, end_m, end_s2)):
            if b > bn_count or (b == bn_count and (i, j) != (0, 0)):
                raise ValueError("FA range outside the replay batch")
            if b == bn_count:
                continue
            row = b // bn_per_row
            m_tiles = (m_per_query * query_lengths[row] + mbase - 1) // mbase
            s2_tiles = (lengths[row] + s2base - 1) // s2base
            if i > m_tiles or j > s2_tiles or (i == m_tiles and j != 0):
                raise ValueError("FA range outside the replay tile geometry")
    for _, _, bn, m, _, _, start, count in schedule.fd_writers:
        if bn >= bn_count:
            raise ValueError("FD writer outside the replay batch")
        rows_left = m_per_query * query_lengths[bn // bn_per_row] - m * mbase
        if start + count > min(mbase, rows_left):
            raise ValueError("FD writer outside its output rows")


@dataclass(frozen=True)
class MixCohort:
    """Required runtime guarantees; schedule metadata does not supply them."""
    groups: int
    vector_lanes: int
    waits_for_all_inputs: bool
    publishes_after_all_lanes: bool


def original_gate_inputs(groups, batch):
    """Map each original 128-(token,query head) gate program to its MIX groups."""
    if batch not in (1, 2, 4, 8) or set(groups) != {(r, h) for r in range(batch) for h in range(2)}:
        raise ValueError("requires all (row, KV group) owners")
    return tuple(
        tuple(sorted({groups[flat // 24, (flat % 24) // 12]
                      for flat in range(start, min(start + 128, batch * 24))}))
        for start in range(0, batch * 24, 128)
    )


def short_decode_final_groups(schedule, lengths, *, query_lengths, window, need_init_output, cohort=None):
    """Identify FA final MIX groups for this verified single-Q TND shape only.

    Requires unpadded Bx24 query heads, two KV head groups, all S2 in one
    tile, no FD, and every FA interval aligned
    exactly to complete (batch, KV head groups). The pid order comes from the
    supplied schedule on *this* replay. When CANN globally clears output, the
    full MIX cohort must enter behind an all-input wait. Its AIV SyncAll orders
    every clear before any FA writer. A group can publish only after all three
    physical lanes have returned and drained their GM writes. This establishes
    logical owners conditional on those guarantees, never hardware completion.
    """
    validate_decode_ranges(schedule, lengths, query_lengths)
    sections, fd, mbase, s2base, cubes, vectors, mode = schedule.header
    if mode != 0 or sections != 1 or fd:
        raise ValueError("requires one BN2_S1G_S2 section with FD disabled")
    if need_init_output is not False:
        if need_init_output is not True or cohort is None:
            raise ValueError("global output initialization unknown or full-cohort guarantees missing")
        if ((cohort.groups, cohort.vector_lanes) != (cubes, vectors)
                or not cohort.waits_for_all_inputs or not cohort.publishes_after_all_lanes):
            raise ValueError("global clear requires full-cohort entry and visible completion of every lane")
    if window not in (-1, 511) or len(lengths) not in (1, 2, 4, 8):
        raise ValueError("unsupported window or decode bucket")
    if tuple(query_lengths) != (1,) * len(lengths):
        raise ValueError("requires exactly one query token per active row")
    if mbase < 12 or min(lengths) <= 0 or max(lengths) > s2base:
        raise ValueError("requires full active rows and a single M and S2 tile per group")
    if window == 511 and max(lengths) > window + 1:
        raise ValueError("window truncates a KV row")
    group_count = 2 * len(lengths)
    owners = []
    for _, core, bn, m, s2, end_bn, end_m, end_s2, _ in schedule.fa_ranges:
        # Only accept the canonical whole-group boundaries. A split, overlap,
        # gap or alternate end representation needs independent verification.
        if (m, s2, end_m, end_s2) != (0, 0, 0, 0) or bn != len(owners) or not bn < end_bn <= group_count:
            raise ValueError("unaligned, overlapping, missing or out-of-bounds FA range")
        owners.extend([core] * (end_bn - bn))
    if len(owners) != group_count:
        raise ValueError("FA intervals do not cover every output group")
    return {(row, head): owners[2 * row + head] for row in range(len(lengths)) for head in range(2)}
