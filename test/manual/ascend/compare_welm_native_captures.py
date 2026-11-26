"""Compare two single-rank WeLM MK debug logs from the native-operator A/B."""

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re


KEY = re.compile(r"WeLM MK capture ShapeKey\(size=(\d+), stream_idx=None, variant_label=None\)")
COUNTS = re.compile(
    r": (\d+) kernels, (\d+) scopes, (\d+) fused kernels, (\d+) Torch boundaries, (\d+) tile edges"
)
SCOPE = re.compile(r" scope (\d+) (kernels|tile edges) (.+)")
FLASH = "static_kernel_FlashAttn_"
SCATTER = "static_kernel_ScatterPaKvCache_"
GATE = "sigmoid_mul_kernel_npu"
# MK-specific replacements for original BF16 decode operators. Match both
# scope kernel names and retained launch names such as triton::_prepare_flash_pages.
REPLACEMENTS = {
    "_page_attention_partials", "_merge_page_attention", "_welm_paged_qk_cube",
    "_welm_paged_softmax", "_welm_paged_value_pack", "_welm_paged_pv_cube",
    "_welm_paged_normalize", "_write_kv_pages", "_read_flash_pages", "_prepare_flash_pages",
    "_welm_bf16_qkv_kernel", "_welm_pack_query_kernel", "_welm_row_projection",
    "_welm_o_projection_partials", "_welm_o_projection_merge", "_welm_narrow_projection_kernel",
    "_welm_base_lookup", "_welm_embedding_mean", "_welmv4_oe_lookup_4way_kernel",
    "_welmv4_oe_concat_4way_kernel", "_welm_ffn_swiglu", "_welm_expert_slot_mm",
    "_welm_combine_expert_slots", "_welm_sorted_expert_ids", "_welm_clear_sorted_tail",
    "_welm_head_fanout", "_welm_head_projection", "_welm_head_cast",
}


def _literal(value):
    if len(value) > 4 * 1024 * 1024:
        raise ValueError("debug literal exceeds 4 MiB")
    return ast.literal_eval(value)


def read_captures(lines, buckets):
    captures = {}
    for lineno, line in enumerate(lines, 1):
        match = KEY.search(line)
        if not match or int(match[1]) not in buckets:
            continue
        bs = int(match[1])
        body = line[match.end():].rstrip()
        counts = COUNTS.fullmatch(body)
        if counts:
            if bs in captures:
                raise ValueError(f"bs={bs} has multiple captures; supply one run per log")
            captures[bs] = dict(counts=tuple(map(int, counts.groups())), kernels={}, edges={}, coarse_edges={})
            continue
        try:
            cap = captures[bs]
            scope = SCOPE.fullmatch(body)
            if scope:
                index, kind, value = scope.groups()
                if kind == "kernels":
                    names, sep, after = value.partition(" coarse edges ")
                    if not sep:
                        raise ValueError("incomplete kernel list")
                    pairs, sep, _ = after.partition(" tile ratios ")
                    if not sep:
                        raise ValueError("incomplete coarse edge list")
                    cap["coarse_edges"][int(index)] = _literal(pairs)
                    dest, value = cap["kernels"], _literal(names)
                    if not isinstance(value, tuple) or not all(isinstance(k, str) for k in value):
                        raise ValueError("expected tuple of kernel names")
                else:
                    dest, value = cap["edges"], _literal(value)
                    if not isinstance(value, list) or not all(
                        isinstance(p, tuple) and len(p) == 2 and all(isinstance(k, str) for k in p)
                        for p in value
                    ):
                        raise ValueError("expected list of kernel name pairs")
                if int(index) in dest:
                    raise ValueError(f"duplicate {kind} for scope {index}")
                dest[int(index)] = value
            elif body.startswith(" boundaries "):
                if "boundaries" in cap:
                    raise ValueError("duplicate boundary report")
                cap["boundaries"] = _literal(body.removeprefix(" boundaries "))
        except (KeyError, SyntaxError, ValueError) as exc:
            raise ValueError(f"bs={bs}, log line {lineno}: {exc}") from exc

    for bs in buckets:
        if bs not in captures or "boundaries" not in captures[bs]:
            raise ValueError(f"missing debug capture or boundary report for bs={bs}")
        cap = captures[bs]
        _, scopes, fused, boundaries, tiles = cap["counts"]
        kernels, edges = cap["kernels"], cap["edges"]
        if (kernels.keys() != set(range(scopes)) or not edges.keys() <= kernels.keys()
                or sum(map(len, kernels.values())) != fused or sum(map(len, edges.values())) != tiles):
            raise ValueError(f"bs={bs} scope/edge evidence differs from capture totals")
        for kind in ("edges", "coarse_edges"):
            for i, pairs in cap[kind].items():
                names = set(kernels[i])
                if not isinstance(pairs, list) or any(
                    not isinstance(pair, tuple) or len(pair) != 2 or not all(name in names for name in pair)
                    for pair in pairs
                ):
                    raise ValueError(f"bs={bs} scope {i} has invalid {kind} or unknown kernels")
        if sum(row["calls"] for row in cap["boundaries"]["torch_boundaries"]) != boundaries:
            raise ValueError(f"bs={bs} boundary details differ from capture totals")
    return captures


def _flash_chains(cap, edge_key):
    """Name-only logs identify a Flash instance by (scope, unique kernel name)."""
    both = scatter_in = gate_out = native_chains = 0
    for scope, kernels in cap["kernels"].items():
        counts = Counter(kernels)
        pairs = cap[edge_key].get(scope, ())
        inputs = {dst for src, dst in pairs if src != dst}
        outputs = {src for src, dst in pairs if src != dst}
        from_scatter = {dst for src, dst in pairs if src.startswith(SCATTER)}
        to_gate = {src for src, dst in pairs if dst == GATE}
        unique = {name for name, n in counts.items() if n == 1 and name.startswith(FLASH)}
        both += len(unique & inputs & outputs)
        scatter_in += len(unique & from_scatter)
        gate_out += len(unique & to_gate)
        native_chains += len(unique & from_scatter & to_gate)
    return dict(two_sided=both, scatter_inputs=scatter_in, gate_outputs=gate_out,
                scatter_flash_gate=native_chains)


def summarize(cap):
    names = [k for group in cap["kernels"].values() for k in group]
    def flash_neighbours(edges):
        inputs, outputs = Counter(), Counter()
        for pairs in edges.values():
            for src, dst in pairs:
                if dst.startswith(FLASH):
                    inputs[src] += 1
                if src.startswith(FLASH):
                    outputs[dst] += 1
        return inputs, outputs

    inputs, outputs = flash_neighbours(cap["edges"])
    coarse_inputs, coarse_outputs = flash_neighbours(cap["coarse_edges"])
    kernels, scopes, fused, boundaries, tiles = cap["counts"]
    rows = cap["boundaries"]["torch_boundaries"]
    boundaries_by_op = Counter()
    for row in rows:
        boundaries_by_op[row["operation"]] += row["calls"]
    for rows in cap["boundaries"].values():
        names.extend(row.get("operation", row.get("kernel", "")) for row in rows)
    replacements = sorted({k for name in names for k in re.findall(r"\w+", name)} & REPLACEMENTS)
    exclusive = sum(
        len(group) == 3 and group[0].startswith(SCATTER) and group[1].startswith(FLASH) and group[2] == GATE
        for group in cap["kernels"].values()
    )
    return dict(kernels=kernels, scopes=scopes, fused=fused, torch_boundaries=boundaries,
                exclusive_scatter_flash_gate_scopes=exclusive,
                coarse_flash_chains=_flash_chains(cap, "coarse_edges"),
                tile_flash_chains=_flash_chains(cap, "edges"),
                boundary_ops=boundaries_by_op, tile_edges=tiles,
                flash_kernels=sum(k.startswith(FLASH) for group in cap["kernels"].values() for k in group),
                flash_inputs=inputs, flash_outputs=outputs, flash_coarse_inputs=coarse_inputs,
                flash_coarse_outputs=coarse_outputs, replacements=replacements)


def _compiled_contract(name, manifest):
    record = manifest.get(name, {})
    metadata = record.get("json", {})
    if (metadata.get("kernelName") != name or not metadata.get("sub_operator_op_type")
            or not metadata.get("supportInfo") or not metadata.get("blockDim")
            or not re.fullmatch("[0-9a-f]{64}", record.get("ELF_sha256", ""))
            or metadata.get("sha256") != record["ELF_sha256"]):
        return None
    sections = {key: value for key, value in record.get("sections", {}).items()
                if int(value[0], 0) & 2 or key in (".rel.text", ".rela.text")}
    if ".text" not in sections or any(not re.fullmatch("[0-9a-f]{64}", v[2]) for v in sections.values()):
        return None
    ids = {"kernelName", "binFileName", "sha256", "sub_operator_kernel_name"}
    return {key: value for key, value in metadata.items() if key not in ids}, sections


def compare(coarse, fine, buckets, require_flash_tiles=False, elf_manifest=None):
    result = {"buckets": {}, "errors": []}
    errors = result["errors"]
    for bs in buckets:
        a, b = coarse[bs], fine[bs]
        arms = {key: summarize(cap) for key, cap in (("coarse", a), ("fine", b))}
        result["buckets"][bs] = arms
        for arm, summary in arms.items():
            if summary["flash_kernels"] != 48:
                errors.append(f"bs={bs} {arm}: expected 48 complete fused FlashAttn kernels")
            if summary["replacements"]:
                errors.append(f"bs={bs} {arm}: found MK replacement operators")
        if arms["coarse"]["tile_edges"]:
            errors.append(f"bs={bs} coarse arm contains tile edges")
        if a["counts"][:4] != b["counts"][:4]:
            errors.append(f"bs={bs} capture totals differ between arms")
        if a["boundaries"] != b["boundaries"]:
            errors.append(f"bs={bs} boundary signatures differ between arms")
        verified = set()
        for i in sorted(a["kernels"].keys() | b["kernels"].keys()):
            left, right = a["kernels"].get(i, ()), b["kernels"].get(i, ())
            if len(left) != len(right):
                errors.append(f"bs={bs} scope {i} has different kernel counts")
                continue
            for j, (src, dst) in enumerate(zip(left, right)):
                if src == dst or (src, dst) in verified:
                    continue
                if elf_manifest:
                    original = _compiled_contract(src, elf_manifest)
                    if original is not None and original == _compiled_contract(dst, elf_manifest):
                        verified.add((src, dst))
                        continue
                errors.append(f"bs={bs} scope {i} kernel {j} differs: {src} vs {dst}")
                break
        if elf_manifest is not None:
            arms["verified_specializations"] = [list(pair) for pair in sorted(verified)]
        if require_flash_tiles and not arms["fine"]["tile_flash_chains"]["two_sided"]:
            errors.append(f"bs={bs} fine arm has no unambiguous FlashAttn with direct input and output tile edges in one scope")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("fine", type=Path)
    parser.add_argument("--buckets", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--require-flash-tiles", action="store_true",
                        help="require a uniquely identified FlashAttn with both direct tile edges in one scope per bucket")
    parser.add_argument("--elf-manifest", type=Path,
                        help="compare differing names by recorded ELF code, allocatable sections, text relocations and launch descriptors")
    args = parser.parse_args()
    try:
        with args.coarse.open() as left, args.fine.open() as right:
            result = compare(read_captures(left, args.buckets), read_captures(right, args.buckets),
                             args.buckets, args.require_flash_tiles,
                             json.loads(args.elf_manifest.read_text()) if args.elf_manifest else None)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        result = {"buckets": {}, "errors": [str(exc)]}
    print(json.dumps(result, indent=2))
    return int(bool(result["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
