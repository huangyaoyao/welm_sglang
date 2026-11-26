"""List the actual indexed waits of a captured WeLM BF16 decode, with decoder layer ownership."""

import argparse
import csv
import sys
from pathlib import Path

from compare_welm_native_captures import read_captures


INPUT_NORMS = {
    "_welmv4_fused_rms_norm_true_true_kernel_npu",
    "_welmv4_fused_rms_norm_false_false_kernel_npu",
}
FIELDS = ("bs", "scope", "source_id", "target_id", "source_region", "target_region",
          "source_kernel", "target_kernel", "wait", "origin")


def family(name):
    return name.split("_", 3)[2] if name.startswith("static_kernel_") else name


def layer_dependencies(cap, expected_layers):
    kernels, indexed = cap["kernels"], cap["indexed"]
    if set(kernels) != set(indexed):
        raise ValueError("Every scope needs its indexed dependencies from the DEBUG capture log")
    locations = [(scope, i) for scope in sorted(kernels) for i in range(len(kernels[scope]))]
    names = [family(kernels[s][i]) for s, i in locations]
    starts = [i for i, name in enumerate(names) if name in INPUT_NORMS]
    if len(starts) != expected_layers or not starts:
        raise ValueError(f"Expected {expected_layers} decoder input norms, found {len(starts)}")
    regions = ["prefix"] * len(names)
    for layer, start in enumerate(starts):
        stop = starts[layer + 1] if layer + 1 < len(starts) else len(names)
        operations = names[start:stop]
        if operations.count("FlashAttn") != 1 or operations.count("MoeFinalizeRoutingV2") != 1:
            raise ValueError(f"Ambiguous native attention or routed MoE in decoder layer {layer}")
        flash = names.index("FlashAttn", start, stop)
        end = names.index("MoeFinalizeRoutingV2", start, stop) + 1
        if end <= flash:
            raise ValueError(f"Invalid native attention/MoE order in decoder layer {layer}")
        if end < stop and names[end] == "Add":
            end += 1
        if layer + 1 < len(starts) and (end != stop or names[end - 1] != "Add"):
            raise ValueError(f"Unrecognized decoder output at layer {layer}")
        # The last layer can return its experts separately. Their FP32 rebuild,
        # final normalization and logits then belong to the model epilogue.
        regions[start:end] = [f"layer{layer}"] * (end-start)
        if layer == len(starts) - 1:
            regions[end:] = ["suffix"] * (len(names)-end)
    owners = dict(zip(locations, regions))
    for scope in sorted(kernels):
        for src, dst, wait, origin in indexed[scope]:
            yield dict(scope=scope, source_id=src, target_id=dst,
                       source_region=owners[scope, src], target_region=owners[scope, dst],
                       source_kernel=kernels[scope][src], target_kernel=kernels[scope][dst],
                       wait=wait, origin=origin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server_log", type=Path)
    parser.add_argument("--buckets", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--layers", type=int, default=48)
    args = parser.parse_args()
    with args.server_log.open() as stream:
        captures = read_captures(stream, args.buckets)
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    for bs in args.buckets:
        for row in layer_dependencies(captures[bs], args.layers):
            writer.writerow({"bs": bs, **row})


if __name__ == "__main__":
    main()
