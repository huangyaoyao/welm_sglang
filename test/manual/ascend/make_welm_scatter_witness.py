"""Replay the CANN 9.2.1 A5 ScatterPaKvCache host tiler for a captured static package.

After sourcing the same CANN environment as the worker, run this with its Python:

    python make_welm_scatter_witness.py --compiler-input PATH/compiler_input.json \
        --descriptor PATH/static_kernel_ScatterPaKvCache.json --output witness.json

Keep the compiler's package layout, including config/ascend950/shape_info,
binary_info_config.json and the ELF adjacent to the descriptor. The resulting
JSON is accepted by test_welm_scatter_flash_group_port.py --witness. This
command reads the host compiler inputs and installed SDK without launching NPU
work. Runtime pointers, pages, and actual FA sections belong to the replay.
"""

import argparse
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import struct

# ScatterPaKvCacheTilingData in ops-transformer/.../op_host/scatter_pa_kv_cache_tiling.h
TILING_FIELDS = (
    "usedCoreNum blockFactor tailBlockFactor kHandleNumPerCore vHandleNumPerCore "
    "kTailHandleNum vTailHandleNum kLoopNum vLoopNum kHandleNumPerLoop vHandleNumPerLoop "
    "keyStride0 keyStride1 keyStride2 valueStride0 valueStride1 valueStride2 kHeadSize vHeadSize "
    "batch numBlocks blockSize seqLen numHead numTokens ubSize kStride vStride kOffset vOffset "
    "keyCacheStride0 keyCacheStride1 keyCacheStride2 keyCacheStride3 "
    "valueCacheStride0 valueCacheStride1 valueCacheStride2 valueCacheStride3 "
    "kCacheBlockStride vCacheBlockStride ubBlockFactor"
).split()


def read_json(path):
    return json.loads(path.read_bytes())


def generate(compiler_input, descriptor):
    cfg = read_json(compiler_input)
    desc = read_json(descriptor)
    elf = descriptor.with_suffix(".o")
    digest = sha256(elf.read_bytes()).hexdigest()
    if (desc["sub_operator_op_type"] != "ScatterPaKvCache" or desc["core_type"] != "AIV"
            or elf.name != desc["binFileName"] + desc["binFileSuffix"] or desc["sha256"] != digest):
        raise ValueError("ScatterPaKvCache descriptor and AIV ELF must match")

    # A package can contain many other ops. Reject ambiguity about which
    # Scatter compiler input emitted this ELF, even when its blockDim matches.
    manifest = descriptor.parents[2] / "config" / descriptor.parent.parent.name / "binary_info_config.json"
    shape_dir = manifest.parent / "shape_info"
    inputs = [p for p in shape_dir.glob("*compiler_input*.json") if read_json(p).get("op_type") == "ScatterPaKvCache"]
    records = read_json(manifest)["ScatterPaKvCache"]["staticList"]
    if (inputs != [compiler_input] or len(records) != 1
            or records[0]["binDesc"]["kernelName"] != desc["kernelName"]
            or Path(records[0]["binPath"]).parts[-3:] != elf.parts[-3:]):
        raise ValueError("expected one matching scatter specialization in the compiler's static package")

    key, key_cache, slots, value, value_cache, *optional = cfg["inputs"]
    batch = key["shape"][0]
    attrs = {a["name"]: a["value"] for a in cfg["attrs"]}
    if (batch not in (1, 2, 4, 8) or key["shape"] != [batch, 2, 256] or value["shape"] != key["shape"]
            or key_cache["shape"][1:] != [64, 2, 256] or value_cache["shape"] != key_cache["shape"]
            or slots["shape"] != [batch] or slots["dtype"] != "int32" or any(optional)
            or any(t["dtype"] != "bfloat16" for t in (key, value, key_cache, value_cache))
            or attrs != {"cache_mode": "Norm", "scatter_mode": "None", "strides": [1, 1], "offsets": [0, 0]}):
        raise ValueError("unsupported WeLM BF16 dual KV scatter geometry")
    if (desc["platformInfo"]["cubeCoreCnt"], desc["platformInfo"]["vectorCoreCnt"]) != (32, 64) or (
            cfg["platform_info"]["cube_core_cnt"], cfg["platform_info"]["vector_core_cnt"]) != (32, 64):
        raise ValueError("unexpected static package platform")

    from asc_op_compile_base.common.platform.platform_info import set_current_compile_soc_info
    from tbe.common.utils.op_tiling import do_op_tiling

    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    soc = acl.aclrtGetSocName().decode()
    if soc != "Ascend950DT_9582":
        raise ValueError(f"unsupported SoC for this witness: {soc}")
    set_current_compile_soc_info(soc, aicore_num=32)

    def tensor(t):
        if t is None:
            return None
        return dict(t, param_name=t["name"], ori_format=t["origin_format"], ori_shape=t["origin_shape"])

    result = do_op_tiling("ScatterPaKvCache", {}, list(map(tensor, cfg["inputs"])),
                          list(map(tensor, cfg["outputs"])), attrs=cfg["attrs"])
    payload = result["tiling_data"]
    if (result["tiling_key"] != 141 or result["block_dim"] != desc["blockDim"] or desc["blockDim"] != batch
            or result["workspaces"] != desc["workspace"]["size"] or len(payload) != 8 * len(TILING_FIELDS)):
        raise ValueError("host tiler did not reproduce the expected static scatter specialization")
    fields = dict(zip(TILING_FIELDS, struct.unpack("<41q", payload)))
    expected = {"usedCoreNum": batch, "blockFactor": 1, "tailBlockFactor": 1, "numHead": 2,
                "kHandleNumPerCore": 512, "vHandleNumPerCore": 512, "kHeadSize": 256, "vHeadSize": 256,
                "blockSize": 64, "numBlocks": key_cache["shape"][0]}
    for port in ("key", "value"):
        expected.update({f"{port}CacheStride{i}": stride for i, stride in enumerate((32768, 512, 256, 1))})
    if any(fields[k] != v for k, v in expected.items()):
        raise ValueError("unexpected key141 row assignment or cache layout")
    # numTokens, batch and seqLen have no row-ownership semantics on this key.
    return {
        "soc": soc, "compiler_input": str(compiler_input), "compiler_input_sha256": sha256(compiler_input.read_bytes()).hexdigest(),
        "descriptor": str(descriptor), "elf_sha256": digest, "tiling_data_hex": payload.hex(),
        "cert": {"tiling_key": result["tiling_key"], "block_dim": result["block_dim"], "dtype": "bfloat16",
                 "layout": "PA_BBND", "key_shape": key["shape"], "value_shape": value["shape"], "tiling": fields},
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compiler-input", type=Path, required=True)
    p.add_argument("--descriptor", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    witness = generate(args.compiler_input.resolve(), args.descriptor.resolve())
    args.output.write_text(json.dumps(witness, indent=2) + "\n")
    print(f"{args.output}  key={witness['cert']['tiling_key']}  blocks={witness['cert']['block_dim']}  elf={witness['elf_sha256']}")
