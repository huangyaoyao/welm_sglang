"""Check the complete frozen A5 ScatterPaKvCache -> FlashAttn -> gate chain.

Use --inspect-only to list the real capture and argument slots. To exercise the
fine plan, pass the SDK host-tiler report for the installed matching Scatter ELF
as --witness. Generate this JSON with make_welm_scatter_witness.py.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import struct
from time import perf_counter_ns
from unittest.mock import patch


def pointer_slots(image, tensors):
    data = image.data
    pointers = struct.unpack_from("<" + "Q" * (len(data) // 8), data)
    return {name: [i for i, pointer in enumerate(pointers) if pointer == tensor.data_ptr()]
            for name, tensor in tensors.items()}


def run(directory, device, witness, inspect_only, shared):
    import cann_ops_transformer as cann
    import torch
    import torch_npu
    from megakernel._fixed import capture_fixed, prepare_static
    from megakernel._runtime import _CANN, _HOME, _runtime
    from megakernel._resources import parent_binary, scope_resources
    from megakernel.triton import prepare_bindings
    from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import decode_tile_scope, install_welm_tile_layouts
    from sglang.srt.layers.welmv4_npu_op import inplace_sigmoid_mul_npu
    from welm_flash_schedule import MixCohort, decode_schedule, short_decode_final_groups

    torch.npu.set_device(device)
    rng = torch.Generator().manual_seed(7284)

    def random(*shape):
        return torch.randn(shape, dtype=torch.bfloat16, generator=rng).npu()

    def ints(rows):
        return torch.tensor(rows, dtype=torch.int32).npu()

    lengths = [4, 8] * 4
    page_ids = [2, 6, 1, 4, 3, 7, 5, 8]
    slots = [p * 64 + n - 1 for p, n in zip(page_ids, lengths)]
    page_rows = [[p] + [0] * 15 for p in page_ids]
    if shared:
        page_rows[1][0] = page_rows[0][0]
        slots[1] = page_rows[0][0] * 64 + lengths[1] - 1
    table = ints(page_rows)
    slots_dev = ints(slots)
    q = random(8, 24, 256)
    key, value = random(8, 2, 256), random(8, 2, 256)
    kc, vc = random(17, 64, 2, 256), random(17, 64, 2, 256)
    gate = random(8, 24, 1)
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()
    params = dict(cu_seqlens_q=ints(range(9)), cu_seqlens_kv=None,
                  seqused_q=ints([1]*8), seqused_kv=ints(lengths), max_seqlen_q=1, max_seqlen_kv=-1,
                  mask_mode=3, win_left=-1, win_right=-1, layout_q="TND", layout_kv="PA_BBND", layout_out="TND")
    desc = cann.flash_attn_metadata(24, 2, 256, **params)
    schedule = decode_schedule(desc.cpu().tolist())
    owners = short_decode_final_groups(schedule, lengths, query_lengths=[1]*8, window=-1,
        need_init_output=True, cohort=MixCohort(32, 64, True, True))

    def forward():
        torch_npu.npu_scatter_pa_kv_cache(key, value, kc, vc, slots_dev, cache_mode="Norm")
        out = cann.flash_attn(q, kc, vc, attn_mask=mask, block_table=table, sinks=None,
                              metadata=desc, softmax_scale=1/16, return_softmax_lse=False, **params)[0]
        inplace_sigmoid_mul_npu(gate, out)
        return out

    install_welm_tile_layouts()
    rt = _runtime()
    new_call = rt.NativeCall
    seen = []

    def record(home, sizes, ops):
        seen.append((list(sizes), deepcopy(ops)))
        return new_call(home, sizes, ops)

    with torch.inference_mode(), decode_tile_scope(), patch.dict(os.environ, {"MK_NATIVE_TILE_DEPENDENCIES":"0"}):
        with prepare_bindings():
            prepare_static(forward, directory)
        with patch.object(rt, "NativeCall", record):
            graph = capture_fixed(forward, directory, prepared=True, tile_chains=True, retain_intermediates=True)
        calls = []
        try:
            print("captured", [(scope.kernel_names, scope.dependencies) for scope in graph.plans], flush=True)
            print("boundaries", graph.execution_summary, flush=True)
            if len(seen) != 1 or len(graph.steps) != 1 or len(graph.plans) != 1:
                raise RuntimeError("expected one original three-operator MK segment")
            scope, = graph.plans
            scatter, flash, sigmoid = scope.kernel_names
            assert scatter.startswith("static_kernel_ScatterPaKvCache_") and flash.startswith("static_kernel_FlashAttn_") and sigmoid == "sigmoid_mul_kernel_npu"
            sizes, defs = seen[0]
            tasks = [graph.tasks[i] for i in scope.task_indices]
            geom = [(t["core"], t["blocks"], d["workspaces"], d["entry_banks"], d["child_entry"]) for t, d in zip(tasks, defs)]
            print("geometry", geom, "defs", defs, flush=True)
            ptrs = [pointer_slots(graph.images[i], ts) for i, ts in zip(scope.task_indices, [
                dict(key=key,value=value,kc=kc,vc=vc,slots=slots_dev),
                dict(q=q,kc=kc,vc=vc,table=table,schedule=desc,out=graph.outputs),
                dict(gate=gate,out=graph.outputs)])]
            print("packet_slots", ptrs, flush=True)
            if inspect_only:
                return
            assert ptrs == [
                {"key":[0],"value":[3],"kc":[1,5],"vc":[4,6],"slots":[2]},
                {"q":[0],"kc":[1],"vc":[2],"table":[3],"schedule":[8],"out":[9]},
                {"gate":[2],"out":[3]}]
            assert scope.dependencies == [[], [0], [1]] and scope.memory_dependencies == [[], [0], [1]]
            assert not graph.updates and not any(t.get("dynamic") or "sync_buffer" in t or t.get("replay_services") for t in tasks)
            assert [t[:2] for t in geom] == [(2,8),(3,32),(2,2)]
            assert sizes == [1,1,1,1] and len(scope.addresses[1]) == 12
            assert getattr(rt, "_capture_sdk_vector_mix_tile_inputs", False)
            from inspect_welm_kv_flash import frozen_scatter_flash_waits
            from make_welm_scatter_witness import TILING_FIELDS

            evidence = json.loads(witness.read_text())
            cert = dict(evidence["cert"])
            raw_fields = dict(zip(TILING_FIELDS, struct.unpack("<41q", bytes.fromhex(evidence["tiling_data_hex"])), strict=True))
            if cert["tiling"] != raw_fields:
                raise ValueError("normalized fields differ from the SDK tiler payload")
            desc_path = Path(evidence["descriptor"])
            compiled = json.loads(desc_path.read_text())
            if (compiled["kernelName"] != scatter or compiled["blockDim"] != 8 or compiled["core_type"] != "AIV"
                    or hashlib.sha256(Path(evidence["compiler_input"]).read_bytes()).hexdigest() != evidence["compiler_input_sha256"]
                    or hashlib.sha256(desc_path.with_suffix(".o").read_bytes()).hexdigest() != evidence["elf_sha256"]):
                raise RuntimeError("actual scatter ELF does not match the SDK tiler witness")
            cert.update(key_shape=list(key.shape), value_shape=list(value.shape), scatter_cache_ptrs=[kc.data_ptr(), vc.data_ptr()], flash_cache_ptrs=[kc.data_ptr(), vc.data_ptr()],
                        slot_mapping=slots_dev.cpu().tolist(), block_table=table.cpu().tolist(), kv_lengths=params["seqused_kv"].cpu().tolist())
            try:
                frozen_scatter_flash_waits(cert, owners, 32)
            except ValueError:
                if not shared:
                    raise
                q.copy_(random(*q.shape)); value.copy_(random(*value.shape))
                expected=forward().cpu(); graph.outputs.fill_(float("nan")); torch.npu.synchronize()
                torch.testing.assert_close(graph.replay().cpu(), expected, atol=0,rtol=0)
                print("shared-page graph retained coarse waits and matched eager",flush=True)
                return
            if shared:
                raise AssertionError("cross-row written page admitted")
            configs = {}
            raw = torch_npu._C._npu_getCurrentRawStream(device)
            ops = deepcopy(defs)
            ops[0].update(vector_group_tiles=(8,2),tile_inputs=[False])
            ops[1].update(mix_group_tiles=(32,2),tile_inputs=[dict(groups_per_source=2,idle_groups="wait_all")])
            ops[2]["tile_inputs"] = [dict(source_span=12,consumer_span=128,extent=192,idle_source_blocks=16)]
            for position, fields in [(0, {"vector_group_tiles":(7,2)}),
                                     (1, {"tile_inputs":[dict(groups_per_source=2,idle_groups="skip")]})]:
                invalid=deepcopy(ops)
                invalid[position].update(fields)
                try:
                    new_call(str(_HOME),[1,8,32,2],invalid)
                except ValueError as exc:
                    print("expected prelaunch rejection",str(exc),flush=True)
                else:
                    raise AssertionError("invalid source geometry or idle-group wait admitted")
            resources = scope_resources(tasks)
            for name, use_input, use_output in (
                    ("coarse", False, False), ("input", True, False),
                    ("output", False, True), ("both", True, True)):
                arm = deepcopy(ops)
                if not use_input:
                    arm[1]["tile_inputs"] = [False]
                if not use_output:
                    arm[2]["tile_inputs"] = [False]
                # Each measured call, including the baseline, is built with the
                # same original block geometry, tokens, and banked entry points.
                call = new_call(str(_HOME), [1, 8, 32, 2], arm)
                if resources:
                    call._set_resources(parent_binary(_HOME, _CANN, resources), resources["dynamic_ub_bytes"])
                call._validate_vector_mix_entry_queues()
                print("canonical 96-lane queues verified", name, flush=True)
                call._prepare(scope.addresses, raw)
                calls.append(call)
                configs[name] = rt._PreparedSequence([call])
            refs=(kc.clone(), vc.clone())
            for i in range(8):
                for row in (key, value, q): row.copy_(random(*row.shape))
                kc.copy_(refs[0]); vc.copy_(refs[1])
                expected=forward().cpu()
                expected_cache=(kc.cpu(), vc.cpu())
                for name, seq in configs.items():
                    kc.copy_(refs[0]); vc.copy_(refs[1]); graph.outputs.fill_(float("nan")); torch.npu.synchronize()
                    with patch.object(graph,"_sequence",seq): got=graph.replay().cpu()
                    torch.testing.assert_close(got,expected,atol=0,rtol=0)
                    torch.testing.assert_close(kc.cpu(),expected_cache[0],atol=0,rtol=0)
                    torch.testing.assert_close(vc.cpu(),expected_cache[1],atol=0,rtol=0)
                    print("case",i,name,"passed",flush=True)
            def sample(seq):
                torch.npu.synchronize()
                before=perf_counter_ns()
                for _ in range(128):seq.run(raw)
                torch.npu.synchronize()
                return (perf_counter_ns()-before)/128000
            for seq in configs.values(): sample(seq)
            samples={name:[] for name in configs}
            names=list(configs)
            for i in range(8):
                shift = (i // 2) % len(names)
                rotated = names[shift:] + names[:shift]
                order = rotated if i % 2 == 0 else reversed(rotated)
                for name in order:
                    samples[name].append(sample(configs[name]))
            print(json.dumps(dict(samples_us=samples,median_us={n:median(s) for n,s in samples.items()},
                                  replays_per_sample=128,changed_kv_cases=8)),flush=True)
        finally:
            torch.npu.synchronize()
            calls.clear()
            graph.reset()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir",type=Path,required=True)
    parser.add_argument("--device",type=int,default=0)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--witness",type=Path,help="SDK host-tiler report for the captured B8 Scatter binary")
    source.add_argument("--inspect-only",action="store_true")
    args=parser.parse_args()
    for scenario in ("independent",) if args.inspect_only else ("independent","shared"):
        directory=args.build_dir/scenario
        directory.mkdir(parents=True,exist_ok=True)
        run(directory,args.device,args.witness,args.inspect_only,scenario=="shared")
