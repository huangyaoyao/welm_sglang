"""A5 device fixture: frozen CANN Flash + original sigmoid, same kernels in both arms."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from statistics import median
import struct
from time import perf_counter_ns
from unittest.mock import patch


def exercise(directory, device):
    import cann_ops_transformer as cann
    import torch
    import torch_npu
    from megakernel._fixed import capture_fixed, prepare_static
    from megakernel._resources import parent_binary, scope_resources
    from megakernel._runtime import _CANN, _HOME, _runtime
    from megakernel.triton import prepare_bindings
    from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import (
        decode_tile_scope, install_welm_tile_layouts,
    )
    from sglang.srt.layers.welmv4_npu_op import inplace_sigmoid_mul_npu
    from welm_flash_schedule import MixCohort, decode_schedule, original_gate_inputs, short_decode_final_groups

    torch.npu.set_device(device)
    assert torch.npu.get_device_name().startswith("Ascend950")
    rt = _runtime()
    if not getattr(rt, "_capture_mix_group_tile_inputs", False) or not getattr(rt, "_capture_span_tile_inputs", False):
        raise RuntimeError("A5 native_torch_rt does not support the banked MIX group and span tile ports")
    rng = torch.Generator().manual_seed(1051)
    def values(*shape):
        return torch.randn(shape, generator=rng, dtype=torch.bfloat16).npu()

    lengths = [4, 8] * 4
    q = values(8, 24, 256)
    k, v = values(34, 64, 2, 256), values(34, 64, 2, 256)
    gate = values(8, 24, 1)
    table = (torch.arange(20, dtype=torch.int32).repeat(8, 1) * 3
             + torch.arange(8, dtype=torch.int32)[:, None]).remainder(34).npu()
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()
    params = dict(cu_seqlens_q=torch.arange(9, dtype=torch.int32).npu(), cu_seqlens_kv=None,
                  seqused_q=torch.ones(8, dtype=torch.int32).npu(), seqused_kv=torch.tensor(lengths, dtype=torch.int32).npu(),
                  max_seqlen_q=1, max_seqlen_kv=-1, mask_mode=3, win_left=-1, win_right=-1,
                  layout_q="TND", layout_kv="PA_BBND", layout_out="TND")
    # Kept outside capture. All schedule/length/page tensors and the input KV
    # stay immutable for both plans. Only Q changes between correctness replays.
    desc = cann.flash_attn_metadata(24, 2, 256, **params)
    schedule = decode_schedule(desc.cpu().tolist())

    def forward():
        out = cann.flash_attn(q, k, v, attn_mask=mask, block_table=table, sinks=None,
                              metadata=desc, softmax_scale=1/16, return_softmax_lse=False, **params)[0]
        inplace_sigmoid_mul_npu(gate, out)
        return out

    install_welm_tile_layouts()
    seen = []
    native_call = rt.NativeCall

    def capture_definition(home, sizes, ops):
        seen.append((list(sizes), deepcopy(ops)))
        return native_call(home, sizes, ops)

    with torch.inference_mode(), decode_tile_scope(), patch.dict(os.environ, {"MK_NATIVE_TILE_DEPENDENCIES": "0"}):
        with prepare_bindings():
            prepare_static(forward, directory)
        with patch.object(rt, "NativeCall", capture_definition):
            graph = capture_fixed(forward, directory, prepared=True, tile_chains=True, retain_intermediates=True)
        fine = None
        try:
            if len(seen) != 1 or len(graph.plans) != 1 or graph.host_boundary_count or len(graph.steps) != 1:
                raise RuntimeError(f"expected one native scope; scopes={[p.kernel_names for p in graph.plans]} "
                                   f"boundaries={graph.execution_summary}")
            coarse, = graph.plans
            assert len(coarse.kernel_names) == 2, coarse.kernel_names
            flash_name, gate_name = coarse.kernel_names
            assert flash_name.startswith("static_kernel_FlashAttn_") and gate_name == "sigmoid_mul_kernel_npu", coarse.kernel_names
            assert coarse.dependencies == [[], [0]] and coarse.memory_dependencies == [[], [0]], coarse.dependencies
            assert not coarse.tile_dependencies
            flash, sigmoid = [graph.tasks[i] for i in coarse.task_indices]
            assert not graph.updates and not any(t.get("dynamic") or "sync_buffer" in t or t.get("replay_services")
                                                 for t in (flash, sigmoid)), "requires static tasks without services"
            sizes, defs = seen[0]
            print("captured", flash_name, gate_name,
                  [(op["type"], op["blocks"], op["entry_banks"], op["child_entry"]) for op in defs],
                  "resources", bool(scope_resources([flash, sigmoid])), flush=True)
            assert sizes == [1, 1, 1] and defs[0]["inputs"] == [0] and defs[1]["inputs"] == [1], defs
            assert flash["blocks"] == defs[0]["blocks"] == 32 and flash["vectors_per_group"] == 2
            assert flash["core"] == 3 and sigmoid["core"] == 2 and sigmoid["blocks"] == defs[1]["blocks"] == 2
            if not defs[0]["entry_banks"] or len(coarse.addresses[0]) != 12:
                raise RuntimeError("captured Flash lacks the banked SDK MIX calling convention")
            assert defs[1]["child_entry"] and not defs[0]["child_entry"] and defs[0]["workspaces"] == 1
            assert sigmoid["provider"] == "triton" and sigmoid.get("binding")
            assert schedule.header[4:6] == (32, 64)
            owners = short_decode_final_groups(schedule, lengths, query_lengths=[1]*8, window=-1,
                need_init_output=True, cohort=MixCohort(32, 64, True, True))
            assert all(owners[r, h] == 2*r+h for r in range(8) for h in (0, 1)), owners
            waits = original_gate_inputs(owners, 8)
            assert waits == (tuple(range(11)), tuple(range(10, 16))), waits
            defs[0].update(mix_group_tiles=(32, 2), tile_inputs=[False])
            defs[1]["tile_inputs"] = [dict(source_span=12, consumer_span=128, extent=192, idle_source_blocks=16)]
            fine = native_call(str(_HOME), [1, 32, 2], defs)
            resources = scope_resources([flash, sigmoid])
            if resources:
                fine._set_resources(parent_binary(_HOME, _CANN, resources), resources["dynamic_ub_bytes"])
            raw = torch_npu._C._npu_getCurrentRawStream(device)
            fine._prepare(coarse.addresses, raw)
            sequences = {"coarse": graph._sequence, "fine":rt._PreparedSequence([fine])}
            out_ptr = graph.outputs.data_ptr()
            # SDK Flash may have opaque access metadata. Check the actual
            # fixed argument packets that coarse and fine share before using
            # the scheduler's output ownership relation in this fixture.
            packet_slots = {}
            for index, tensors in zip(coarse.task_indices,
                    ({"q": q, "k": k, "v": v, "table": table, "schedule": desc,
                      "out": graph.outputs}, {"gate": gate, "out": graph.outputs})):
                data = graph.images[index].data
                words = struct.unpack_from("<" + "Q"*(len(data)//8), data)
                slots = {name:[j for j, pointer in enumerate(words) if pointer == t.data_ptr()]
                         for name,t in tensors.items()}
                packet_slots[index] = slots
            print("packet_slots", packet_slots, flush=True)
            assert all(found for slots in packet_slots.values() for found in slots.values()), packet_slots
            for case in range(8):
                q.copy_(values(*q.shape))
                expected = forward().cpu()
                assert torch.isfinite(expected).all()
                for arm, sequence in sequences.items():
                    graph.outputs.fill_(float("nan"))
                    # torch_npu may enqueue fills through its host-side task
                    # queue; complete the sentinel before direct ACL submission.
                    torch.npu.synchronize()
                    with patch.object(graph, "_sequence", sequence):
                        actual = graph.replay()
                    assert actual.data_ptr() == out_ptr
                    got = actual.cpu()
                    print("case",case,arm,"finite",torch.isfinite(got).sum().item(),
                          "match",torch.eq(got,expected).sum().item(), flush=True)
                    torch.testing.assert_close(got, expected, atol=0, rtol=0)
            samples = {"coarse": [], "fine": []}
            repetitions = 128
            def sample(sequence):
                # Direct ACL submissions and torch_npu event recording can use
                # different host queues. Drain both ends and report wall time.
                torch.npu.synchronize()
                start = perf_counter_ns()
                for _ in range(repetitions):
                    sequence.run(raw)
                torch.npu.synchronize()
                return (perf_counter_ns() - start) / (1000 * repetitions)
            for sequence in sequences.values():
                sample(sequence)
            for repeat in range(8):
                order = tuple(sequences.items())
                for arm, call in (order if repeat % 2 == 0 else reversed(order)):
                    samples[arm].append(sample(call))
            print(json.dumps(dict(flash=flash_name, sigmoid=gate_name, direct_flash_to_gate_edges=1,
                                  gate_wait_groups=waits, bitwise_eager_cases=8, timing="synchronized_wall", launches_per_sample=repetitions, samples_us=samples,
                                  median_us={arm:median(rows) for arm, rows in samples.items()})), flush=True)
        finally:
            torch.npu.synchronize()
            del fine
            graph.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    args.build_dir.mkdir(parents=True, exist_ok=True)
    exercise(args.build_dir, args.device)
