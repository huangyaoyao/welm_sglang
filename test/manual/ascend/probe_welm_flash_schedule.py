"""Sample the installed CANN FlashAttention schedule and bitwise-check fixed replay."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import torch
import torch_npu
import cann_ops_transformer as cann
from welm_flash_schedule import (
    MixCohort, decode_schedule, original_gate_inputs, short_decode_final_groups, validate_decode_ranges,
)

def semantic_fields(schedule):
    if schedule.dtype != torch.int32 or schedule.ndim != 1:
        raise ValueError("expected rank-one int32 FlashAttnMetadata")
    return decode_schedule(schedule.cpu().tolist())

def geometry(n, window, run_fixed, output):
    q = torch.randn(n, 24, 256, dtype=torch.bfloat16).npu()
    k = torch.randn(34, 64, 2, 256, dtype=torch.bfloat16).npu()
    v = torch.randn_like(k)
    table = torch.arange(20, dtype=torch.int32).repeat(n, 1).npu()
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()
    sinks = torch.linspace(-1, 1, 24, device='npu', dtype=torch.float32) if window > 0 else None
    used = torch.ones(n, dtype=torch.int32, device='npu')
    q_used = torch.ones_like(used)
    cu = torch.arange(n + 1, dtype=torch.int32, device='npu')
    args = dict(cu_seqlens_q=cu, cu_seqlens_kv=None, seqused_q=q_used, seqused_kv=used,
                max_seqlen_q=1, max_seqlen_kv=-1,
                mask_mode=4 if window > 0 else 3, win_left=window,
                win_right=0 if window > 0 else -1, layout_q='TND', layout_kv='PA_BBND', layout_out='TND')
    def forward():
        schedule = cann.flash_attn_metadata(24, 2, 256, **args)
        result = cann.flash_attn(q, k, v, block_table=table, sinks=sinks, attn_mask=mask,
                                 metadata=schedule, softmax_scale=1 / 16,
                                 return_softmax_lse=False, **args)[0]
        return result, schedule

    plan = None
    with tempfile.TemporaryDirectory() as build:
        try:
            if run_fixed:
                from megakernel._fixed import capture_fixed
                print('capture start', n, window, flush=True)
                plan = capture_fixed(forward, build, tile_chains=True)
                print('captured', [scope.kernel_names for scope in plan.plans], flush=True)
            for seq, spans in enumerate(((4, 8), (20, 33), (256, 65), (1025, 513), (65, 193))):
                active = (n + 1) // 2 if seq == 4 else n
                lengths = [spans[row % 2] if row < active else 0 for row in range(n)]
                used.copy_(torch.tensor(lengths, dtype=torch.int32))
                q_used.copy_(torch.tensor([1] * active + [0] * (n-active), dtype=torch.int32))
                for row in range(n):
                    table[row].copy_((torch.arange(20, dtype=torch.int32) * 3 + row + seq) % 34)
                q.copy_(torch.randn(n, 24, 256, dtype=torch.bfloat16))
                ref, sched = forward()
                ref = ref.cpu()
                expected = semantic_fields(sched)
                assert torch.isfinite(ref[:active]).all()
                if run_fixed:
                    print('replay', n, window, seq, lengths, flush=True)
                    got, sched = plan.replay()
                    torch.testing.assert_close(got.cpu()[:active], ref[:active], atol=0, rtol=0)
                    assert semantic_fields(sched) == expected
                # See FlashAttnTilingBase::CheckNeedInitOutput: this call
                # supplies both seqused_q and seqused_kv. The runtime tiling
                # packet is not exposed by flash_attn_metadata itself.
                queries = q_used.cpu().tolist()
                validate_decode_ranges(expected, lengths, queries)
                cubes, vectors = expected.header[4:6]
                required_cohort = MixCohort(cubes, vectors, True, True)
                try:
                    owners = short_decode_final_groups(
                        expected, lengths, query_lengths=queries, window=window,
                        need_init_output=True, cohort=required_cohort,
                    )
                except ValueError as exc:
                    final_groups, gate_waits, reason = None, None, str(exc)
                else:
                    final_groups = [[*key, group] for key, group in owners.items()]
                    gate_waits = original_gate_inputs(owners, n)
                    reason = None
                report = {'bs': n, 'window': window, 'lengths': lengths,
                          'mode': 'fixed' if run_fixed else 'eager',
                          'digest': hashlib.sha256(ref[:active].contiguous().view(torch.int16).numpy().tobytes()).hexdigest(),
                          'need_init_output_source_prediction': True,
                          'conditional_output_groups': final_groups,
                          'original_gate_group_inputs_if_cohort_guaranteed': gate_waits,
                          'unavailable_reason': reason,
                          **expected.as_dict()}
                with output.open('a') as stream:
                    print(json.dumps(report, separators=(',', ':')), file=stream)
        finally:
            if plan is not None:
                plan.reset()
    print('OK', n, window, run_fixed, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed', action='store_true')
    parser.add_argument('--batch', type=int, choices=(1, 2, 4, 8))
    parser.add_argument('--window', type=int, choices=(-1, 511))
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    torch.manual_seed(419)
    torch.npu.set_device(0)
    with torch.inference_mode():
        for bs in ((args.batch,) if args.batch else (1, 2, 4, 8)):
            for window in ((args.window,) if args.window is not None else (-1, 511)):
                geometry(bs, window, args.fixed, args.out)
