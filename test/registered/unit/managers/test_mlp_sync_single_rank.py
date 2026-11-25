"""Keep the idle record intact when a single-rank collective writes its result."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.managers.scheduler_components import dp_attn


class SingleRankMLPSyncTest(unittest.TestCase):
    def test_active_and_idle_single_rank(self):
        for reduce in (False, True):
            for active in (False, True):
                info = dp_attn.MLPSyncBatchInfo(
                    dp_size=1, tp_size=1, cp_size=1, num_tokens=7, num_tokens_for_logprob=3,
                    num_requests=2, can_run_decode_cuda_graph=True, can_run_prefill_cuda_graph=False,
                    is_extend_in_batch=False, local_can_run_tbo=False, local_forward_mode=2,
                )
                group = SimpleNamespace(active_ranks_cpu=torch.tensor([int(active)]))
                with self.subTest(reduce=reduce, active=active), \
                        patch.object(dp_attn, "get_tp_group", return_value=group), \
                        patch.object(torch.distributed, "get_rank", return_value=0), \
                        patch.object(torch.distributed, "all_reduce"), \
                        patch.object(torch.distributed, "all_gather_into_tensor",
                                     side_effect=lambda out, data, **_: out.copy_(data)):
                    info.all_gather("cpu", object(), use_all_reduce=reduce)
                    self.assertEqual(info.global_num_tokens, [7 if active else 0])
                    self.assertEqual(info.global_num_requests, [2 if active else 0])
                    self.assertEqual(info.global_num_tokens_for_logprob, [3 if active else 0])


if __name__ == "__main__":
    unittest.main()
