"""Verify WeLM's source program footprints on a host without torch or CANN."""
from dataclasses import dataclass
from math import prod
from pathlib import Path
import runpy
import unittest


CONTRACT = runpy.run_path(str(
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/hardware_backend/npu/graph_runner/welm_tile_layouts.py"
))


@dataclass(eq=False)
class Tensor:
    shape: tuple
    width: int

    def numel(self):
        return prod(self.shape)

    def element_size(self):
        return self.width


class WelmLayoutContractTest(unittest.TestCase):
    def setUp(self):
        self.k = Tensor((8, 2, 256), 2)
        self.norm = Tensor(self.k.shape, 2)
        self.gamma = Tensor((256,), 2)
        self.q = Tensor((8, 24, 256), 2)
        self.positions = Tensor((8,), 8)
        self.table = Tensor((16384, 64), 4)
        self.norm_args = dict(x=self.k, gamma=self.gamma, output=self.norm, tokens=8,
                              kv_heads=2, head_dim=256, x_stride=512, output_stride=512,
                              eps=1e-5, kv_heads_block=2)
        self.rope_args = dict(q=self.q, k=self.norm, positions=self.positions,
                              table=self.table, last_index=None, tokens=8,
                              last_query_rows=0, q_stride=6144, k_stride=512,
                              head_dim=256, rope_dim=64, stages=4, q_heads=24,
                              k_heads=2, q_heads_block=32, k_heads_block=2)

    def test_decode_rows_and_shared_reads(self):
        norm = CONTRACT["_k_norm_layout"](**self.norm_args)
        self.assertEqual(norm, dict(count=8, block=1, reads=[(self.k, 1024)],
                                    writes=[(self.norm, 1024)], shared_reads=[self.gamma]))
        rope = CONTRACT["_rope_layout"](**self.rope_args)
        rows = [(self.q, 12288), (self.norm, 1024)]
        self.assertEqual(rope, dict(count=8, block=1, reads=[*rows, (self.positions, 8)],
                                    writes=rows, shared_reads=[self.table]))

    def test_k_norm_rejects_padded_strides_and_unmatched_shapes(self):
        for override in (dict(x_stride=768), dict(output_stride=768),
                         dict(output=Tensor((8, 1, 256), 2)), dict(tokens=16),
                         dict(gamma=Tensor((128,), 2))):
            with self.subTest(override=override):
                self.assertIsNone(CONTRACT["_k_norm_layout"](**(self.norm_args | override)))

    def test_rope_excludes_mirror_indices_and_nonmatching_row_layouts(self):
        for override in (dict(last_index=self.positions), dict(last_query_rows=8),
                         dict(q=Tensor((4, 24, 256), 2)), dict(k_stride=1024),
                         dict(q_stride=8192), dict(positions=Tensor((9,), 8)),
                         dict(rope_dim=63)):
            with self.subTest(override=override):
                self.assertIsNone(CONTRACT["_rope_layout"](**(self.rope_args | override)))


if __name__ == "__main__":
    unittest.main()
