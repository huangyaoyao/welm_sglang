import unittest

from report_welm_layer_dependencies import layer_dependencies


def static(name):
    return "static_kernel_" + name + "_audit"


NORM = "_welmv4_fused_rms_norm_true_true_kernel_npu"
FLASH = static("FlashAttn")
END = static("MoeFinalizeRoutingV2")
ADD = static("Add")


class DecoderLayers(unittest.TestCase):
    def test_actual_edges_keep_inter_layer_and_final_fp32_rebuild_outside(self):
        cap = dict(kernels={0: ("embed", NORM, FLASH, END, ADD),
                            1: (NORM, FLASH, END, static("Cast"), ADD)},
                   indexed={0: [(0, 1, "full", "storage"), (1, 3, "tile", "storage")],
                            1: [(0, 1, "full", "stream"), (2, 3, "full", "storage")]})
        rows = list(layer_dependencies(cap, 2))
        self.assertEqual([(r["source_region"], r["target_region"]) for r in rows],
                         [("prefix", "layer0"), ("layer0", "layer0"),
                          ("layer1", "layer1"), ("layer1", "suffix")])
        self.assertEqual((rows[1]["source_id"], rows[1]["target_id"], rows[1]["wait"]),
                         (1, 3, "tile"))
        self.assertEqual(rows[2]["scope"], 1)
        self.assertEqual(rows[2]["origin"], "stream")

    def test_full_wait_or_tile_at_the_layer_handoff_has_both_owners(self):
        cap = dict(kernels={0: (NORM, FLASH, END, ADD, NORM, FLASH, END, ADD)},
                   indexed={0: [(3, 4, "tile", "storage")]})
        row, = layer_dependencies(cap, 2)
        self.assertEqual((row["source_region"], row["target_region"]), ("layer0", "layer1"))

    def test_rejects_incomplete_or_ambiguous_layer_records(self):
        for ops in ((NORM, END), (NORM, FLASH, FLASH, END), (NORM, FLASH, ADD)):
            with self.subTest(ops=ops), self.assertRaises(ValueError):
                list(layer_dependencies(dict(kernels={0: ops}, indexed={0: []}), 1))
        with self.assertRaises(ValueError):
            list(layer_dependencies(dict(kernels={0: (NORM, FLASH, END)}, indexed={}), 1))


if __name__ == "__main__":
    unittest.main()
