import unittest

from compare_welm_native_captures import compare, read_captures


FLASH = "static_kernel_FlashAttn_abcd_123_d0"
NAMES = ("kv_write",) + (FLASH,) * 48 + ("sigmoid_mul_kernel_npu",)


def log(names=NAMES, edges=(), operation="cann_ops_transformer.flash_attn_metadata.default"):
    prefix = "[time] WeLM MK capture ShapeKey(size=1, stream_idx=None, variant_label=None)"
    boundary = {"operation": operation, "reason": "AI CPU", "calls": 1, "observed_kernels": 1,
                "tensor_signatures": [[{"shape": [1], "dtype": "torch.int32"}]]}
    summary = {"torch_boundaries": [boundary], "native_globals": [], "native_collectives": []}
    lines = [
        f"{prefix}: {len(names) + 1} kernels, 1 scopes, {len(names)} fused kernels, "
        f"1 Torch boundaries, {len(edges)} tile edges",
        f"{prefix} scope 0 kernels {names!r} coarse edges [('kept', '{FLASH}')] tile ratios {{}}",
        f"{prefix} boundaries {summary!r}",
    ]
    if edges:
        lines.insert(1, f"{prefix} scope 0 tile edges {list(edges)!r}")
    return lines


class NativeCaptureComparisonTest(unittest.TestCase):
    def check(self, coarse, fine, strict=False):
        return compare(read_captures(coarse, [1]), read_captures(fine, [1]), [1], strict)

    def test_direct_edges_only_and_direction(self):
        edges = [("kv_write", FLASH), (FLASH, "sigmoid_mul_kernel_npu")]
        result = self.check(log(), log(edges=edges), strict=True)
        self.assertEqual(result["errors"], [])
        summary = result["buckets"][1]["fine"]
        self.assertEqual(summary["flash_kernels"], 48)
        self.assertEqual(summary["flash_inputs"], {"kv_write": 1})
        self.assertEqual(summary["flash_outputs"], {"sigmoid_mul_kernel_npu": 1})

    def test_other_tile_edges_do_not_count_as_flash_edges(self):
        fine = log(edges=[("kv_write", "sigmoid_mul_kernel_npu")])
        self.assertEqual(self.check(log(), fine)["errors"], [])
        self.assertTrue(self.check(log(), fine, strict=True)["errors"])

    def test_mismatches_and_split_kernels_fail(self):
        cases = [
            (log(edges=[("kv_write", FLASH)]), log()),
            (log(), log(names=tuple(reversed(NAMES)))),
            (log(), log(names=NAMES[1:-1] + ("_merge_page_attention",))),
            (log(operation="triton::_prepare_flash_pages"), log(operation="triton::_prepare_flash_pages")),
            (log(), log(operation="aten.copy_.default")),
            (log(names=NAMES[1:4]), log(names=NAMES[1:4])),
        ]
        for coarse, fine in cases:
            with self.subTest(coarse=coarse[0], fine=fine[0]):
                self.assertTrue(self.check(coarse, fine)["errors"])

    def test_missing_or_incomplete_debug_logs_fail(self):
        for lines in ([], log()[:1], log()[:-1], log() + log()):
            with self.subTest(lines=len(lines)):
                with self.assertRaises(ValueError):
                    read_captures(lines, [1])

    def test_named_elf_evidence_for_recompiled_specializations(self):
        import copy
        a, b = "static_kernel_Fill_a", "static_kernel_Fill_b"
        coarse = read_captures(log(names=(*NAMES, a)), [1])
        fine = read_captures(log(names=(*NAMES, b)), [1])
        def artifact(name, hash_digit):
            return {"ELF_sha256": hash_digit * 64,
                    "json": {"sha256": hash_digit * 64, "kernelName": name,
                             "sub_operator_op_type": "Fill", "blockDim": 1, "supportInfo": {"opMode": "static"}},
                    "sections": {".text": ["0x6", 96, "a" * 64],
                                 ".ascendc_tiling.Fill": ["0x2", 16, "b" * 64],
                                 ".rela.text": ["0x40", 24, "c" * 64]}}
        evidence = {a: artifact(a, "d"), b: artifact(b, "e")}
        result = compare(coarse, fine, [1], elf_manifest=evidence)
        self.assertFalse(result["errors"])
        self.assertEqual(result["buckets"][1]["verified_specializations"], [[a, b]])
        for field in ("code", "tiling", "relocation", "launch", "identity"):
            changed = copy.deepcopy(evidence)
            record = changed[b]
            if field == "launch":
                record["json"]["blockDim"] = 2
            elif field == "identity":
                record["json"]["kernelName"] = a
            else:
                section = {"code": ".text", "tiling": ".ascendc_tiling.Fill", "relocation": ".rela.text"}[field]
                record["sections"][section][-1] = "f" * 64
            with self.subTest(field=field):
                self.assertTrue(compare(coarse, fine, [1], elf_manifest=changed)["errors"])

    def test_summary_cannot_claim_unlogged_tile_edges(self):
        lines = log(edges=[("kv_write", FLASH)])
        with self.assertRaises(ValueError):
            read_captures([line for line in lines if "scope 0 tile edges" not in line], [1])


if __name__ == "__main__":
    unittest.main()
