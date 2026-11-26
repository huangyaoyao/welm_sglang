import unittest

from compare_welm_native_captures import compare, read_captures


FLASH = "static_kernel_FlashAttn_abcd_123_d0"
SCATTER = "static_kernel_ScatterPaKvCache_abcd"
GATE = "sigmoid_mul_kernel_npu"
FLASHES = (FLASH,) + tuple(f"{FLASH}_layer{i}" for i in range(47))
NAMES = ("kv_write",) + FLASHES + (GATE,)


def log(names=NAMES, edges=(), operation="cann_ops_transformer.flash_attn_metadata.default"):
    return scoped_log([(names, (), edges)], operation)


def scoped_log(scopes, operation="cann_ops_transformer.flash_attn_metadata.default"):
    prefix = "[time] WeLM MK capture ShapeKey(size=1, stream_idx=None, variant_label=None)"
    boundary = {"operation": operation, "reason": "AI CPU", "calls": 1, "observed_kernels": 1,
                "tensor_signatures": [[{"shape": [1], "dtype": "torch.int32"}]]}
    summary = {"torch_boundaries": [boundary], "native_globals": [], "native_collectives": []}
    count = sum(len(names) for names, _, _ in scopes)
    tiles = sum(len(edges) for _, _, edges in scopes)
    lines = [f"{prefix}: {count + 1} kernels, {len(scopes)} scopes, {count} fused kernels, 1 Torch boundaries, {tiles} tile edges"]
    for i, (names, full, edges) in enumerate(scopes):
        lines.append(f"{prefix} scope {i} kernels {tuple(names)!r} coarse edges {list(full)!r} tile ratios {{}}")
        if edges:
            lines.append(f"{prefix} scope {i} tile edges {list(edges)!r}")
    lines.append(f"{prefix} boundaries {summary!r}")
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

    def test_coarse_flash_consumers_are_reported_separately(self):
        lines = scoped_log([(NAMES, [(FLASH, GATE)], [])])
        summary = self.check(lines, lines)["buckets"][1]["fine"]
        self.assertEqual(summary["flash_coarse_outputs"], {"sigmoid_mul_kernel_npu": 1})
        self.assertEqual(summary["flash_outputs"], {})

    def test_other_tile_edges_do_not_count_as_flash_edges(self):
        fine = log(edges=[("kv_write", "sigmoid_mul_kernel_npu")])
        self.assertEqual(self.check(log(), fine)["errors"], [])
        self.assertTrue(self.check(log(), fine, strict=True)["errors"])

    def test_edges_on_same_named_flashes_in_different_scopes_do_not_join(self):
        scopes = [((SCATTER, FLASH), (), [(SCATTER, FLASH)]),
                  ((FLASH, GATE), (), [(FLASH, GATE)]), (FLASHES[2:], (), ())]
        coarse = scoped_log([(names, (), ()) for names, _, _ in scopes])
        result = self.check(coarse, scoped_log(scopes), strict=True)
        self.assertTrue(result["errors"])
        self.assertEqual(result["buckets"][1]["fine"]["tile_flash_chains"],
                         dict(two_sided=0, scatter_inputs=1, gate_outputs=1, scatter_flash_gate=0))

    def test_ambiguous_same_scope_flash_name_cannot_pass(self):
        names = (SCATTER, FLASH, FLASH, GATE, *FLASHES[2:])
        edges = [(SCATTER, FLASH), (FLASH, GATE)]
        fine = scoped_log([(names, (), edges)])
        result = self.check(log(names), fine, strict=True)
        self.assertTrue(result["errors"])
        self.assertEqual(result["buckets"][1]["fine"]["tile_flash_chains"]["two_sided"], 0)

    def test_repeated_flash_specialization_uses_scope_local_ids(self):
        names = (SCATTER, FLASH, FLASH, GATE, *FLASHES[2:])
        edges = [(SCATTER, FLASH), (FLASH, GATE)]
        lines = scoped_log([(names, (), edges)])
        prefix = lines[0].split(": ")[0]
        for flash, count in ((1, 1), (2, 0)):
            with self.subTest(flash=flash):
                records = [(0, 1, "tile", "storage"), (flash, 3, "tile", "storage")]
                fine = lines + [f"{prefix} scope 0 indexed dependencies {records!r}"]
                result = self.check(log(names), fine, strict=True)
                stats = result["buckets"][1]["fine"]["tile_flash_chains"]
                self.assertEqual(stats["scatter_flash_gate"], count)
                self.assertEqual(bool(result["errors"]), count == 0)

    def test_exclusive_and_embedded_chains_count_instances(self):
        native_edges = [(SCATTER, FLASH), (FLASH, GATE)]
        names = (SCATTER, FLASH, GATE)
        layout = [names, ("extra", *names), FLASHES[2:]]
        coarse = scoped_log([(k, native_edges if i < 2 else (), ()) for i, k in enumerate(layout)])
        fine = scoped_log([(k, (), native_edges if i < 2 else ()) for i, k in enumerate(layout)])
        result = self.check(coarse, fine, strict=True)
        self.assertFalse(result["errors"])
        summary = result["buckets"][1]
        self.assertEqual(summary["fine"]["exclusive_scatter_flash_gate_scopes"], 1)
        self.assertEqual(summary["coarse"]["coarse_flash_chains"]["scatter_flash_gate"], 2)
        self.assertEqual(summary["fine"]["tile_flash_chains"]["scatter_flash_gate"], 2)

    def test_indexed_edges_match_names_and_keep_repeated_instances(self):
        names = ("src", "src", FLASH, GATE, *FLASHES[1:])
        lines = scoped_log([(names, [("src", FLASH)], [(FLASH, GATE)])])
        prefix = lines[0].split(": ")[0]
        records = [(1, 2, "full", "stream"), (2, 3, "tile", "storage")]
        def parse(entries):
            return read_captures(lines + [f"{prefix} scope 0 indexed dependencies {entries!r}"], [1])
        cap = parse(records)
        self.assertEqual(cap[1]["indexed"][0], records)
        for invalid in (records[1:], records + records[:1], [(4, 2, "full", "stream"), records[1]]):
            with self.assertRaises(ValueError):
                parse(invalid)

    def test_different_flashes_in_one_scope_cannot_join(self):
        edges = [("kv_write", FLASHES[0]), (FLASHES[1], GATE)]
        result = self.check(log(), log(edges=edges), strict=True)
        self.assertTrue(result["errors"])
        self.assertEqual(result["buckets"][1]["fine"]["tile_flash_chains"]["two_sided"], 0)

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
