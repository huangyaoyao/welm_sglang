import unittest

from inspect_welm_kv_flash import frozen_scatter_flash_waits, scatter_inputs_by_row, scatter_inputs_by_flash_group


def capture():
    return dict(
        tiling_key=141, block_dim=2, dtype="bfloat16", layout="PA_BBND",
        key_shape=[3, 2, 256], value_shape=[3, 2, 256],
        tiling=dict(usedCoreNum=2, blockFactor=2, tailBlockFactor=1,
                    numBlocks=6, blockSize=64, numTokens=0, numHead=2,
                    kHeadSize=256, vHeadSize=256, kHandleNumPerCore=512,
                    vHandleNumPerCore=512,
                    **{f"{cache}CacheStride{i}": stride
                       for cache in ("key", "value") for i, stride in enumerate((32768, 512, 256, 1))}),
        scatter_cache_ptrs=[10000, 900000], flash_cache_ptrs=[10000, 900000],
        slot_mapping=[64, 65, 128], block_table=[[1, 2], [2, -1], [-1, -1]],
        kv_lengths=[65, 3, 0],
    )


class ScatterFlashInputsTest(unittest.TestCase):
    def test_page_sharing_tail_and_padded_row(self):
        self.assertEqual(scatter_inputs_by_row(**capture()), [[0, 1], [1], []])

    def test_changed_swa_table_and_idle_writer(self):
        inputs = capture()
        inputs.update(slot_mapping=[64, -1, 128], block_table=[[2, 3], [4, 1], [6, 6]])
        self.assertEqual(scatter_inputs_by_row(**inputs), [[1], [], []])

    def test_padding_can_write_shared_sentinel_page(self):
        inputs = capture()
        inputs.update(slot_mapping=[64, 65, 0], block_table=[[1, 0], [1, -1], [0, -1]])
        self.assertEqual(scatter_inputs_by_row(**inputs), [[0, 1], [0], []])

    def test_groups_join_physical_writers_for_all_owned_rows(self):
        # One MIX group serves both halves of the second row and part of the first.
        owners = {(0, 0): 3, (0, 1): 1, (1, 0): 1, (1, 1): 1, (2, 0): 2, (2, 1): 2}
        self.assertEqual(scatter_inputs_by_flash_group([[0], [1], [0, 1]], owners),
                         {1: [0, 1], 2: [0, 1], 3: [0]})
        del owners[(2, 1)]
        with self.assertRaises(ValueError):
            scatter_inputs_by_flash_group([[0], [1], [0]], owners)

    def test_frozen_b8_physical_waits(self):
        opts = capture()
        opts["tiling"].update(usedCoreNum=8, blockFactor=1, tailBlockFactor=1, numTokens=0, numBlocks=16)
        opts.update(block_dim=8, key_shape=[8, 2, 256], value_shape=[8, 2, 256], scatter_cache_ptrs=[10000, 2000000], flash_cache_ptrs=[10000, 2000000],
                    slot_mapping=[64 * i + 6 for i in range(8)], block_table=[[i] for i in range(8)],
                    kv_lengths=[7] * 8)
        owners = {(r, h): 2 * r + h for r in range(8) for h in range(2)}
        waits = frozen_scatter_flash_waits(opts, owners, 32)
        self.assertEqual(waits[:16], [[group // 2] for group in range(16)])
        self.assertEqual(waits[16:], [list(range(8))] * 16)
        # A physically shared prefix requires waiting for a second row's writer.
        bad = opts | {"block_table": [[1, 0], *opts["block_table"][1:]], "kv_lengths": [71] + [7] * 7}
        with self.assertRaisesRegex(ValueError, "another scatter"):
            frozen_scatter_flash_waits(bad, owners, 32)
        for change in ({"slot_mapping": [5] + opts["slot_mapping"][1:]},
                       {"kv_lengths": [0] + opts["kv_lengths"][1:]},
                       {"tiling": opts["tiling"] | {"blockFactor": 2}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                frozen_scatter_flash_waits(opts | change, owners, 32)
        with self.assertRaises(ValueError):
            frozen_scatter_flash_waits(opts, owners | {(0, 0): 2}, 32)

    def test_rejects_unmatched_geometry_or_identity(self):
        for changes in (dict(tiling_key=700), dict(block_dim=4), dict(dtype="int8"),
                        dict(layout="NZ"), dict(key_shape=[1, 2, 256]), dict(value_shape=[3, 1, 512]), dict(flash_cache_ptrs=[10000, 990000]),
                        dict(block_table=[[9, 1], [2], []]), dict(block_table=[[1], [2], []]),
                        dict(kv_lengths=[-1, 0, 0]), dict(slot_mapping=[64, 999, 128]),
                        dict(tiling=capture()["tiling"] | {"tailBlockFactor": 2}),
                        dict(tiling=capture()["tiling"] | {"valueCacheStride1": 42})):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                scatter_inputs_by_row(**(capture() | changes))
