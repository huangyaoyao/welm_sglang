import unittest

from inspect_welm_kv_flash import scatter_inputs_by_row, scatter_inputs_by_flash_group


def capture():
    return dict(
        tiling_key=141, block_dim=2, dtype="bfloat16", layout="PA_BBND",
        tiling=dict(usedCoreNum=2, blockFactor=2, tailBlockFactor=1,
                    numBlocks=6, blockSize=64, numTokens=3, numHead=2,
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

    def test_rejects_unmatched_geometry_or_identity(self):
        for changes in (dict(tiling_key=700), dict(block_dim=4), dict(dtype="int8"),
                        dict(layout="NZ"), dict(flash_cache_ptrs=[10000, 990000]),
                        dict(block_table=[[9, 1], [2], []]), dict(block_table=[[1], [2], []]),
                        dict(kv_lengths=[-1, 0, 0]), dict(slot_mapping=[64, 999, 128]),
                        dict(tiling=capture()["tiling"] | {"tailBlockFactor": 2}),
                        dict(tiling=capture()["tiling"] | {"valueCacheStride1": 42})):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                scatter_inputs_by_row(**(capture() | changes))
