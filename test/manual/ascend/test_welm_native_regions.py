"""Host coverage for the optional block-table, OE and Flash record changes."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

SRT = Path(__file__).resolve().parents[3] / "python/sglang/srt"
DIR = SRT / "hardware_backend/npu/graph_runner"


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeRegionsTest(unittest.TestCase):
    def test_scope_is_supported_bucket_native_only_and_fine_only(self):
        layouts = load("welm_tile_layouts")
        utils = types.SimpleNamespace(get_bool_env_var=lambda key: False)
        gates = {"SGLANG_NPU_WELMV4_MK_NATIVE_REGIONS": "pages,oe,metadata",
                 "MK_NATIVE_TILE_DEPENDENCIES": "1", "MK_FIXED_NATIVE_FLASH_METADATA": "1"}
        mk = types.SimpleNamespace(_fixed=types.SimpleNamespace(SUPPORTS_WELM_DECODE_FLASH_METADATA=True))
        with patch.dict(sys.modules, {"sglang.srt.utils": utils, "megakernel": mk}), patch.dict("os.environ", gates):
            for bucket in (1, 2, 4, 8, 3, 16, None):
                for native in (False, True):
                    with layouts.decode_tile_scope(native_ops_only=native, batch=bucket):
                        expected = native and bucket in (1, 2, 4, 8)
                        for name in ("pages", "oe", "metadata"):
                            self.assertEqual(layouts.welm_native_region_active(name), expected)
                        self.assertEqual(layouts.welm_decode_tiles_active(), not native)
                        self.assertTrue(layouts.welm_original_kernels_active())
            for key in ("MK_NATIVE_TILE_DEPENDENCIES", "MK_FIXED_NATIVE_FLASH_METADATA"):
                with patch.dict("os.environ", {key: "0"}), layouts.decode_tile_scope(native_ops_only=True, batch=2):
                    self.assertFalse(layouts.welm_native_region_active("metadata"))
                    self.assertEqual(layouts.welm_native_region_active("oe"), key != "MK_NATIVE_TILE_DEPENDENCIES")
            with patch.object(mk._fixed, "SUPPORTS_WELM_DECODE_FLASH_METADATA", False):
                with layouts.decode_tile_scope(native_ops_only=True, batch=2):
                    self.assertFalse(layouts.welm_native_region_active("metadata"))
            with patch.dict("os.environ", {"SGLANG_NPU_WELMV4_MK_NATIVE_REGIONS": "pages"}):
                with layouts.decode_tile_scope(native_ops_only=True, batch=4):
                    self.assertTrue(layouts.welm_native_region_active("pages"))
                    self.assertFalse(layouts.welm_native_region_active("oe"))
                self.assertFalse(layouts.welm_native_region_active("pages"))

    def test_stable_page_tables_preserve_padding_and_signed_swa_conversion(self):
        import numpy as np
        from test_welmv4_prefill_graph_contracts import load_method
        device = types.SimpleNamespace(type="npu")
        events = []

        class Array(np.ndarray):
            def __new__(cls, values, dtype):
                return np.array(values, dtype=dtype).view(cls)

            def is_contiguous(self):
                return self.flags.c_contiguous

            @property
            def device(self):
                return device

            def copy_(self, value):
                events.append(("copy", self))
                self[...] = value

        def floor(locs, divisor, out):
            events.append(("out", out))
            out[...] = locs // divisor

        torch = types.SimpleNamespace(
            where=lambda *xs: np.where(*xs).view(Array), int32=np.int32,
            ops=types.SimpleNamespace(aten=types.SimpleNamespace(
                floor_divide=types.SimpleNamespace(out=floor))),
        )
        fn = load_method(SRT / "hardware_backend/npu/attention/ascend_backend.py",
                         "AscendAttnBackend", "_prepare_welm_graph_page_tables_device",
                         {"torch": torch, "torch_npu": types.SimpleNamespace(get_npu_format=lambda t: 2)})
        length = Array([66, 0], np.int64)
        req = Array([1, 99999], np.int64)
        starts = Array([0, 64, 128], np.int64)
        table = Array(np.zeros((2, 97)), np.int32)
        table[1, 0], table[1, 64] = 128, -1
        dst, swa = (Array(np.ones((2, 3)), np.int32) for _ in range(2))
        kwargs = dict(page_size=64, req_to_token=table, is_hybrid_swa=True)
        for mapping_dtype, enabled, expected in ((np.int32, True, ["out", "out"]),
                                                  (np.int64, True, ["out", "copy"]),
                                                  (np.int32, False, ["copy", "copy"])):
            with self.subTest(mapping=mapping_dtype, active=enabled):
                events.clear()
                mapping = Array(np.arange(200) + 64, mapping_dtype)
                mapping[-1] = -65
                fn(types.SimpleNamespace(**kwargs, full_to_swa_index_mapping=mapping),
                   types.SimpleNamespace(block_tables=dst, block_tables_swa=swa),
                   req, length, length > 0, starts, starts < 97,
                   page_divisor=64 if enabled else None)
                self.assertEqual([e[0] for e in events], expected)
                self.assertIs(events[0][1], dst)
                self.assertIs(events[1][1], swa)
                self.assertEqual(dst.tolist(), [[2, -1, 0], [0, 0, 0]])
                self.assertEqual(swa.tolist(), [[3, -2, 0], [0, 0, 0]])

    def test_metadata_buffer_contract_rejects_aliases_layouts_and_dtypes(self):
        from test_welmv4_prefill_graph_contracts import load_nodes
        device = types.SimpleNamespace(type="npu")
        torch = types.SimpleNamespace(int32="i32", int64="i64")
        fn = load_nodes(DIR / "welm_flash_metadata.py", ["can_record_decode_metadata"], {
            "torch": torch, "torch_npu": types.SimpleNamespace(get_npu_format=lambda t: t.fmt),
        })["can_record_decode_metadata"]
        def fixed(size, start, dtype="i32"):
            return types.SimpleNamespace(numel=lambda: size, shape=(size,), dtype=dtype, device=device,
                                         is_contiguous=lambda: True, fmt=2, data_ptr=lambda: start,
                                         element_size=lambda: 4 if dtype == "i32" else 8)
        for batch in (1, 2, 4, 8):
            args = [fixed(batch, 100, "i64"), fixed(batch + 1, 512), fixed(batch, 768), fixed(batch, 1024)]
            self.assertTrue(fn(*args))
            for index, invalid in ((0, fixed(batch, 100, "bf16")), (2, fixed(batch, 513)),
                                   (3, fixed(batch, 769)), (1, fixed(batch, 512))):
                changed = list(args)
                changed[index] = invalid
                self.assertFalse(fn(*changed))
            args[1].fmt = 29
            self.assertFalse(fn(*args))

    def test_full_swa_metadata_uses_this_replays_lengths_before_pages(self):
        import numpy as np
        from test_welmv4_prefill_flash_metadata import (
            TORCH, Tensor, layer, make_backend, metadata, tensor,
        )
        from test_welmv4_prefill_graph_contracts import load_method, load_nodes

        path = SRT / "hardware_backend/npu/attention/ascend_backend.py"
        fake_torch = types.SimpleNamespace(**vars(TORCH))
        fake_torch.where = lambda condition, value, other: tensor(
            np.where(condition.data, value.data, other)
        )
        backend, calls, _ = make_backend()
        for name in ("init_forward_metadata_in_graph", "_prepare_welm_decode_metadata_device",
                     "_prime_welm_decode_flash_schedules"):
            fn = load_method(path, "AscendAttnBackend", name, {"torch": fake_torch})
            setattr(backend, name, types.MethodType(fn, backend))
        state = metadata(
            welm_decode_metadata_in_graph=True, swa_out_cache_loc=None,
            block_tables=tensor([[1], [2]]), block_tables_swa=tensor([[4], [5]]),
            welm_flash_cu_seqlens_q=tensor([0, 1, 2]), welm_flash_max_seqlen_q=1,
            welm_flash_seqused_q=tensor([-3, -3]), welm_flash_seqused_kv=tensor([-7, -7]),
        )
        backend.forward_metadata = state
        backend.use_welm_flash_attn = backend.graph_mode = True
        backend.graph_metadata = dict(welm_decode_page_starts=None, welm_decode_page_in_bounds=None)
        backend._prepare_welm_graph_page_tables_device = lambda *a, **kw: calls.append(("pages",))
        def schedule(*heads, **kw):
            record = ("schedule", heads, kw["mask_mode"], kw["win_left"],
                      tuple(kw["seqused_q"].data), tuple(kw["seqused_kv"].data))
            calls.append(record)
            return object()
        backend._welm_flash_attn_metadata = schedule
        wrapper = load_nodes(DIR / "welm_flash_metadata.py", ["_decode_flash_metadata"],
                             {"torch": fake_torch})["_decode_flash_metadata"]
        fake_torch.ops = types.SimpleNamespace(sglang=types.SimpleNamespace(
            welm_decode_flash_metadata=types.SimpleNamespace(default=wrapper)
        ))
        batch = types.SimpleNamespace(seq_lens=tensor([19, 23]), req_pool_indices=tensor([1, 2]))
        tile_module = types.SimpleNamespace(welm_native_region_active=lambda region: region == "metadata",
                                            welm_decode_tiles_active=lambda: False)
        q = Tensor(np.ones((2, 4, 4)))
        cache = backend.token_to_kv_pool.get_key_buffer(0)
        package = "sglang.srt.hardware_backend.npu.graph_runner"
        with patch.dict(sys.modules, {
            f"{package}.welm_tile_layouts": tile_module,
            f"{package}.welm_flash_metadata": types.SimpleNamespace(can_record_decode_metadata=lambda *a: True),
            "cann_ops_transformer": types.SimpleNamespace(flash_attn_metadata=schedule),
        }):
            def forward():
                calls.clear()
                backend.init_forward_metadata_in_graph(batch)
                for swa in (False, True, False):
                    backend._forward_welm_flash_attention(q, cache, cache, layer(swa), None)

            forward()
            self.assertEqual([x[0] for x in calls], ["pages", "schedule", "flash", "schedule", "flash", "flash"])
            last_schedules = tuple(state.welm_flash_schedules.values())
            for lengths in ((65, 0), (-1, 129), (31, 17)):
                batch.seq_lens.data[...] = lengths
                forward()
                self.assertEqual([x[0] for x in calls], ["schedule", "schedule", "pages", "flash", "flash", "flash"])
                self.assertEqual([(x[2], x[3]) for x in calls[:2]],
                                 [(3, -1), (4, layer(True).sliding_window_size)])
                for record in calls[:2]:
                    self.assertEqual(record[4:], (tuple(int(n > 0) for n in lengths),
                                                  tuple(max(n, 0) for n in lengths)))
                current = tuple(state.welm_flash_schedules.values())
                self.assertTrue(all(a is not b for a, b in zip(current, last_schedules)))
                self.assertIs(calls[3][2]["metadata"], current[0])
                self.assertIs(calls[4][2]["metadata"], current[1])
                self.assertIs(calls[5][2]["metadata"], current[0])
                last_schedules = current


if __name__ == "__main__":
    unittest.main()
