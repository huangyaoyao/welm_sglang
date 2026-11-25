"""Cross-check the WeLM decode attention contracts against the source offsets."""
import ast
from dataclasses import dataclass
from math import prod
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt"
CONTRACT = ROOT / "hardware_backend/npu/graph_runner/welm_attention_tiles.py"


@dataclass(eq=False)
class Tensor:
    shape: tuple
    width: int = 2
    strides: tuple = ()

    def stride(self):
        return self.strides or tuple(prod(self.shape[i + 1:]) for i in range(len(self.shape)))

    def numel(self):
        return prod(self.shape)

    def element_size(self):
        return self.width


class WelmAttentionTilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layouts = runpy.run_path(str(CONTRACT))

    def test_specialized_norm_multiple_outputs_tail_and_residual(self):
        rows, cols = 9, 2048
        x = Tensor((rows, cols))
        r, out, aux = Tensor(x.shape, 4), Tensor(x.shape), Tensor(x.shape, 4)
        gamma = Tensor((cols,))
        for block in (1, 2, 4):
            for residual in (r, None):
                layout = self.layouts["_specialized_norm"](
                    x, residual, gamma, out, aux, rows, cols, 1e-5, cols, block
                )
                self.assertEqual((layout["count"], layout["block"]), (rows, block))
                self.assertEqual(layout["reads"], [(x, 4096)] + ([(r, 8192)] if residual else []))
                self.assertEqual(layout["writes"], [(out, 4096), (aux, 8192)])
                self.assertEqual(layout["shared_reads"], [gamma])
                # The source assigns floor(pid * blocks / grid) through the
                # following floor bound. MK admits grid == ceil(rows/block).
                grid = (rows + block - 1) // block
                by_pid = [list(range(pid * grid // grid * block,
                                     min(rows, (pid + 1) * grid // grid * block))) for pid in range(grid)]
                self.assertEqual(sum(by_pid, []), list(range(rows)))
        self.assertIsNone(self.layouts["_specialized_norm"](
            x, r, gamma, out, aux, rows, cols, 1e-5, cols * 2
        ))

    def test_generic_norm_strides_and_optional_outputs(self):
        rows, cols, heads = 16, 256, 2
        x = Tensor((rows // heads, heads, cols))
        out = Tensor(x.shape)
        gamma = Tensor((cols,))
        residual, residual_out, fp32 = (Tensor((rows, cols), 4) for _ in range(3))
        args = [x, residual, gamma, out, residual_out, fp32, rows, cols, 1e-6,
                cols, heads, heads * cols, cols, False, cols]
        layout = self.layouts["_generic_norm"](*args)
        self.assertEqual(layout["reads"], [(x, 512), (residual, 1024)])
        self.assertEqual(layout["writes"], [(out, 512), (residual_out, 1024), (fp32, 1024)])
        for index, value in ((9, cols * 2), (11, heads * cols + 1), (12, cols + 1)):
            other = args.copy()
            other[index] = value
            self.assertIsNone(self.layouts["_generic_norm"](*other))
        packed_q = Tensor(x.shape, strides=(heads * cols + 128, cols, 1))
        other = args.copy()
        other[0], other[11] = packed_q, packed_q.stride()[0]
        packed = self.layouts["_generic_norm"](*other)
        self.assertEqual(packed["shared_reads"], [gamma, packed_q])
        self.assertEqual(packed["reads"], [(residual, 1024)])
        self.assertEqual(packed["writes"], layout["writes"])
        args[1] = args[4] = args[5] = None
        self.assertEqual(self.layouts["_generic_norm"](*args)["writes"], [(out, 512)])

    def test_post_attention_optional_output_alias(self):
        rows, cols = 8, 2048
        x, r, y, resid_out, fp32 = [Tensor((rows, cols), size) for size in (2, 4, 2, 4, 4)]
        g1, g2 = Tensor((cols,)), Tensor((cols,))
        post = self.layouts["_post_attention_norm"]
        params = dict(rows=rows, cols=cols, eps=1e-5, BLOCK_SIZE=cols)
        enabled = post(x, r, g1, g2, y, resid_out, fp32, **params, WRITE_FP32_OUT=True)
        self.assertEqual(enabled["writes"], [(y, 4096), (resid_out, 8192), (fp32, 8192)])
        disabled = post(x, r, g1, g2, y, resid_out, resid_out, **params, WRITE_FP32_OUT=False)
        self.assertEqual(disabled["writes"], [(y, 4096), (resid_out, 8192)])
        self.assertIsNone(post(x, r, g1, g2, y, resid_out, fp32, **params, WRITE_FP32_OUT=False))

    def test_sigmoid_gate_uses_head_rows_and_updates_only_attention(self):
        rows, cols = 8 * 24, 256
        gates, values = Tensor((rows, 1), 4), Tensor((rows, cols))
        sigmoid = self.layouts["_sigmoid_gate"]
        layout = sigmoid(gates, values, rows, cols, 1, cols, 1, BLOCK_SIZE=128)
        self.assertEqual((layout["count"], layout["block"]), (192, 128))
        self.assertEqual(layout["reads"], [(gates, 4), (values, 512)])
        self.assertEqual(layout["writes"], [(values, 512)])
        self.assertIsNone(sigmoid(gates, values, rows, cols, 2, cols, 1, BLOCK_SIZE=128))

    def test_large_verify_selects_token_blocks_with_masked_tail(self):
        rows, block = 641, 64
        q, k = Tensor((rows, 24, 256)), Tensor((rows, 2, 256))
        pos, table = Tensor((rows,), 4), Tensor((16384, 64), 4)
        args = (q, k, pos, table, 11, rows, 24 * 256, 2 * 256, 256, 64, block)
        callback = self.layouts["_blocked_rope"]
        layout = callback(*args, True, 1, 24, 2, multibuffer=True)
        self.assertEqual(layout["reads"], [(q, 12288), (k, 1024), (pos, 4)])
        self.assertEqual(layout["writes"], [(q, 12288), (k, 1024)])
        self.assertEqual(layout["shared_reads"], [table])
        self.assertEqual(layout["block"], 64)
        self.assertIsNone(callback(*args, False, 1, 24, 2, multibuffer=True))

    def test_registered_callbacks_bind_real_kernel_arguments(self):
        from inspect import signature
        for module, bindings in self.layouts["KERNEL_LAYOUTS"].items():
            tree = ast.parse((ROOT / "layers" / (module + ".py")).read_text())
            source = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
            for name, callback in bindings:
                args = source[name].args.args
                # Source launches pass BLOCK_SIZE by keyword on the sigmoid path.
                if name == "sigmoid_mul_kernel_npu":
                    signature(callback).bind(*[None] * (len(args) - 1), BLOCK_SIZE=128)
                else:
                    signature(callback).bind(*[None] * len(args))


class WelmAttentionDeviceTest(unittest.TestCase):
    def test_paged_decode_mixed_replays_page_table_and_lengths(self):
        try:
            import torch
            import torch_npu
        except ImportError as exc:
            self.skipTest(f"requires torch and torch_npu: {exc}")
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires an Ascend A5 NPU")
        from megakernel._fixed import capture_fixed
        from megakernel.triton import prepare_bindings
        from sglang.srt.hardware_backend.npu.attention.sink_full_attention import (
            paged_attention_decode_impl,
        )
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import (
            decode_tile_scope,
            install_welm_tile_layouts,
        )

        with torch.inference_mode(), tempfile.TemporaryDirectory() as directory:
            q = torch.randn(2, 24, 256, dtype=torch.bfloat16, device="npu")
            k, v = [
                torch.randn(16, 64, 2, 256, dtype=q.dtype, device=q.device).transpose(1, 2)
                for _ in range(2)
            ]
            sinks = torch.randn(24, dtype=torch.float32, device=q.device)
            lengths = torch.tensor([7, 1], dtype=torch.int32, device=q.device)
            pages = torch.arange(16, dtype=torch.int32, device=q.device).view(2, 8)

            def forward():
                return paged_attention_decode_impl(
                    q, k, v, lengths, pages, False, sinks=sinks, max_kv_len_hint=512
                )

            install_welm_tile_layouts()
            with decode_tile_scope():
                with prepare_bindings():
                    forward()
                graph = capture_fixed(forward, Path(directory))
            try:
                self.assertEqual(graph.host_boundary_count, 0)
                self.assertEqual(graph.fused_operator_count, 1)
                self.assertTrue(any(t and t.get("core") == 3 for t in graph.tasks))
                for seq in ([64, 65], [511, 512]):
                    lengths.copy_(torch.tensor(seq, dtype=lengths.dtype, device=q.device))
                    pages.copy_(pages.roll(1, dims=1))
                    q.normal_()
                    expected = forward().cpu()
                    graph.outputs.fill_(float("nan"))
                    torch.testing.assert_close(graph.replay().cpu(), expected, rtol=0, atol=0)
            finally:
                graph.reset()

    def test_post_attention_and_fused_norm_replay(self):
        try:
            import torch
            import torch_npu  # Registers the NPU device.
        except ImportError as exc:
            self.skipTest(f"A5 requires torch and torch_npu: {exc}")
        if not torch.npu.is_available() or not torch.npu.get_device_name().startswith("Ascend950"):
            self.skipTest("requires an Ascend A5 NPU")

        from megakernel._fixed import capture_fixed
        from sglang.srt.hardware_backend.npu.graph_runner.welm_tile_layouts import (
            decode_tile_scope,
            install_welm_tile_layouts,
        )
        from sglang.srt.layers.welmv4_npu_op import welmv4_fused_rms_norm_false_false_npu
        from sglang.srt.layers.welmv4_op import mmq_style_norm_after_attn

        rows, cols = 8, 2048
        generator = torch.Generator().manual_seed(456)
        gamma = torch.randn(cols, generator=generator).to(dtype=torch.bfloat16, device="npu")

        def forward(x, residual):
            activations, updated, _ = mmq_style_norm_after_attn(
                x, residual, gamma, gamma, 1e-5, return_fp32_out=False
            )
            return welmv4_fused_rms_norm_false_false_npu(
                activations, updated, gamma, 1e-5, torch.bfloat16, rows
            )

        with torch.inference_mode(), tempfile.TemporaryDirectory() as directory:
            examples = []
            for _ in range(3):
                x = torch.randn(rows, cols, generator=generator).to(torch.bfloat16)
                r = torch.randn(rows, cols, generator=generator)
                reference = tuple(t.cpu() for t in forward(x.to("npu"), r.to("npu")))
                examples.append((x, r, reference))
            x, r = (t.to("npu") for t in examples[0][:2])
            install_welm_tile_layouts()
            with decode_tile_scope():
                plan = capture_fixed(lambda: forward(x, r), directory)
            try:
                self.assertEqual(plan.scope_count, 1, plan.execution_summary)
                self.assertEqual(plan.plans[0].tile_dependencies, [(0, 1)])
                self.assertEqual(plan.host_boundary_count, 0)
                for new_x, new_r, reference in examples:
                    x.copy_(new_x)
                    r.copy_(new_r)
                    for actual, expected in zip(plan.replay(), reference):
                        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            finally:
                plan.reset()


if __name__ == "__main__":
    unittest.main()
