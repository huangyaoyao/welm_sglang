"""Exercise the ordinary WeLM FFN boundary selection without an NPU."""

import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch


MODEL = Path(__file__).resolve().parents[3] / 'python/sglang/srt/models/welmv4.py'


class BoundaryDispatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        block = next(n for n in ast.parse(MODEL.read_text()).body if getattr(n, 'name', '') == 'Qwen2MoeSparseMoeBlock')
        init = next(n for n in block.body if getattr(n, 'name', '') == '__init__')
        forward = next(n for n in block.body if getattr(n, 'name', '') == 'forward')
        attr = next(n for n in init.body if isinstance(n, ast.Assign) and
                    ast.unparse(n.targets[0]) == 'self.use_mk_ffn_layer_boundaries')
        select = next(n for n in forward.body if isinstance(n, ast.If) and
                      ast.unparse(n.test).startswith('self.use_mk_ffn_tiles and'))
        routed = next(n for n in ast.walk(forward) if isinstance(n, ast.If) and
                      ast.unparse(n.test) == 'tiled_ffn and (not tiled_ep)')
        original = next(n for n in ast.walk(forward) if isinstance(n, ast.If) and
                        ast.unparse(n.test) == 'experts_output is None')
        cls.dispatch = compile(ast.Module([attr, select, routed, original], []), str(MODEL), 'exec')

    def run_path(self, *, flag=True, coarse=False, tiles=True, decode=True, spec=None,
                 old=False, accepted=True, capture=True):
        events = []
        def mark():
            if capture:  # The MK test exercises the real context-local API.
                events.append('mark')
        def routed(*args):
            events.append('slots')
            return object() if accepted else None
        fallback = Mock(side_effect=lambda *_: events.append('original') or object())
        layer = NS(use_mk_ffn_tiles=True, tp_size=1, experts=fallback)
        mk = NS() if old else NS(mark_native_boundary=mark)
        scope = 'sglang.srt.hardware_backend.npu.graph_runner.'
        modules = {'megakernel':mk, scope+'welm_tile_layouts': NS(welm_decode_tiles_active=lambda:tiles),
                   scope+'welm_ffn_tiles':NS(try_routed_expert=routed)}
        state = dict(self=layer, get_bool_env_var=lambda _name, default: flag if flag is not None else default == "true", os=os, tiled_ffn=False,
                     ffn_boundary=None, experts_output=None, hidden_states=object(), topk_output=object(),
                     get_global_server_args=lambda:NS(speculative_algorithm=spec), tiled_ep=False,
                     moe_a2a_backend=NS(is_none=lambda: True), forward_batch=NS(forward_mode=NS(is_decode=lambda:decode)),
                     envs=NS(SGLANG_NPU_USE_MULTI_STREAM=NS(get=lambda:False)), torch=NS(compiler=NS(is_compiling=lambda:False)))
        with patch.dict('os.environ', MK_NATIVE_TILE_DEPENDENCIES='0' if coarse else '1'), patch.dict(sys.modules, modules):
            exec(self.dispatch, state)
        return events

    def test_boundaries_follow_completed_fine_ffn_only(self):
        self.assertEqual(self.run_path(), ['slots', 'mark'])
        self.assertEqual(self.run_path(flag=None), ['slots', 'mark'])
        self.assertEqual(self.run_path(accepted=False), ['slots', 'original'])
        self.assertEqual(self.run_path(old=True), ['original'])
        self.assertEqual(self.run_path(flag=False, old=True), ['slots'])
        self.assertEqual(self.run_path(coarse=True, old=True), ['slots'])
        self.assertEqual(self.run_path(spec='EAGLE'), ['slots'])
        self.assertEqual(self.run_path(capture=False), ['slots'])
        self.assertEqual(self.run_path(tiles=False), ['original'])
        self.assertEqual(self.run_path(decode=False), ['original'])


if __name__ == '__main__':
    unittest.main()
