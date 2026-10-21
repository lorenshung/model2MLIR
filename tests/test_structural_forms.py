"""Pools, pool backwards, col2im, pads, repeat and select_scatter lower to structural forms.

No gather loop (``scf.for`` / scalar ``tensor.extract`` / ``linalg.index``) remains:
average pools are windowed sum reductions, backwards and col2im accumulate shifted
windows, pads/repeat/select_scatter are slices, concats, broadcasts and inserts.
Native execution must equal eager PyTorch bit for bit (the CPU summation order,
divisors and division sequence are kept).
"""

from __future__ import annotations

import hashlib

import pytest
import torch

import m2m
from tests.native_execution import _Call, assert_native_matches_eager, run_native, seeded_randn as _r

A = torch.ops.aten
GATHER = ("scf.for", "scf.while", "tensor.extract ", "linalg.index")


CASES = [
    # (id, fn, inputs, marker)
    ("avg_pool1d_k3s2p1_ceil_nocip", lambda x: A.avg_pool1d.default(x, [3], [2], [1], True, False),
     (_r(1, 3, 10),), "window_sum"),
    ("avg_pool1d_k2", lambda x: A.avg_pool1d.default(x, [2]), (_r(1, 3, 10),), "window_sum"),
    ("avg_pool2d_k3s2p1", lambda x: A.avg_pool2d.default(x, [3, 3], [2, 2], [1, 1]), (_r(1, 3, 7, 7),),
     "window_sum"),
    ("avg_pool2d_ceil_cip", lambda x: A.avg_pool2d.default(x, [3, 3], [2, 2], [1, 1], True, True),
     (_r(1, 3, 7, 7),), "window_sum"),
    ("avg_pool2d_ceil_nocip", lambda x: A.avg_pool2d.default(x, [3, 3], [2, 2], [1, 1], True, False),
     (_r(1, 3, 7, 7),), "window_sum"),
    ("avg_pool2d_divisor5", lambda x: A.avg_pool2d.default(x, [3, 3], [2, 2], [1, 1], False, True, 5),
     (_r(1, 3, 7, 7),), "window_sum"),
    ("avg_pool2d_rect", lambda x: A.avg_pool2d.default(x, [2, 3], [1, 2]), (_r(2, 3, 7, 7),), "window_sum"),
    ("avg_pool2d_unbatched", lambda x: A.avg_pool2d.default(x, [2, 2]), (_r(3, 6, 6),), "window_sum"),
    ("avg_pool2d_f16", lambda x: A.avg_pool2d.default(x, [3, 3], [1, 1], [1, 1]),
     (_r(1, 2, 6, 6, dtype=torch.float16),), "window_sum"),
    ("avg_pool2d_bf16", lambda x: A.avg_pool2d.default(x, [2, 2]), (_r(1, 2, 6, 6, dtype=torch.bfloat16),),
     "window_sum"),
    ("avg_pool3d_ceil_nocip", lambda x: A.avg_pool3d.default(x, [2, 2, 2], [2, 2, 2], [1, 1, 1], True, False),
     (_r(1, 2, 5, 5, 5),), "window_sum"),
    ("avg_pool3d_divisor3", lambda x: A.avg_pool3d.default(x, [2, 2, 2], [2, 2, 2], [0, 0, 0], False, True, 3),
     (_r(1, 2, 4, 4, 4),), "window_sum"),
    ("adaptive_avg_pool1d_10_4", lambda x: A.adaptive_avg_pool1d.default(x, [4]), (_r(1, 3, 10),), "window_sum"),
    ("adaptive_avg_pool2d_7_3", lambda x: A._adaptive_avg_pool2d.default(x, [3, 3]), (_r(1, 3, 7, 7),),
     "window_sum"),
    ("adaptive_avg_pool2d_global", lambda x: A._adaptive_avg_pool2d.default(x, [1, 1]), (_r(2, 3, 7, 7),),
     "window_sum"),
    ("adaptive_avg_pool2d_runs", lambda x: A._adaptive_avg_pool2d.default(x, [4, 3]), (_r(1, 2, 10, 7),),
     "window_sum"),
    ("adaptive_avg_pool2d_bf16", lambda x: A._adaptive_avg_pool2d.default(x, [3, 3]),
     (_r(1, 2, 7, 7, dtype=torch.bfloat16),), "window_sum"),
    ("adaptive_avg_pool3d", lambda x: A._adaptive_avg_pool3d.default(x, [2, 3, 4]), (_r(1, 2, 5, 6, 7),),
     "window_sum"),
    ("adaptive_avg_pool3d_f16", lambda x: A._adaptive_avg_pool3d.default(x, [2, 2, 2]),
     (_r(1, 2, 5, 4, 3, dtype=torch.float16),), "window_sum"),
    ("avg_pool2d_backward_ceil_nocip", lambda g, x: A.avg_pool2d_backward.default(
        g, x, [3, 3], [2, 2], [1, 1], True, False, None), (_r(1, 3, 4, 4), _r(1, 3, 7, 7)), "tap_accumulate"),
    ("avg_pool2d_backward_k2", lambda g, x: A.avg_pool2d_backward.default(
        g, x, [2, 2], [2, 2], [0, 0], False, True, None), (_r(1, 3, 4, 4), _r(1, 3, 8, 8)), "tap_accumulate"),
    ("avg_pool2d_backward_divisor4", lambda g, x: A.avg_pool2d_backward.default(
        g, x, [3, 3], [2, 2], [1, 1], False, True, 4), (_r(1, 3, 4, 4), _r(1, 3, 7, 7)), "tap_accumulate"),
    ("avg_pool2d_backward_empty_stride", lambda g, x: A.avg_pool2d_backward.default(
        g, x, [2, 2], [], [0, 0], False, True, None), (_r(1, 3, 4, 4), _r(1, 3, 8, 8)), "tap_accumulate"),
    ("adaptive_avg_pool2d_backward_7_3", lambda g, x: A._adaptive_avg_pool2d_backward.default(g, x),
     (_r(1, 3, 3, 3), _r(1, 3, 7, 7)), "tap_accumulate"),
    ("adaptive_avg_pool2d_backward_8_4", lambda g, x: A._adaptive_avg_pool2d_backward.default(g, x),
     (_r(2, 3, 4, 4), _r(2, 3, 8, 8)), "tap_accumulate"),
    ("col2im_k2", lambda x: A.col2im.default(x, [4, 5], [2, 2], [1, 1], [0, 0], [1, 1]), (_r(1, 12, 12),),
     "tap_accumulate"),
    ("col2im_dilated_padded", lambda x: A.col2im.default(x, [5, 6], [2, 3], [1, 2], [1, 1], [2, 1]),
     (_r(1, 12, 12),), "tap_accumulate"),
    ("col2im_unbatched", lambda x: A.col2im.default(x, [3, 3], [2, 2], [1, 1], [0, 0], [1, 1]), (_r(8, 4),),
     "tap_accumulate"),
    ("reflection_pad1d", lambda x: A.reflection_pad1d.default(x, [2, 1]), (_r(1, 2, 5),), "mirror_copy"),
    ("reflection_pad2d", lambda x: A.reflection_pad2d.default(x, [1, 2, 2, 1]), (_r(1, 2, 4, 5),),
     "mirror_copy"),
    ("reflection_pad3d", lambda x: A.reflection_pad3d.default(x, [1, 1, 1, 0, 0, 1]), (_r(1, 2, 3, 4, 3),),
     "mirror_copy"),
    ("replication_pad2d", lambda x: A.replication_pad2d.default(x, [1, 2, 2, 1]), (_r(1, 2, 4, 5),), "edge_copy"),
    ("replication_pad3d", lambda x: A.replication_pad3d.default(x, [1, 1, 1, 0, 0, 1]), (_r(1, 2, 3, 4, 3),),
     "edge_copy"),
    ("repeat_leading", lambda x: A.repeat.default(x, [2, 1, 3]), (_r(2, 3),), "tile"),
    ("repeat_cols", lambda x: A.repeat.default(x, [1, 2]), (_r(2, 3),), "tile"),
    ("repeat_int", lambda x: A.repeat.default(x, [3, 2]), (torch.arange(6, dtype=torch.int32).reshape(2, 3),),
     "tile"),
    ("select_scatter_dim1_neg", lambda x, s: A.select_scatter.default(x, s, 1, -1), (_r(2, 3, 4), _r(2, 4)),
     "scatter"),
    ("select_scatter_dim0", lambda x, s: A.select_scatter.default(x, s, 0, 1), (_r(3, 4), _r(4)), "scatter"),
    ("select_scatter_1d", lambda x, s: A.select_scatter.default(x, s, 0, 2), (_r(5), torch.tensor(7.0)),
     "scatter"),
]


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_structural_form(case):
    _, fn, inputs, marker = case
    result = m2m.convert(_Call(fn).eval(), inputs, backend="fx_importer")
    assert result.ok and "func.call" not in result.mlir_text
    assert not [token for token in GATHER if token in result.mlir_text], result.mlir_text
    assert f'prov.role = "{marker}"' in result.mlir_text
    if marker == "window_sum":
        windows = [line for line in result.mlir_text.splitlines() if 'prov.role = "window_sum"' in line
                   and "linalg.generic" in line]
        assert windows and all('"reduction"' in line and "prov.window" in line for line in windows)
        assert 'prov.family = "reduce"' in result.mlir_text


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_structural_form_matches_eager_exactly(tmp_path, case):
    _, fn, inputs, _ = case
    assert_native_matches_eager(tmp_path, fn, inputs, rtol=0, atol=0)


def test_nonfinite_values_propagate_through_windows(tmp_path):
    x = _r(1, 1, 6, 6)
    x[0, 0, 0, 1] = float("nan")
    x[0, 0, 3, 3] = float("inf")
    x[0, 0, 5, 0] = float("-inf")
    for i, fn in enumerate((lambda v: A.avg_pool2d.default(v, [3, 3], [2, 2], [1, 1], True, False),
                            lambda v: A._adaptive_avg_pool2d.default(v, [4, 4]),
                            lambda v: A.reflection_pad2d.default(v, [2, 2, 1, 1]))):
        # A distinct directory per program: ctypes caches a library by path.
        (tmp_path / str(i)).mkdir()
        assert_native_matches_eager(tmp_path / str(i), fn, (x,), rtol=0, atol=0)


@pytest.mark.parametrize("sizes", [((4, 4), (10, 10)), ((3, 4), (7, 10)), ((4, 3), (10, 5))])
def test_multi_run_adaptive_backward_keeps_cpu_accumulation_order(tmp_path, sizes):
    # Several window runs per axis: runs and taps are interleaved per axis so each input
    # element still receives its contributions in lexicographic output order.
    out, size = sizes
    fn = lambda g, x: A._adaptive_avg_pool2d_backward.default(g, x)  # noqa: E731
    inputs = (_r(1, 2, *out), _r(1, 2, *size))
    result = m2m.convert(_Call(fn).eval(), inputs, backend="fx_importer")
    assert result.ok and not [token for token in GATHER if token in result.mlir_text]
    assert_native_matches_eager(tmp_path, fn, inputs, rtol=0, atol=0)


def test_gather_lowering_flag_restores_previous_kernels(monkeypatch):
    monkeypatch.setenv("M2M_WINDOW_LOWERING", "gather")
    fn = lambda x: A.avg_pool2d.default(x, [2, 2])  # noqa: E731
    result = m2m.convert(_Call(fn).eval(), (_r(1, 1, 4, 4),), backend="fx_importer")
    assert result.ok and "scf.for" in result.mlir_text and 'prov.role = "window_sum"' not in result.mlir_text


# sha256 of the output bytes of the removed gather kernel (941eac9, M2M_WINDOW_LOWERING=gather), which the
# structural kernel matched byte for byte: the CPU column/col2im accumulation order is kept.
_CONV_INPUT_GRADIENT_SHA256 = {
    (0, torch.float32): "49c9c8b4fe8038888dab53abf8638042ac98679f304ee9ee0055c9652f9162b0",
    (0, torch.float16): "37ae353b4046943c7d68d0d00a189ac2bc75b5de060b5a3b05edc2e70959d0ad",
    (1, torch.float32): "44916fadfab78ab66352645597092f8ff3d9dca65245f7e3fd8e453398aa24ca",
    (1, torch.float16): "7e963363e511e8547f19a65c7ceed05db50b3b30359d364bd93bc02a31d13e44",
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("index,geometry", enumerate([([1, 1], [1, 1], [1, 1], 1), ([2, 1], [0, 1], [2, 1], 2)]))
def test_convolution_backward_input_gradient_is_structural_and_unchanged(tmp_path, index, geometry, dtype):
    stride, padding, dilation, groups = geometry

    def fn(go, x, w):
        return A.convolution_backward.default(go, x, w, None, stride, padding, dilation, False, [0, 0], groups,
                                              [True, False, False])[0]

    x, w = _r(2, 4, 7, 6, dtype=dtype), _r(6, 4 // groups, 3, 3, dtype=dtype)
    go = _r(*A.convolution.default(x.float(), w.float(), None, stride, padding, dilation, False, [0, 0],
                                   groups).shape, dtype=dtype)
    result = m2m.convert(_Call(fn).eval(), (go, x, w), backend="fx_importer")
    assert result.ok and not [token for token in GATHER if token in result.mlir_text]
    assert 'prov.role = "tap_accumulate"' in result.mlir_text
    contraction = [line for line in result.mlir_text.splitlines()
                   if "linalg.generic" in line and 'prov.role = "contraction"' in line]
    assert len(contraction) == 1 and contraction[0].count('"reduction"') == 1
    structural = run_native(tmp_path, fn, (go, x, w))[0]
    assert hashlib.sha256(structural.tobytes()).hexdigest() == _CONV_INPUT_GRADIENT_SHA256[index, dtype]
    torch.testing.assert_close(torch.from_numpy(structural), fn(go, x, w), rtol=1e-3, atol=1e-3)


def test_cat_skips_legacy_empty_operand(tmp_path):
    fn = lambda a, b: A.cat.default([a, b], 0)  # noqa: E731
    inputs = (_r(2, 3), torch.empty(0))
    assert_native_matches_eager(tmp_path, fn, inputs, rtol=0, atol=0)
