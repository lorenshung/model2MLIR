"""Convolution as activation rows x weight (weight on the RHS), 3-D and grouped-transposed conv,
and transposed / 1-D convolution_backward.

The rows form must be bit-identical to the previous weight x columns form (same K
order and reduction order) and match eager PyTorch; previously opaque geometries
must now lower and match eager.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import torch

import m2m
from tests.native_execution import _Call, assert_native_matches_eager, run_native

A = torch.ops.aten


def _r(*shape, seed=0, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed + sum(shape))
    return torch.randn(*shape, generator=generator).to(dtype)


def conv(stride, padding, dilation, groups, transposed=False, output_padding=None):
    def fn(x, w, *bias):
        rank = x.ndim - 2
        return A.convolution.default(x, w, bias[0] if bias else None, stride, padding, dilation, transposed,
                                     output_padding or [0] * rank, groups)
    return fn


FORWARD = {
    "2d_bias": (conv([1, 1], [1, 1], [1, 1], 1), (_r(2, 3, 7, 7), _r(4, 3, 3, 3), _r(4))),
    "2d_s2_d2": (conv([2, 2], [1, 1], [2, 2], 1), (_r(1, 3, 9, 9), _r(4, 3, 3, 3))),
    "grouped": (conv([1, 1], [0, 0], [1, 1], 2), (_r(1, 4, 6, 6), _r(6, 2, 3, 3), _r(6))),
    "depthwise": (conv([1, 1], [1, 1], [1, 1], 4), (_r(2, 4, 6, 6), _r(4, 1, 3, 3))),
    "1d": (conv([2], [1], [1], 1), (_r(1, 3, 10), _r(4, 3, 3), _r(4))),
    "patch_embed": (conv([4, 4], [0, 0], [1, 1], 1), (_r(1, 3, 8, 8), _r(8, 3, 4, 4))),
    "3d": (conv([1, 1, 1], [1, 1, 1], [1, 1, 1], 1), (_r(1, 2, 4, 4, 4), _r(3, 2, 2, 2, 2))),
    "3d_grouped_strided": (conv([2, 1, 1], [0, 1, 0], [1, 1, 2], 2), (_r(2, 4, 5, 4, 6), _r(4, 2, 2, 3, 2), _r(4))),
    "transposed": (conv([2, 2], [1, 1], [1, 1], 1, True, [1, 1]), (_r(1, 4, 5, 5), _r(4, 3, 3, 3), _r(3))),
    "transposed_grouped_dilated": (conv([2, 2], [0, 0], [2, 2], 2, True), (_r(1, 4, 4, 4), _r(4, 3, 3, 3))),
    "transposed_grouped_1d": (conv([2], [1], [1], 2, True, [1]), (_r(2, 4, 6), _r(4, 2, 3), _r(4))),
}


@pytest.mark.parametrize("name", sorted(FORWARD))
def test_convolution_matches_eager(tmp_path, name):
    fn, inputs = FORWARD[name]
    assert_native_matches_eager(tmp_path, fn, inputs, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", ["2d_bias", "grouped", "1d", "3d", "patch_embed"])
def test_weight_is_the_right_hand_operand(name):
    fn, inputs = FORWARD[name]
    text = m2m.convert(_Call(fn).eval(), inputs, backend="fx_importer").mlir_text
    assert "func.call" not in text and 'prov.conv_form = "rows_x_weight"' in text
    assert 'prov.conv_geometry = "stride=' in text
    contraction = [line for line in text.splitlines() if 'prov.role = "contraction"' in line
                   and ("linalg.matmul" in line or "linalg.generic" in line)]
    assert len(contraction) == 1
    line = contraction[0]
    weight = inputs[1]
    k = weight[0].numel()  # (C/G) * prod(kernel)
    if "linalg.matmul" in line:
        f = weight.shape[0]
        assert re.search(rf"tensor<\d+x{k}xf32>, tensor<{k}x{f}xf32>", line), line
    else:
        assert line.count('"reduction"') == 1


@pytest.mark.parametrize("name", ["2d_bias", "2d_s2_d2", "grouped", "depthwise", "1d", "patch_embed", "transposed"])
def test_rows_form_is_bit_identical_to_columns_form(tmp_path, monkeypatch, name):
    fn, inputs = FORWARD[name]
    (tmp_path / "rows").mkdir()
    rows = run_native(tmp_path / "rows", fn, inputs)
    monkeypatch.setenv("M2M_CONV_LOWERING", "im2col_lhs")
    (tmp_path / "columns").mkdir()
    columns = run_native(tmp_path / "columns", fn, inputs)
    assert all(np.array_equal(a, b) for a, b in zip(rows, columns))


def backward(stride, padding, dilation, groups, mask, transposed=False, output_padding=None, bias=True):
    def fn(grad, x, w):
        rank = x.ndim - 2
        sizes = [w.shape[1] * groups if transposed else w.shape[0]] if bias else None
        out = A.convolution_backward.default(grad, x, w, sizes, stride, padding, dilation, transposed,
                                             output_padding or [0] * rank, groups, mask)
        # Select the requested slots by index: eager 1-D convolution_backward returns a
        # tensor even in an unrequested slot, where the exported program has None.
        return tuple(out[i] for i, requested in enumerate(mask) if requested)
    return fn


BACKWARD = {
    "transposed_TTT": (backward([2, 2], [1, 1], [1, 1], 1, [True, True, True], True, [1, 1]),
                       (_r(1, 3, 10, 10), _r(1, 4, 5, 5), _r(4, 3, 3, 3))),
    "transposed_grouped_TTF": (backward([2, 2], [0, 0], [1, 1], 2, [True, True, False], True, bias=False),
                               (_r(2, 4, 9, 9), _r(2, 4, 4, 4), _r(4, 2, 3, 3))),
    "transposed_dilated_FTT": (backward([1, 1], [1, 1], [2, 2], 1, [False, True, True], True),
                               (_r(1, 2, 8, 8), _r(1, 3, 6, 6), _r(3, 2, 3, 3))),
    "1d_TTT": (backward([1], [0], [1], 1, [True, True, True]), (_r(1, 4, 8), _r(1, 3, 10), _r(4, 3, 3))),
    "1d_strided_grouped_TFT": (backward([2], [1], [2], 2, [True, False, True]),
                               (_r(2, 4, 4), _r(2, 4, 9), _r(4, 2, 3))),
    "1d_transposed_TTT": (backward([2], [1], [1], 1, [True, True, True], True, [1]),
                          (_r(1, 3, 10), _r(1, 4, 5), _r(4, 3, 3))),
}


@pytest.mark.parametrize("name", sorted(BACKWARD))
def test_convolution_backward_variants_match_eager(tmp_path, name):
    fn, inputs = BACKWARD[name]
    result = m2m.convert(_Call(fn).eval(), inputs, backend="fx_importer")
    assert result.ok and "func.call" not in result.mlir_text
    assert_native_matches_eager(tmp_path, fn, inputs, rtol=1e-4, atol=1e-4)
