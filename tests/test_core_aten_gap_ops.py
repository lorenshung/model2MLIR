"""Focused coverage for Core ATen operators found by the full-op corpus."""

from __future__ import annotations

import pytest
import torch

from m2m.coverage import validate_op
from tests.native_execution import assert_native_matches_eager


_UNARY = {
    "acos": lambda x: torch.ops.aten.acos.default(x),
    "acosh": lambda x: torch.ops.aten.acosh.default(x.abs() + 1.25),
    "asin": lambda x: torch.ops.aten.asin.default(x.tanh()),
    "asinh": lambda x: torch.ops.aten.asinh.default(x),
    "atan": lambda x: torch.ops.aten.atan.default(x),
    "atanh": lambda x: torch.ops.aten.atanh.default(x.tanh() * 0.5),
    "cosh": lambda x: torch.ops.aten.cosh.default(x),
    "erf": lambda x: torch.ops.aten.erf.default(x),
    "expm1": lambda x: torch.ops.aten.expm1.default(x),
    "log10": lambda x: torch.ops.aten.log10.default(x.abs() + 0.5),
    "log1p": lambda x: torch.ops.aten.log1p.default(x.abs()),
    "log2": lambda x: torch.ops.aten.log2.default(x.abs() + 0.5),
    "sinh": lambda x: torch.ops.aten.sinh.default(x),
    "tan": lambda x: torch.ops.aten.tan.default(x),
    "trunc": lambda x: torch.ops.aten.trunc.default(x),
}


@pytest.mark.parametrize("name,fn", _UNARY.items(), ids=_UNARY.keys())
def test_core_aten_unary_math_lowers_without_opaque_calls(name, fn):
    verdict = validate_op(fn, (torch.linspace(-0.75, 0.75, 6).reshape(2, 3),), name=name)
    assert verdict.error is None, verdict.error
    assert verdict.lowered, verdict.opaque_calls
    assert verdict.shape_ok


def test_core_aten_atan2_lowers_without_opaque_calls():
    fn = torch.ops.aten.atan2.default
    inputs = (torch.randn(2, 3), torch.randn(2, 3))
    verdict = validate_op(fn, inputs, name="atan2")
    assert verdict.error is None, verdict.error
    assert verdict.lowered, verdict.opaque_calls
    assert verdict.shape_ok


@pytest.mark.parametrize(
    "name,fn,inputs",
    [
        (
            "bitwise_and_scalar",
            lambda x: torch.ops.aten.bitwise_and.Scalar(x, 3),
            (torch.arange(6, dtype=torch.int64).reshape(2, 3),),
        ),
        (
            "bitwise_or_tensor",
            torch.ops.aten.bitwise_or.Tensor,
            (
                torch.arange(6, dtype=torch.int64).reshape(2, 3),
                torch.full((2, 3), 2, dtype=torch.int64),
            ),
        ),
        (
            "bitwise_xor_scalar",
            lambda x: torch.ops.aten.bitwise_xor.Scalar(x, 3),
            (torch.arange(6, dtype=torch.int64).reshape(2, 3),),
        ),
        (
            "logical_or",
            torch.ops.aten.logical_or.default,
            (torch.tensor([[True, False]]), torch.tensor([[False, False]])),
        ),
        (
            "logical_xor",
            torch.ops.aten.logical_xor.default,
            (torch.tensor([[True, False]]), torch.tensor([[False, True]])),
        ),
        (
            "logical_or_numeric_truthiness",
            torch.ops.aten.logical_or.default,
            (
                torch.tensor([[0.5, 0.0, float("nan")]]),
                torch.tensor([[0.0, -2.0, 0.0]]),
            ),
        ),
    ],
)
def test_core_aten_integer_binary_ops_lower_without_opaque_calls(name, fn, inputs):
    verdict = validate_op(fn, inputs, name=name)
    assert verdict.error is None, verdict.error
    assert verdict.lowered, verdict.opaque_calls
    assert verdict.shape_ok


def test_core_aten_clamp_tensor_uses_tensor_bounds():
    fn = torch.ops.aten.clamp.Tensor
    inputs = (
        torch.linspace(-2, 2, 6).reshape(2, 3),
        torch.full((2, 3), -0.5),
        torch.full((2, 3), 0.75),
    )
    verdict = validate_op(fn, inputs, name="clamp_tensor")
    assert verdict.error is None, verdict.error
    assert verdict.lowered, verdict.opaque_calls
    assert verdict.shape_ok


def test_core_aten_clamp_tensor_supports_integer_bounds():
    fn = torch.ops.aten.clamp.Tensor
    inputs = (
        torch.arange(6, dtype=torch.int64).reshape(2, 3),
        torch.full((2, 3), 1, dtype=torch.int64),
        torch.full((2, 3), 4, dtype=torch.int64),
    )
    verdict = validate_op(fn, inputs, name="clamp_tensor_i64")
    assert verdict.error is None, verdict.error
    assert verdict.lowered, verdict.opaque_calls
    assert verdict.shape_ok


@pytest.mark.parametrize(
    "name,fn,inputs",
    [
        (
            "log_softmax",
            lambda x: torch.ops.aten._log_softmax.default(x, 1, False),
            (torch.randn(2, 3),),
        ),
        (
            "hardtanh",
            lambda x: torch.ops.aten.hardtanh.default(x, -0.5, 0.75),
            (torch.randn(2, 3),),
        ),
        (
            "index_select",
            lambda x, index: torch.ops.aten.index_select.default(x, 1, index),
            (torch.randn(2, 3), torch.tensor([2, 0], dtype=torch.int64)),
        ),
        (
            "isinf",
            torch.ops.aten.isinf.default,
            (torch.tensor([[0.0, float("inf"), float("-inf")]]),),
        ),
        (
            "leaky_relu",
            lambda x: torch.ops.aten.leaky_relu.default(x, 0.2),
            (torch.randn(2, 3),),
        ),
    ],
)
def test_selected_upstream_decompositions_close_over_supported_ops(name, fn, inputs):
    verdict = validate_op(fn, inputs, name=name)
    assert verdict.error is None, verdict.error
    assert verdict.lowered, verdict.opaque_calls
    assert verdict.shape_ok


@pytest.mark.parametrize("fill", [0.5, -1.25])
def test_constant_pad_nd_accepts_nonzero_static_fill(fill):
    fn = lambda x: torch.ops.aten.constant_pad_nd.default(x, [1, 2, 1, 0], fill)
    verdict = validate_op(fn, (torch.randn(2, 3),), name=f"constant_pad_{fill}")
    assert verdict.error is None, verdict.error
    assert verdict.lowered, verdict.opaque_calls
    assert verdict.shape_ok


@pytest.mark.parametrize(
    "name,fn,inputs",
    [
        (
            "isnan",
            torch.ops.aten.isnan.default,
            (torch.tensor([[float("nan"), 0.0, float("inf")]]),),
        ),
        (
            "sign_float",
            torch.ops.aten.sign.default,
            (torch.tensor([[-2.0, -0.0, 0.0, 3.0, float("nan")]]),),
        ),
        (
            "sign_int",
            torch.ops.aten.sign.default,
            (torch.tensor([[-2, 0, 3]], dtype=torch.int64),),
        ),
        (
            "masked_scatter",
            torch.ops.aten.masked_scatter.default,
            (
                torch.zeros(2, 3),
                torch.tensor([[False, True, False], [True, False, True]]),
                torch.arange(6, dtype=torch.float32),
            ),
        ),
    ],
)
def test_classification_sign_and_masked_scatter_lower(name, fn, inputs):
    verdict = validate_op(fn, inputs, name=name)
    assert verdict.error is None, verdict.error
    assert verdict.lowered, verdict.opaque_calls
    assert verdict.shape_ok


@pytest.mark.parametrize(
    "name,fn,inputs",
    [
        ("prod_all", torch.ops.aten.prod.default, (torch.rand(2, 3) + 0.5,)),
        (
            "prod_dim_keep",
            lambda x: torch.ops.aten.prod.dim_int(x, 1, True),
            (torch.rand(2, 3) + 0.5,),
        ),
        (
            "prod_int",
            lambda x: torch.ops.aten.prod.dim_int(x, 0, False),
            (torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64),),
        ),
    ],
)
def test_prod_reductions_lower(name, fn, inputs):
    verdict = validate_op(fn, inputs, name=name)
    assert verdict.error is None, verdict.error
    assert verdict.lowered, verdict.opaque_calls
    assert verdict.shape_ok


@pytest.mark.parametrize(
    "name,fn,inputs",
    [
        (
            "elu_parameters",
            lambda x: torch.ops.aten.elu.default(x, 1.25, 0.75, 0.5),
            (torch.linspace(-2, 2, 6).reshape(2, 3),),
        ),
        (
            "div_floor_tensor_i64",
            lambda x, y: torch.ops.aten.div.Tensor_mode(x, y, rounding_mode="floor"),
            (
                torch.tensor([[-5, -4, 4, 5]], dtype=torch.int64),
                torch.tensor([[2, -2, 2, -2]], dtype=torch.int64),
            ),
        ),
        (
            "div_trunc_scalar_i64",
            lambda x: torch.ops.aten.div.Scalar_mode(x, -2, rounding_mode="trunc"),
            (torch.tensor([[-5, -4, 4, 5]], dtype=torch.int64),),
        ),
    ],
)
def test_elu_and_rounded_division_lower(name, fn, inputs):
    verdict = validate_op(fn, inputs, name=name)
    assert verdict.error is None, verdict.error
    assert verdict.lowered, verdict.opaque_calls
    assert verdict.shape_ok


@pytest.mark.parametrize(
    "name,fn,inputs",
    [
        (
            "flip_multi_dim",
            lambda x: torch.ops.aten.flip.default(x, [0, -1]),
            (torch.arange(24).reshape(2, 3, 4),),
        ),
        (
            "gather_dim_1",
            lambda x, index: torch.ops.aten.gather.default(x, 1, index),
            (
                torch.arange(12, dtype=torch.float32).reshape(3, 4),
                torch.tensor([[3, 0], [1, 2], [0, 0]], dtype=torch.int64),
            ),
        ),
    ],
)
def test_flip_and_gather_lower(name, fn, inputs):
    verdict = validate_op(fn, inputs, name=name)
    assert verdict.error is None, verdict.error
    assert verdict.lowered, verdict.opaque_calls
    assert verdict.shape_ok


_NAN = float("nan")

_NATIVE = {
    "clamp_tensor_nan_and_ties": (
        torch.ops.aten.clamp.Tensor,
        (
            torch.tensor([_NAN, 1.0, -0.0, 0.0, 5.0, 2.0]),
            torch.tensor([0.0, _NAN, 0.0, -0.0, 1.0, 3.0]),
            torch.tensor([1.0, 2.0, 1.0, 1.0, _NAN, 1.0]),
        ),
    ),
    "clamp_tensor_min_only": (
        lambda x, lo: torch.ops.aten.clamp.Tensor(x, lo),
        (torch.tensor([_NAN, 1.0, -0.0, 5.0]), torch.tensor([0.0, _NAN, 0.0, 6.0])),
    ),
    "clamp_tensor_max_only": (
        lambda x, hi: torch.ops.aten.clamp.Tensor(x, None, hi),
        (torch.tensor([_NAN, 1.0, -0.0, 5.0]), torch.tensor([0.0, _NAN, 0.0, 6.0])),
    ),
    "clamp_tensor_uint8": (
        torch.ops.aten.clamp.Tensor,
        (
            torch.tensor([5, 200, 7], dtype=torch.uint8),
            torch.tensor([6, 0, 0], dtype=torch.uint8),
            torch.tensor([9, 250, 3], dtype=torch.uint8),
        ),
    ),
    "logical_or_float_truthiness": (
        torch.ops.aten.logical_or.default,
        (torch.tensor([0.0, 0.5, _NAN, 0.0]), torch.tensor([0.0, 0.0, 0.0, -0.0])),
    ),
    "bitwise_or_scalar": (
        lambda x: torch.ops.aten.bitwise_or.Scalar(x, 3),
        (torch.arange(-3, 3, dtype=torch.int64).reshape(2, 3),),
    ),
    "bitwise_xor_scalar": (
        lambda x: torch.ops.aten.bitwise_xor.Scalar(x, 3),
        (torch.arange(-3, 3, dtype=torch.int64).reshape(2, 3),),
    ),
    "masked_scatter_2d_source": (
        torch.ops.aten.masked_scatter.default,
        (
            torch.zeros(2, 3),
            torch.tensor([[True, False, True], [False, True, True]]),
            torch.arange(8.0).reshape(2, 4),
        ),
    ),
    "sign_nan_and_zeros": (
        torch.ops.aten.sign.default,
        (torch.tensor([_NAN, -0.0, 0.0, -2.0, 3.0]),),
    ),
    "gather": (
        lambda x, index: torch.ops.aten.gather.default(x, 1, index),
        (torch.arange(12.0).reshape(3, 4), torch.tensor([[3, 0], [1, 2], [0, 0]])),
    ),
    "flip": (
        lambda x: torch.ops.aten.flip.default(x, [0, -1]),
        (torch.arange(24.0).reshape(2, 3, 4),),
    ),
    "prod_keepdim": (
        lambda x: torch.ops.aten.prod.dim_int(x, 1, True),
        (torch.arange(1.0, 7.0).reshape(2, 3),),
    ),
    "elu_parameters": (
        lambda x: torch.ops.aten.elu.default(x, 1.25, 0.75, 0.5),
        (torch.linspace(-2, 2, 6),),
    ),
    "constant_pad_fill": (
        lambda x: torch.ops.aten.constant_pad_nd.default(x, [1, 1], 2.5),
        (torch.ones(2, 2),),
    ),
    "atan2": (
        torch.ops.aten.atan2.default,
        (torch.tensor([1.0, -1.0, 0.0, -0.0]), torch.tensor([-1.0, -1.0, -1.0, 1.0])),
    ),
}


@pytest.mark.parametrize("name", _NATIVE)
def test_core_aten_gap_ops_match_eager_natively(tmp_path, name):
    fn, inputs = _NATIVE[name]
    assert_native_matches_eager(tmp_path, fn, inputs)
