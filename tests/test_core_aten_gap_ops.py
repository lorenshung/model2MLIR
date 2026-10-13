"""Focused coverage for Core ATen operators found by the full-op corpus."""

from __future__ import annotations

import pytest
import torch

from m2m.coverage import CANONICAL_FAMILIES, family_histogram, untagged_compute_ops, validate_op
from tests.native_execution import assert_native_matches_eager


class _Module(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def _convert(fn, inputs):
    import m2m

    result = m2m.convert(_Module(fn).eval(), inputs, backend="fx_importer")
    assert result.ok
    assert "func.call" not in result.mlir_text
    return result.mlir_text


@pytest.mark.parametrize(
    "name,value,expected_type",
    [
        ("float", torch.tensor(-3.25), "f32"),
        ("int64", torch.tensor(-7, dtype=torch.int64), "i64"),
        ("bool", torch.tensor(True), "i1"),
    ],
)
def test_local_scalar_dense_preserves_scalar_abi_and_value(name, value, expected_type):
    fn = torch.ops.aten._local_scalar_dense.default
    eager = fn(value)
    assert eager == value.item()

    mlir = _convert(fn, (value,))
    assert f") -> {expected_type}" in mlir
    assert "func.return %" in mlir and f": {expected_type}" in mlir
    assert "tensor.extract" in mlir
    assert 'prov.aten = "aten._local_scalar_dense.default"' in mlir


def test_local_scalar_dense_runtime_value_flows_to_tensor_consumer():
    fn = lambda scalar, values: torch.ops.aten.add.Tensor(
        values, torch.ops.aten._local_scalar_dense.default(scalar)
    )
    scalar = torch.tensor(-2.5)
    values = torch.tensor([1.0, 3.0])
    torch.testing.assert_close(fn(scalar, values), torch.tensor([-1.5, 0.5]))

    mlir = _convert(fn, (scalar, values))
    assert "tensor.splat" in mlir
    assert 'prov.aten = "aten._local_scalar_dense.default"' in mlir
    assert 'prov.aten = "aten.add.Tensor"' in mlir


def test_local_scalar_dense_corpus_tensor_wrapper_remains_exact_and_clean():
    """The portable corpus wraps scalar results in a rank-0 tensor for its ABI."""
    fn = lambda scalar: torch.ops.aten.scalar_tensor.default(
        torch.ops.aten._local_scalar_dense.default(scalar)
    )
    value = torch.tensor(2.5)
    torch.testing.assert_close(fn(value), value)

    mlir = _convert(fn, (value,))
    assert "tensor.splat" in mlir
    assert 'prov.aten = "aten._local_scalar_dense.default"' in mlir
    assert 'prov.aten = "aten.scalar_tensor.default"' in mlir


def test_literal_full_like_still_lowers_after_runtime_scalar_support():
    """A literal fill has no scalar SSA operand and must retain its old path."""
    fn = lambda x: torch.ops.aten.full_like.default(x, 3.0)
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch.testing.assert_close(fn(value), torch.full_like(value, 3.0))

    mlir = _convert(fn, (value,))
    assert "tensor.splat" in mlir
    assert "func.call" not in mlir
    assert 'prov.aten = "aten.full_like.default"' in mlir


@pytest.mark.parametrize(
    "name,value,expected",
    [
        (
            "float_fraction_nan",
            torch.tensor([[0.0, 0.25, float("nan")], [-0.0, -3.0, 0.0]]),
            torch.tensor([[0, 1], [0, 2], [1, 1]], dtype=torch.int64),
        ),
        (
            "int64",
            torch.tensor([[0, 2], [-1, 0]], dtype=torch.int64),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
        ),
        (
            "bool",
            torch.tensor([[False, True], [True, False]]),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
        ),
    ],
)
def test_nonzero_dynamic_compaction_has_exact_numeric_contract(name, value, expected):
    fn = torch.ops.aten.nonzero.default
    torch.testing.assert_close(fn(value), expected)

    mlir = _convert(fn, (value,))
    assert f"tensor<?x{value.ndim}xi64>" in mlir
    assert "tensor<-1" not in mlir
    assert mlir.count("scf.for") == 2
    assert "scf.if" in mlir
    assert 'prov.aten = "aten.nonzero.default"' in mlir
    # Floating truthiness must compare to zero, not truncate fractions.
    if value.is_floating_point():
        assert "arith.cmpf une" in mlir
    else:
        assert "arith.cmpi ne" in mlir


def test_nonzero_capture_does_not_specialize_data_dependent_count():
    fn = torch.ops.aten.nonzero.default
    sparse = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    dense = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert fn(sparse).shape[0] == 1
    assert fn(dense).shape[0] == 4

    sparse_mlir = _convert(fn, (sparse,))
    dense_mlir = _convert(fn, (dense,))
    assert sparse_mlir == dense_mlir
    assert "tensor<?x2xi64>" in sparse_mlir


@pytest.mark.parametrize(
    "value,expected_rows",
    [(torch.tensor(0.0), 0), (torch.tensor(2.0), 1)],
)
def test_nonzero_scalar_uses_dynamic_rows_and_zero_coordinate_columns(value, expected_rows):
    fn = torch.ops.aten.nonzero.default
    eager = fn(value)
    assert tuple(eager.shape) == (expected_rows, 0)
    mlir = _convert(fn, (value,))
    assert "tensor<?x0xi64>" in mlir
    assert "tensor<-1" not in mlir


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


@pytest.mark.parametrize(
    "name,fn",
    [
        ("fmod_tensor", lambda a, b: torch.ops.aten.fmod.Tensor(a, b)),
        ("fmod_scalar", lambda a, b: torch.ops.aten.fmod.Scalar(a, -2.0)),
        ("remainder_tensor", lambda a, b: torch.ops.aten.remainder.Tensor(a, b)),
        ("remainder_scalar", lambda a, b: torch.ops.aten.remainder.Scalar(a, -2.0)),
    ],
)
def test_fmod_and_remainder_lower_to_standard_rem(name, fn):
    a = torch.tensor([-4.0, -3.0, 3.0, 4.0], dtype=torch.float32)
    b = torch.tensor([2.0, 2.0, -2.0, -2.0], dtype=torch.float32)
    verdict = validate_op(fn, (a, b), name=name)
    assert verdict.error is None, verdict.error
    assert verdict.lowered, verdict.opaque_calls
    assert verdict.shape_ok
    mlir = _convert(fn, (a, b))
    assert "arith.remf" in mlir


@pytest.mark.parametrize(
    "name,fn,expected_shape",
    [
        (
            "var_dim_biased",
            lambda x: torch.ops.aten.var.dim(x, [0], False, False),
            (3,),
        ),
        (
            "var_dim_unbiased_keepdim",
            lambda x: torch.ops.aten.var.dim(x, [1], True, True),
            (2, 1),
        ),
        (
            "var_correction_zero",
            lambda x: torch.ops.aten.var.correction(
                x, [0, 1], correction=0, keepdim=False
            ),
            (),
        ),
        (
            "var_correction_two_keepdim",
            lambda x: torch.ops.aten.var.correction(
                x, [1], correction=2, keepdim=True
            ),
            (2, 1),
        ),
    ],
)
def test_variance_reductions_lower_without_opaque_calls(name, fn, expected_shape):
    inputs = (torch.tensor([[0.2, 0.6, 1.0], [0.8, 1.2, 1.6]]),)
    expected = fn(*inputs)
    assert tuple(expected.shape) == expected_shape
    verdict = validate_op(fn, inputs, name=name)
    assert verdict.error is None, verdict.error
    assert verdict.lowered, verdict.opaque_calls
    assert verdict.shape_ok
    mlir = _convert(fn, inputs)
    assert mlir.count("linalg.reduce") == 2
    assert "arith.subf" in mlir
    assert "arith.mulf" in mlir
    assert mlir.count("arith.divf") >= 2


def test_var_mean_preserves_two_result_abi():
    fn = lambda x: torch.ops.aten.var_mean.correction(
        x, [1], correction=1, keepdim=True
    )
    inputs = (torch.tensor([[0.2, 0.6, 1.0], [0.8, 1.2, 1.6]]),)
    mlir = _convert(fn, inputs)
    assert "-> (tensor<2x1xf32>, tensor<2x1xf32>)" in mlir
    assert mlir.count("linalg.reduce") == 3


@pytest.mark.parametrize(
    "name,fn,inputs,expected_shape",
    [
        (
            "avg_pool1d",
            lambda x: torch.ops.aten.avg_pool1d.default(x, [2], [2], [0], False, True),
            (torch.arange(12, dtype=torch.float32).reshape(1, 2, 6),),
            (1, 2, 3),
        ),
        (
            "avg_pool2d_padded",
            lambda x: torch.ops.aten.avg_pool2d.default(x, [2, 2], [2, 2], [1, 1], False, True),
            (torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),),
            (1, 1, 3, 3),
        ),
        (
            "avg_pool3d",
            lambda x: torch.ops.aten.avg_pool3d.default(x, [2, 2, 2], [2, 2, 2], [0, 0, 0], False, True),
            (torch.arange(64, dtype=torch.float32).reshape(1, 1, 4, 4, 4),),
            (1, 1, 2, 2, 2),
        ),
        (
            "adaptive_avg_pool1d",
            lambda x: torch.ops.aten.adaptive_avg_pool1d.default(x, [3]),
            (torch.arange(12, dtype=torch.float32).reshape(1, 2, 6),),
            (1, 2, 3),
        ),
        (
            "adaptive_avg_pool2d",
            lambda x: torch.ops.aten._adaptive_avg_pool2d.default(x, [2, 2]),
            (torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4),),
            (1, 2, 2, 2),
        ),
        (
            "adaptive_avg_pool3d",
            lambda x: torch.ops.aten._adaptive_avg_pool3d.default(x, [2, 2, 2]),
            (torch.arange(64, dtype=torch.float32).reshape(1, 1, 4, 4, 4),),
            (1, 1, 2, 2, 2),
        ),
    ],
)
def test_average_pooling_lowers_to_window_reduction(name, fn, inputs, expected_shape):
    assert tuple(fn(*inputs).shape) == expected_shape
    verdict = validate_op(fn, inputs, name=name)
    assert verdict.error is None, verdict.error
    assert verdict.lowered, verdict.opaque_calls
    assert verdict.shape_ok
    mlir = _convert(fn, inputs)
    assert "linalg.generic" in mlir
    assert "scf.for" in mlir
    assert "arith.addf" in mlir
    assert "arith.divf" in mlir


@pytest.mark.parametrize(
    "name,fn",
    [
        (
            "avg_pool2d_backward",
            lambda grad, x: torch.ops.aten.avg_pool2d_backward.default(
                grad, x, [2, 2], [2, 2], [0, 0], False, True, None
            ),
        ),
        (
            "adaptive_avg_pool2d_backward",
            torch.ops.aten._adaptive_avg_pool2d_backward.default,
        ),
    ],
)
def test_average_pool_backward_lowers_to_scatter_add(name, fn):
    inputs = (torch.ones(1, 1, 2, 2), torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4))
    verdict = validate_op(fn, inputs, name=name)
    assert verdict.error is None, verdict.error
    assert verdict.lowered, verdict.opaque_calls
    assert verdict.shape_ok
    mlir = _convert(fn, inputs)
    assert "scf.for" in mlir
    assert "linalg.generic" in mlir


@pytest.mark.parametrize(
    "name,fn,inputs",
    [
        (
            "cdist",
            lambda x, y: torch.ops.aten._cdist_forward.default(x, y, 2.0, None),
            (torch.randn(2, 3), torch.randn(4, 3)),
        ),
        (
            "pdist",
            lambda x: torch.ops.aten._pdist_forward.default(x, 2.0),
            (torch.randn(4, 3),),
        ),
    ],
)
def test_pairwise_distance_lowers_to_reduction(name, fn, inputs):
    verdict = validate_op(fn, inputs, name=name)
    assert verdict.error is None, verdict.error
    assert verdict.lowered, verdict.opaque_calls
    assert verdict.shape_ok
    mlir = _convert(fn, inputs)
    assert "scf.for" in mlir
    assert "math.sqrt" in mlir


@pytest.mark.parametrize(
    "name,fn,signature",
    [
        ("sort", lambda x: torch.ops.aten.sort.default(x, 1, True), "tensor<2x3xf32>, tensor<2x3xi64>"),
        ("topk", lambda x: torch.ops.aten.topk.default(x, 2, 1, True, True), "tensor<2x2xf32>, tensor<2x2xi64>"),
    ],
)
def test_sort_topk_preserve_value_index_abi(name, fn, signature):
    mlir = _convert(fn, (torch.tensor([[3.0, 1.0, 2.0], [0.0, 4.0, -1.0]]),))
    assert signature in mlir
    assert "arith.cmpf" in mlir
    assert "tensor.insert" in mlir


def test_embedding_bag_sum_preserves_four_result_abi():
    fn = lambda weight, indices, offsets: torch.ops.aten._embedding_bag.default(
        weight, indices, offsets, False, 0, False, None, False, -1
    )
    inputs = (torch.randn(5, 3), torch.tensor([0, 1, 2, 1]), torch.tensor([0, 2]))
    mlir = _convert(fn, inputs)
    assert "tensor<2x3xf32>, tensor<0xi64>, tensor<2xi64>, tensor<2xi64>" in mlir
    assert "scf.for" in mlir
    assert "cf.assert" in mlir


@pytest.mark.parametrize(
    "name,fn",
    [
        ("scatter_src", lambda x, index, src: torch.ops.aten.scatter.src(x, 1, index, src)),
        ("scatter_value", lambda x, index, src: torch.ops.aten.scatter.value(x, 1, index, 2.0)),
        ("scatter_add", lambda x, index, src: torch.ops.aten.scatter_add.default(x, 1, index, src)),
        (
            "scatter_reduce_sum",
            lambda x, index, src: torch.ops.aten.scatter_reduce.two(
                x, 1, index, src, "sum", include_self=True
            ),
        ),
    ],
)
def test_scatter_family_lowers_to_deterministic_loop(name, fn):
    inputs = (
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.tensor([[0, 2], [1, 0]], dtype=torch.int64),
        torch.ones(2, 2, dtype=torch.float32),
    )
    verdict = validate_op(fn, inputs, name=name)
    assert verdict.error is None, verdict.error
    assert verdict.lowered, verdict.opaque_calls
    assert verdict.shape_ok
    mlir = _convert(fn, inputs)
    assert "scf.for" in mlir
    assert "tensor.extract" in mlir
    assert "scf.if" in mlir


@pytest.mark.parametrize("accumulate", [False, True])
def test_integer_index_put_rows_lower_to_deterministic_loop(accumulate):
    fn = lambda x, index, values: torch.ops.aten.index_put.default(
        x, [index], values, accumulate
    )
    inputs = (
        torch.arange(12, dtype=torch.float32).reshape(4, 3),
        torch.tensor([1, 3], dtype=torch.int64),
        torch.ones(2, 3, dtype=torch.float32),
    )
    verdict = validate_op(fn, inputs, name=f"index_put_rows_{accumulate}")
    assert verdict.error is None, verdict.error
    assert verdict.lowered, verdict.opaque_calls
    assert verdict.shape_ok
    mlir = _convert(fn, inputs)
    assert "scf.for" in mlir
    assert "tensor.extract" in mlir
    assert "tensor.insert" in mlir


def test_as_strided_static_view_lowers_to_storage_gather():
    fn = lambda x: torch.ops.aten.as_strided.default(x, [3, 3], [4, 1], 0)
    inputs = (torch.arange(16, dtype=torch.float32).reshape(4, 4),)
    verdict = validate_op(fn, inputs, name="as_strided")
    assert verdict.error is None, verdict.error
    assert verdict.lowered, verdict.opaque_calls
    assert verdict.shape_ok
    mlir = _convert(fn, inputs)
    assert "linalg.index" in mlir
    assert "tensor.extract" in mlir
    assert "arith.muli" in mlir


def test_empty_strided_lowers_to_tensor_empty():
    fn = lambda: torch.ops.aten.empty_strided.default(
        [2, 3], [3, 1], dtype=torch.float32, device=torch.device("cpu")
    )
    mlir = _convert(lambda: fn(), ())
    assert "tensor.empty" in mlir
    assert "func.call" not in mlir


@pytest.mark.parametrize(
    "opname,fn",
    [
        (
            "addf",
            lambda a, b: torch.ops.aten.add.Tensor(a, b, alpha=2),
        ),
        (
            "subf",
            lambda a, b: torch.ops.aten.sub.Tensor(a, b, alpha=2),
        ),
    ],
)
def test_add_sub_preserve_nondefault_alpha(opname, fn):
    mlir = _convert(fn, (torch.randn(2, 3), torch.randn(2, 3)))
    assert "arith.constant 2.000000e+00 : f32" in mlir
    assert "arith.mulf" in mlir
    assert f"arith.{opname}" in mlir


@pytest.mark.parametrize(
    "opname,fn",
    [
        ("addf", lambda a: torch.ops.aten.add.Scalar(a, 0.5, alpha=2)),
        ("subf", lambda a: torch.ops.aten.sub.Scalar(a, 0.5, alpha=2)),
    ],
)
def test_add_sub_scalar_preserve_nondefault_alpha(opname, fn):
    mlir = _convert(fn, (torch.randn(2, 3),))
    assert "5.000000e-01 : f32" in mlir
    assert "arith.constant 2.000000e+00 : f32" in mlir
    assert "arith.mulf" in mlir
    assert f"arith.{opname}" in mlir


def test_addmm_preserves_alpha_beta_and_initializes_compensated_reduction():
    fn = lambda bias, a, b: torch.ops.aten.addmm.default(
        bias, a, b, beta=1.5, alpha=0.5
    )
    mlir = _convert(
        fn,
        (torch.randn(2), torch.randn(2, 3), torch.randn(3, 2)),
    )
    assert "5.000000e-01 : f64" in mlir
    assert "1.500000e+00 : f32" in mlir
    assert "0.000000e+00 : f64" in mlir
    assert "tensor.splat" in mlir
    assert 'prov.accumulation = "compensated_f64"' in mlir
    assert 'iterator_types = ["parallel", "parallel", "reduction"]' in mlir
    assert "arith.extf" in mlir
    assert "arith.truncf" in mlir
    assert mlir.count("arith.mulf") >= 2


def test_native_layer_norm_preserves_three_result_abi():
    fn = lambda x, weight, bias: torch.ops.aten.native_layer_norm.default(
        x, [3], weight, bias, 1e-5
    )
    mlir = _convert(
        fn,
        (torch.randn(2, 3), torch.randn(3), torch.randn(3)),
    )
    assert "-> (tensor<2x3xf32>, tensor<2x1xf32>, tensor<2x1xf32>)" in mlir
    assert "func.return" in mlir
    assert "tensor<2x3xf32>, tensor<2x1xf32>, tensor<2x1xf32>" in mlir


def test_batch_norm_inference_preserves_three_result_abi():
    fn = lambda x, weight, bias, mean, var: (
        torch.ops.aten._native_batch_norm_legit_no_training.default(
            x, weight, bias, mean, var, 0.1, 1e-5
        )
    )
    inputs = (
        torch.randn(2, 4, 2, 2),
        torch.randn(4),
        torch.randn(4),
        torch.randn(4),
        torch.rand(4) + 0.5,
    )
    mlir = _convert(fn, inputs)
    assert "-> (tensor<2x4x2x2xf32>, tensor<0xf32>, tensor<0xf32>)" in mlir
    assert "tensor<2x4x2x2xf32>, tensor<0xf32>, tensor<0xf32>" in mlir
def _convolution_backward(
    grad_output,
    source,
    weight,
    *,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    output_mask=(True, True, True),
):
    return torch.ops.aten.convolution_backward.default(
        grad_output,
        source,
        weight,
        [weight.shape[0]],
        list(stride),
        list(padding),
        list(dilation),
        False,
        [0, 0],
        groups,
        list(output_mask),
    )


@pytest.mark.parametrize(
    "shape,weight_shape,stride,padding,dilation",
    [
        ((1, 1, 5, 5), (2, 1, 3, 3), (1, 1), (0, 0), (1, 1)),
        ((2, 2, 6, 7), (3, 2, 3, 3), (2, 2), (1, 1), (1, 1)),
        ((1, 2, 7, 8), (3, 2, 3, 2), (1, 1), (2, 1), (2, 2)),
    ],
)
def test_convolution_backward_eager_matches_autograd_reference(
    shape, weight_shape, stride, padding, dilation
):
    """Numerically validate the exact ATen cases used by the capture tests."""
    generator = torch.Generator().manual_seed(314159)
    source = torch.randn(shape, generator=generator, requires_grad=True)
    weight = torch.randn(weight_shape, generator=generator, requires_grad=True)
    bias = torch.randn(weight_shape[0], generator=generator, requires_grad=True)
    output = torch.nn.functional.conv2d(
        source, weight, bias, stride=stride, padding=padding, dilation=dilation
    )
    grad_output = torch.randn(output.shape, generator=generator)
    expected = torch.autograd.grad(output, (source, weight, bias), grad_output)
    actual = _convolution_backward(
        grad_output,
        source.detach(),
        weight.detach(),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "shape,weight_shape,stride,padding,dilation,output_mask,return_types",
    [
        (
            (1, 1, 5, 5), (2, 1, 3, 3), (1, 1), (0, 0), (1, 1),
            (True, True, True),
            ("tensor<1x1x5x5xf32>", "tensor<2x1x3x3xf32>", "tensor<2xf32>"),
        ),
        (
            (2, 2, 6, 7), (3, 2, 3, 3), (2, 2), (1, 1), (1, 1),
            (True, False, True),
            ("tensor<2x2x6x7xf32>", "tensor<3xf32>"),
        ),
        (
            (1, 2, 7, 8), (3, 2, 3, 2), (1, 1), (2, 1), (2, 2),
            (False, True, False),
            ("tensor<3x2x3x2xf32>",),
        ),
    ],
)
def test_convolution_backward_captures_standard_contractions_with_tuple_abi(
    shape, weight_shape, stride, padding, dilation, output_mask, return_types
):
    source = torch.linspace(-0.75, 1.25, torch.tensor(shape).prod().item()).reshape(shape)
    weight = torch.linspace(-0.5, 0.75, torch.tensor(weight_shape).prod().item()).reshape(weight_shape)
    output = torch.nn.functional.conv2d(
        source, weight, stride=stride, padding=padding, dilation=dilation
    )
    grad_output = torch.linspace(0.25, 1.0, output.numel()).reshape(output.shape)

    def kernel(grad, x, w):
        return _convolution_backward(
            grad,
            x,
            w,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_mask=output_mask,
        )

    text = _convert(kernel, (grad_output, source, weight))
    assert 'prov.aten = "aten.convolution_backward.default"' in text
    assert 'prov.conv_path = "backward_contractions"' in text
    return_line = next(line.strip() for line in text.splitlines() if "func.return" in line)
    assert all(result_type in return_line for result_type in return_types)
    assert return_line.count("tensor<") == len(return_types)
    families = set(family_histogram(text))
    assert families <= CANONICAL_FAMILIES
    assert "contraction" in families
    assert untagged_compute_ops(text) == 0


def test_convolution_backward_grouped_weight_and_bias_capture():
    """Grouped channel arithmetic is exercised without claiming grouped dInput."""
    source = torch.randn(1, 4, 5, 5)
    weight = torch.randn(6, 2, 3, 3)
    grad_output = torch.randn(1, 6, 3, 3)

    def kernel(grad, x, w):
        return _convolution_backward(
            grad, x, w, groups=2, output_mask=(False, True, True)
        )

    text = _convert(kernel, (grad_output, source, weight))
    assert 'prov.aten = "aten.convolution_backward.default"' in text
    assert "func.call" not in text
    return_line = next(line.strip() for line in text.splitlines() if "func.return" in line)
    assert "tensor<6x2x3x3xf32>" in return_line
    assert "tensor<6xf32>" in return_line
    assert set(family_histogram(text)) <= CANONICAL_FAMILIES
    assert untagged_compute_ops(text) == 0


def test_convolution_backward_all_false_mask_is_a_zero_result_function():
    source = torch.randn(1, 1, 5, 5)
    weight = torch.randn(2, 1, 3, 3)
    grad_output = torch.randn(1, 2, 3, 3)

    def kernel(grad, x, w):
        return _convolution_backward(
            grad, x, w, output_mask=(False, False, False)
        )

    text = _convert(kernel, (grad_output, source, weight))
    assert "func.func @forward" in text
    assert "func.return" in text
    assert "func.call" not in text
    assert "linalg." not in text


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
    "add_alpha": (
        lambda a, b: torch.ops.aten.add.Tensor(a, b, alpha=2),
        (torch.randn(2, 3), torch.randn(2, 3)),
    ),
    "sub_scalar_alpha_i64": (
        lambda a: torch.ops.aten.sub.Scalar(a, 3, alpha=2),
        (torch.arange(6).reshape(2, 3),),
    ),
    "addmm_alpha_beta": (
        lambda bias, a, b: torch.ops.aten.addmm.default(bias, a, b, beta=1.5, alpha=0.5),
        (torch.randn(2), torch.randn(2, 3), torch.randn(3, 2)),
    ),
    "addmm_zero_beta_ignores_nonfinite_bias": (
        lambda bias, a, b: torch.ops.aten.addmm.default(bias, a, b, beta=0),
        (torch.tensor([_NAN, float("inf")]), torch.randn(2, 3), torch.randn(3, 2)),
    ),
    "native_layer_norm_results": (
        lambda x, w, b: torch.ops.aten.native_layer_norm.default(x, [3], w, b, 1e-5),
        (torch.randn(2, 3), torch.randn(3), torch.randn(3)),
    ),
    "native_layer_norm_bias_only": (
        lambda x, b: torch.ops.aten.native_layer_norm.default(x, [3], None, b, 1e-5),
        (torch.randn(2, 3), torch.randn(3)),
    ),
    "batch_norm_no_training_results": (
        lambda x, w, b, m, v: torch.ops.aten._native_batch_norm_legit_no_training.default(
            x, w, b, m, v, 0.1, 1e-5
        ),
        (torch.randn(2, 4, 2, 2), torch.randn(4), torch.randn(4), torch.randn(4),
         torch.rand(4) + 0.5),
    ),
    "var_dim_unbiased": (
        lambda x: torch.ops.aten.var.dim(x, [0], False, False),
        (torch.randn(2, 3),),
    ),
    "var_correction_keepdim": (
        lambda x: torch.ops.aten.var.correction(x, [1], correction=1, keepdim=True),
        (torch.randn(2, 3),),
    ),
    "var_correction_beyond_count": (
        lambda x: torch.ops.aten.var.correction(x, [1], correction=5),
        (torch.randn(2, 3),),
    ),
    "var_mean": (
        lambda x: torch.ops.aten.var_mean.correction(x, [0, 2], correction=0),
        (torch.randn(2, 3, 4),),
    ),
    "avg_pool2d_padded": (
        lambda x: torch.ops.aten.avg_pool2d.default(x, [3, 3], [2, 2], [1, 1]),
        (torch.randn(1, 2, 5, 5),),
    ),
    "avg_pool1d": (
        lambda x: torch.ops.aten.avg_pool1d.default(x, [2]),
        (torch.randn(1, 2, 6),),
    ),
    "avg_pool3d_divisor": (
        lambda x: torch.ops.aten.avg_pool3d.default(x, [2, 2, 2], [2, 2, 2], [0, 0, 0], False, True, 3),
        (torch.randn(1, 1, 4, 4, 4),),
    ),
    "adaptive_avg_pool2d": (
        lambda x: torch.ops.aten._adaptive_avg_pool2d.default(x, [2, 2]),
        (torch.randn(1, 2, 4, 6),),
    ),
    "avg_pool2d_backward_default_stride": (
        lambda g, x: torch.ops.aten.avg_pool2d_backward.default(g, x, [2, 2], [], [0, 0], False, True, None),
        (torch.randn(1, 1, 2, 2), torch.randn(1, 1, 4, 4)),
    ),
    "adaptive_avg_pool2d_backward": (
        torch.ops.aten._adaptive_avg_pool2d_backward.default,
        (torch.randn(1, 1, 2, 2), torch.randn(1, 1, 4, 4)),
    ),
    "cdist": (
        lambda a, b: torch.ops.aten._cdist_forward.default(a, b, 2.0, None),
        (torch.randn(3, 4), torch.randn(2, 4)),
    ),
    "pdist": (
        lambda a: torch.ops.aten._pdist_forward.default(a, 2.0),
        (torch.randn(4, 3),),
    ),
    "sort_descending_with_nan": (
        lambda x: torch.ops.aten.sort.default(x, 1, True),
        (torch.tensor([[1.0, _NAN, -2.0, 3.0], [0.5, 0.25, _NAN, _NAN]]),),
    ),
    "sort_ascending_with_nan": (
        lambda x: torch.ops.aten.sort.default(x, 0, False),
        (torch.tensor([[1.0, _NAN], [_NAN, -2.0], [3.0, 0.0]]),),
    ),
    "topk_smallest": (
        lambda x: torch.ops.aten.topk.default(x, 2, 1, False, True),
        (torch.tensor([[3.0, 1.0, 2.0], [0.0, -1.0, 5.0]]),),
    ),
    "embedding_bag_sum": (
        lambda w, i, o: torch.ops.aten._embedding_bag.default(w, i, o, False, 0, False, None, False, -1),
        (torch.randn(5, 3), torch.tensor([0, 2, 4, 1]), torch.tensor([0, 2])),
    ),
    "scatter_src": (
        lambda x, i, src: torch.ops.aten.scatter.src(x, 1, i, src),
        (torch.zeros(2, 3), torch.tensor([[2, 0], [1, 2]]), torch.randn(2, 2)),
    ),
    "scatter_value": (
        lambda x, i: torch.ops.aten.scatter.value(x, 1, i, 2.0),
        (torch.zeros(2, 3), torch.tensor([[2, 0], [1, 2]])),
    ),
    "scatter_add_repeated": (
        lambda x, i, src: torch.ops.aten.scatter_add.default(x, 1, i, src),
        (torch.ones(2, 3), torch.tensor([[1, 1], [0, 0]]), torch.randn(2, 2)),
    ),
    "scatter_reduce_sum": (
        lambda x, i, src: torch.ops.aten.scatter_reduce.two(x, 1, i, src, "sum"),
        (torch.ones(2, 3), torch.tensor([[1, 1], [0, 2]]), torch.randn(2, 2)),
    ),
    "index_put_integer_broadcast": (
        lambda x, r, c, v: torch.ops.aten.index_put.default(x, [r, c], v, True),
        (torch.zeros(3, 4), torch.tensor([[0], [2]]), torch.tensor([1, 3, 1]), torch.randn(2, 3)),
    ),
    "fmod_signs": (
        torch.ops.aten.fmod.Tensor,
        (torch.tensor([5.5, -5.5, 5.5, -5.5, -0.0]), torch.tensor([2.0, 2.0, -2.0, -2.0, 1.0])),
    ),
    "remainder_signs": (
        torch.ops.aten.remainder.Tensor,
        (torch.tensor([5.5, -5.5, 5.5, -5.5, -4.0]), torch.tensor([2.0, 2.0, -2.0, -2.0, 2.0])),
    ),
    "remainder_scalar_i64": (
        lambda x: torch.ops.aten.remainder.Scalar(x, -3),
        (torch.arange(-4, 4),),
    ),
    "as_strided_subview": (
        lambda x: torch.ops.aten.as_strided.default(x, [2, 2], [4, 1], 1),
        (torch.arange(12.0).reshape(3, 4),),
    ),
    "local_scalar_dense_feeds_add": (
        lambda scalar, values: torch.ops.aten.add.Tensor(
            values, torch.ops.aten._local_scalar_dense.default(scalar)
        ),
        (torch.tensor(-2.5), torch.tensor([1.0, 3.0])),
    ),
    "convolution_backward_strided_padded": (
        lambda g, x, w: torch.ops.aten.convolution_backward.default(
            g, x, w, [3], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]
        ),
        (torch.randn(1, 3, 3, 3), torch.randn(1, 2, 6, 6), torch.randn(3, 2, 3, 3)),
    ),
    "convolution_backward_grouped_weight_bias": (
        lambda g, x, w: torch.ops.aten.convolution_backward.default(
            g, x, w, [4], [1, 1], [0, 0], [1, 1], False, [0, 0], 2, [False, True, True]
        )[1:],
        (torch.randn(1, 4, 3, 3), torch.randn(1, 4, 4, 4), torch.randn(4, 2, 2, 2)),
    ),
    "scalar_tensor_from_item": (
        lambda x: torch.ops.aten.scalar_tensor.default(torch.ops.aten._local_scalar_dense.default(x)),
        (torch.tensor(1.75),),
    ),
    "full_from_item_after_size": (
        lambda x: torch.ops.aten.full.default([2, 3], torch.ops.aten._local_scalar_dense.default(x)),
        (torch.tensor(-0.5),),
    ),
    "full_like_literal": (
        lambda x: torch.ops.aten.full_like.default(x, 3.0),
        (torch.arange(6.0).reshape(2, 3),),
    ),
}


@pytest.mark.parametrize("name", _NATIVE)
def test_core_aten_gap_ops_match_eager_natively(tmp_path, name):
    fn, inputs = _NATIVE[name]
    assert_native_matches_eager(tmp_path, fn, inputs)
