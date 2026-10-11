"""Execute structural kernels against eager; no corpus identifiers or sample folding."""

import m2m
import numpy as np
import pytest
import torch
from m2m.coverage import opaque_report
from native_execution import assert_native_matches_eager


@pytest.mark.parametrize("p", [0.0, 1.0, 2.0, 3.5, float("inf")])
@pytest.mark.parametrize("kind", ["cdist", "pdist"])
def test_distance_norms(tmp_path, p, kind):
    x = torch.tensor([[0.0, -1.0, 2.0], [3.0, 4.0, -5.0], [0.0, -1.0, 2.0], [float("nan"), 2.0, 1.0]])
    if kind == "cdist":

        def fn(a, b):
            return torch.ops.aten._cdist_forward(a, b, p, None)

        inputs = (x, torch.tensor([[1.0, 2.0, 3.0], [float("inf"), 0.0, 0.0]]))
    else:

        def fn(a):
            return torch.ops.aten._pdist_forward(a, p)

        inputs = (x,)
    assert_native_matches_eager(tmp_path, fn, inputs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.bool])
@pytest.mark.parametrize("descending", [False, True])
def test_sort_stable_nan_and_large_extent(tmp_path, dtype, descending):
    x = torch.arange(80, dtype=torch.float32).remainder(5).reshape(2, 40).to(dtype)
    if dtype == torch.float32:
        x[:, ::13] = float("nan")
    assert_native_matches_eager(tmp_path, lambda a: torch.sort(a, stable=True, dim=1, descending=descending), (x,))


@pytest.mark.parametrize("shape", [(), (0,), (2, 0)])
def test_sort_scalar_and_empty(tmp_path, shape):
    x = torch.empty(shape)
    if not shape:
        x.fill_(7)
    assert_native_matches_eager(tmp_path, lambda a: torch.sort(a, stable=True), (x,))


@pytest.mark.parametrize("mode", ["sum", "prod", "mean", "amin", "amax"])
@pytest.mark.parametrize("include", [False, True])
def test_scatter_reductions_repeated_indices(tmp_path, mode, include):
    x = torch.tensor([[4.0, -2.0, 6.0], [7.0, -3.0, 9.0]])
    i = torch.tensor([[1, 1], [0, 2]])
    s = torch.tensor([[2.0, -4.0], [float("nan"), 3.0]])
    assert_native_matches_eager(
        tmp_path, lambda a, b, c: torch.scatter_reduce(a, 1, b, c, mode, include_self=include), (x, i, s)
    )


@pytest.mark.parametrize("mode", ["sum", "prod", "mean", "amin", "amax"])
def test_scatter_integer_reductions(tmp_path, mode):
    x = torch.tensor([4, -9, 7], dtype=torch.int32)
    i = torch.tensor([1, 1, 0])
    s = torch.tensor([-2, -3, 8], dtype=torch.int32)
    assert_native_matches_eager(
        tmp_path, lambda a, b, c: torch.scatter_reduce(a, 0, b, c, mode, include_self=False), (x, i, s)
    )


@pytest.mark.parametrize("offset", [-7, -1, 0, 2, 7])
def test_diagonal_offsets_dimensions_and_empty(tmp_path, offset):
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    assert_native_matches_eager(tmp_path, lambda a: torch.diagonal(a, offset, -1, -2), (x,))


@pytest.mark.parametrize("ceil", [False, True])
@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("override", [None, 3])
def test_avg_pool_padding_divisors_and_backward(tmp_path, ceil, include, override):
    x = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5) - 9
    args = ([3, 3], [2, 2], [1, 1], ceil, include, override)
    op = torch.ops.aten.avg_pool2d.default
    g = torch.ones_like(op(x, *args))
    assert_native_matches_eager(
        tmp_path, lambda a, b: (op(a, *args), torch.ops.aten.avg_pool2d_backward(b, a, *args)), (x, g)
    )


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_adaptive_avg_pool_nondivisible_unbatched(tmp_path, ndim):
    x = torch.arange(2 * 5**ndim, dtype=torch.float32).reshape([2] + [5] * ndim)
    op = getattr(torch.ops.aten, "adaptive_avg_pool1d" if ndim == 1 else f"_adaptive_avg_pool{ndim}d").default
    assert_native_matches_eager(tmp_path, lambda a: op(a, [3] * ndim), (x,))


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("dtype", [torch.int8, torch.int32, torch.int64])
def test_integer_pool_values_and_indices(tmp_path, ndim, dtype):
    x = torch.arange(4**ndim, dtype=torch.int64).remainder(9).reshape([1, 1] + [4] * ndim).to(dtype)
    op = getattr(torch.ops.aten, f"max_pool{ndim}d_with_indices").default
    assert_native_matches_eager(tmp_path, lambda a: op(a, [2] * ndim, [2] * ndim, [1] * ndim, [1] * ndim, True), (x,))


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("last", [False, True])
def test_embedding_bag_auxiliaries_padding_empty_and_tail(tmp_path, mode, last):
    w = torch.tensor([[float("nan"), 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    i = torch.tensor([0, 1, 2, 3])
    o = torch.tensor([0, 0, 2])
    assert_native_matches_eager(
        tmp_path, lambda a, b, c: torch.ops.aten._embedding_bag(a, b, c, True, mode, True, None, last, 1), (w, i, o)
    )


@pytest.mark.parametrize("dtype", [torch.int8, torch.int32, torch.int64, torch.bool])
@pytest.mark.parametrize("freq", [False, True])
def test_embedding_backward_integer_gradient(tmp_path, dtype, freq):
    i = torch.tensor([1, 1, 0])
    g = torch.tensor([[4, 7], [8, 5], [3, 1]], dtype=dtype)
    assert_native_matches_eager(
        tmp_path, lambda a, b: torch.ops.aten.embedding_dense_backward(a, b, 4, -1, freq), (g, i)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_mean_default_dtype_scalar_empty(tmp_path, dtype, shape):
    x = torch.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    assert_native_matches_eager(tmp_path, lambda a: torch.mean(a, dtype=dtype), (x,))


@pytest.mark.parametrize("op", ["sum", "amax", "min", "argmin", "argmax", "var"])
def test_scalar_reductions(tmp_path, op):
    x = torch.tensor(2.0)
    fn = {
        "sum": lambda a: torch.sum(a, dim=[0], keepdim=True),
        "amax": lambda a: torch.amax(a, dim=[-1], keepdim=True),
        "min": lambda a: torch.min(a, dim=0, keepdim=True),
        "argmin": lambda a: torch.argmin(a),
        "argmax": lambda a: torch.argmax(a),
        "var": lambda a: torch.var(a, dim=[0], unbiased=False, keepdim=True),
    }[op]
    assert_native_matches_eager(tmp_path, fn, (x,))


def test_as_strided_contiguous_nonzero_storage_offset(tmp_path):
    x = torch.arange(13, dtype=torch.float32)[1:].reshape(3, 4)
    assert_native_matches_eager(tmp_path, lambda a: torch.as_strided(a, [2, 2], [4, 1], a.storage_offset()), (x,))


def test_resize_preserves_input_storage_and_capture_provenance(tmp_path):
    class Resize(torch.nn.Module):
        def forward(self, x):
            return x.resize_([3, 2])

    x = torch.arange(7, dtype=torch.float32)[1:].reshape(2, 3)
    result = m2m.convert(Resize(), (x,), backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    assert not opaque_report(result.mlir_text)
    assert 'prov.aten = "aten.resize_.default"' in result.mlir_text
    assert x.shape == (2, 3)
    from m2m.ir.decompositions import decompose_view
    from test_core_aten_call_semantics import execute

    expected = x.clone().resize_([3, 2])
    (actual,) = execute(decompose_view, "aten.resize_.default", [x], (None, [3, 2]), expected)
    torch.testing.assert_close(actual, expected)


def test_inference_functional_batch_norm_returns_unmodified_statistics(tmp_path):
    x = torch.arange(32, dtype=torch.float32).reshape(2, 4, 2, 2)
    mean = torch.arange(4, dtype=torch.float32)
    var = mean + 1
    assert_native_matches_eager(
        tmp_path,
        lambda a, b, c: torch.ops.aten._native_batch_norm_legit_functional(a, None, None, b, c, False, 1.0, 0.001),
        (x, mean, var),
    )


@pytest.mark.parametrize("dtype", [torch.int8, torch.int32, torch.int64])
@pytest.mark.parametrize("groups", [1, 2])
def test_integer_convolution_with_bias_and_groups(tmp_path, dtype, groups):
    x = torch.arange(50, dtype=torch.int64).remainder(5).reshape(1, 2, 5, 5).to(dtype)
    w = torch.arange(4 * (2 // groups) * 9, dtype=torch.int64).remainder(3).reshape(4, 2 // groups, 3, 3).to(dtype)
    bias = torch.arange(4, dtype=dtype)
    assert_native_matches_eager(
        tmp_path,
        lambda a, b, c: torch.ops.aten.convolution(a, b, c, [1, 1], [1, 1], [1, 1], False, [0, 0], groups),
        (x, w, bias),
    )
