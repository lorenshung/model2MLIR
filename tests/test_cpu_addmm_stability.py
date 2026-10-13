"""Stable CPU addmm, checked against independent exact-product math.fsum."""

import ctypes
import math

import numpy as np
import pytest
import torch
from native_execution import assert_native_matches_eager, run_native


def mathematical_addmm(bias, a, b, *, alpha=1.0, beta=1.0):
    alpha, beta = ctypes.c_float(alpha).value, ctypes.c_float(beta).value
    output = torch.empty((a.shape[0], b.shape[1]), dtype=torch.float32)
    broadcast = bias.expand(output.shape)
    for row in range(output.shape[0]):
        for column in range(output.shape[1]):
            if alpha == 0 or a.shape[1] == 0:
                value = torch.tensor(0.0)
            else:
                terms = [float(a[row, k]) * float(b[k, column]) for k in range(a.shape[1])]
                try:
                    dot = math.fsum(terms)
                except ValueError:
                    dot = float("nan")
                value = torch.tensor(alpha * dot, dtype=torch.float32)
            if beta != 0:
                value = value + broadcast[row, column] * beta
            else:
                value = value + torch.tensor(0.0)
            output[row, column] = value
    return output


@pytest.mark.parametrize(
    "alpha,beta", [(0.0, 0.0), (0.0, 1.0), (0.0, -2.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0), (-2.0, 0.25), (0.1, -0.3)]
)
@pytest.mark.parametrize("bias_shape", [(), (2,), (2, 1), (1, 2), (2, 2)])
def test_finite_addmm_scalar_factors_and_broadcast(tmp_path, alpha, beta, bias_shape):
    torch.manual_seed(91)
    a, b = torch.randn(2, 7), torch.randn(7, 2)
    bias = torch.full(bias_shape, -0.25)

    def fn(bias, a, b):
        return torch.addmm(bias, a, b, alpha=alpha, beta=beta)

    (got,) = run_native(tmp_path, fn, (bias, a, b))
    expected = mathematical_addmm(bias, a, b, alpha=alpha, beta=beta)
    torch.testing.assert_close(torch.from_numpy(got), expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(expected, fn(bias, a, b), rtol=1e-5, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_other_addmm_dtypes_keep_existing_lowering(tmp_path, dtype):
    a = torch.linspace(-0.5, 1.0, 10, dtype=dtype).reshape(2, 5)
    b = torch.linspace(-1.0, 0.5, 15, dtype=dtype).reshape(5, 3)
    bias = torch.full((3,), 0.25, dtype=dtype)

    def fn(bias, a, b):
        return torch.addmm(bias, a, b, alpha=0.5, beta=1.5)

    assert_native_matches_eager(tmp_path, fn, (bias, a, b), rtol=1e-5 if dtype == torch.float64 else 0.02, atol=1e-6)


@pytest.mark.parametrize("operation", ["mm", "bmm", "convolution"])
def test_other_contractions_keep_existing_lowering(tmp_path, operation):
    if operation == "mm":
        inputs = (torch.arange(10, dtype=torch.float32).reshape(2, 5) / 8, torch.ones(5, 3))
        fn = torch.mm
    elif operation == "bmm":
        inputs = (torch.arange(20, dtype=torch.float32).reshape(2, 2, 5) / 8, torch.ones(2, 5, 3))
        fn = torch.bmm
    else:
        inputs = (torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5) / 8, torch.ones(2, 1, 3, 3))
        fn = torch.nn.functional.conv2d
    assert_native_matches_eager(tmp_path, fn, inputs)


@pytest.mark.parametrize("alpha", [1.0, 0.5, -2.0])
@pytest.mark.parametrize("terms", ["overflow_then_cancel", "lost_small_residual", "product_cancellation"])
def test_compensated_addmm_extreme_finite_terms(tmp_path, alpha, terms):
    maximum = torch.finfo(torch.float32).max
    a, b = {
        "overflow_then_cancel": (torch.tensor([[maximum, maximum, -maximum]]), torch.ones(3, 1)),
        "lost_small_residual": (torch.tensor([[maximum, 1.0, -maximum]]), torch.ones(3, 1)),
        "product_cancellation": (torch.tensor([[maximum, maximum, 1.0]]), torch.tensor([[maximum], [-maximum], [1.0]])),
    }[terms]
    bias = torch.tensor(0.0)

    def fn(bias, a, b):
        return torch.addmm(bias, a, b, alpha=alpha, beta=0)

    (got,) = run_native(tmp_path, fn, (bias, a, b))
    expected = mathematical_addmm(bias, a, b, alpha=alpha, beta=0)
    torch.testing.assert_close(torch.from_numpy(got), expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("alpha,beta", [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.5, -1.0)])
@pytest.mark.parametrize("values", ["nan", "infinity", "opposite_infinities", "zero_times_infinity"])
def test_addmm_nonfinite_and_ignored_operands(tmp_path, alpha, beta, values):
    a, b = {
        "nan": (torch.tensor([[float("nan"), 1.0]]), torch.ones(2, 1)),
        "infinity": (torch.tensor([[float("inf"), 1.0]]), torch.ones(2, 1)),
        "opposite_infinities": (torch.tensor([[float("inf"), -float("inf")]]), torch.ones(2, 1)),
        "zero_times_infinity": (torch.tensor([[0.0, 1.0]]), torch.tensor([[float("inf")], [1.0]])),
    }[values]
    bias = torch.tensor(float("nan")) if beta == 0 else torch.tensor(0.25)

    def fn(bias, a, b):
        return torch.addmm(bias, a, b, alpha=alpha, beta=beta)

    (got,) = run_native(tmp_path, fn, (bias, a, b))
    expected = mathematical_addmm(bias, a, b, alpha=alpha, beta=beta)
    torch.testing.assert_close(torch.from_numpy(got), expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(expected, fn(bias, a, b), rtol=1e-5, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize(
    "alpha,beta,bias_value", [(0.0, 1.0, -0.0), (0.0, -1.0, 0.0), (-1.0, 0.0, -0.0), (1.0, 0.0, -0.0)]
)
def test_addmm_signed_zero_and_zero_factors(tmp_path, alpha, beta, bias_value):
    bias, a, b = torch.tensor(bias_value), torch.tensor([[-0.0, 0.0]]), torch.ones(2, 1)

    def fn(bias, a, b):
        return torch.addmm(bias, a, b, alpha=alpha, beta=beta)

    (got,) = run_native(tmp_path, fn, (bias, a, b))
    expected = fn(bias, a, b).numpy()
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(np.signbit(got), np.signbit(expected))


@pytest.mark.parametrize("alpha,beta", [(0.0, 0.0), (-1.0, 0.0), (0.5, 1.0), (1.0, -2.0)])
def test_empty_addmm_ignores_alpha(tmp_path, alpha, beta):
    bias, a, b = torch.tensor(-0.0), torch.empty(2, 0), torch.empty(0, 2)

    def fn(bias, a, b):
        return torch.addmm(bias, a, b, alpha=alpha, beta=beta)

    assert_native_matches_eager(tmp_path, fn, (bias, a, b), rtol=0, atol=0)
