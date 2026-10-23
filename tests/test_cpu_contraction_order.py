"""Source CPU GEMM rounding, partial sums, alpha and bias initialization."""

import pytest
import torch

from m2m.ir.cpu_gemm import mkl_sse_small_kernel
from native_execution import padded
from test_cpu_normalization_order import assert_native_bits


@pytest.mark.parametrize("shape", [(3, 2, 4), (17, 3, 13), (9, 5, 32)])
@pytest.mark.parametrize("alpha,beta", [(.5, 1), (.7, .3), (-.7, 0), (0, 1)])
@pytest.mark.skipif(not mkl_sse_small_kernel(), reason="source CPU uses another BLAS kernel")
def test_small_addmm_native_alpha_and_alignment(tmp_path, shape, alpha, beta):
    m, n, k = shape
    g = torch.Generator().manual_seed(116)
    a = torch.randn((m, k), generator=g) * 2. ** torch.randint(-10, 20, (m, k), generator=g)
    b = torch.randn((k, n), generator=g) * 2. ** torch.randint(-10, 20, (k, n), generator=g)
    bias = torch.randn((m, n), generator=g)
    def fn(bias, a, b): return torch.addmm(bias, a, b, alpha=alpha, beta=beta)
    assert_native_bits(tmp_path, fn, (bias, a, b))


@pytest.mark.skipif(not mkl_sse_small_kernel(), reason="source CPU uses another BLAS kernel")
def test_addmm_alpha_prevents_intermediate_overflow(tmp_path):
    limit = torch.finfo(torch.float32).max
    a = torch.tensor([[limit, limit, 0., 0.], [-limit, -limit, 1., 0.]])
    b = torch.ones((4, 3))
    bias = torch.zeros((2, 3))
    def fn(bias, a, b): return torch.addmm(bias, a, b, alpha=.5)
    assert torch.isfinite(fn(bias, a, b)).all()
    assert_native_bits(tmp_path, fn, (bias, a, b))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [((1, 1, 5, 5), (2, 1, 3, 3)), ((1, 2, 5, 7), (3, 2, 3, 3))])
@pytest.mark.parametrize("extrema", [False, True])
@pytest.mark.skipif(not mkl_sse_small_kernel(), reason="source CPU uses another BLAS kernel")
def test_convolution_native_bias_and_accumulation(tmp_path, dtype, shape, extrema):
    g = torch.Generator().manual_seed(117)
    x = torch.randn(shape[0], generator=g).to(dtype)
    w = torch.randn(shape[1], generator=g).to(dtype)
    bias = torch.randn(shape[1][0], generator=g).to(dtype)
    if extrema:
        pattern = torch.tensor([-torch.finfo(dtype).max, -1., 0., 1., torch.finfo(dtype).max], dtype=dtype)
        for value in (x, w, bias):
            value.copy_(pattern.repeat((value.numel() + 4) // 5)[:value.numel()].reshape(value.shape))
    def fn(x, w, b):
        return torch.ops.aten.convolution.default(x, w, b, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    assert_native_bits(tmp_path, fn, (x, w, bias))


@pytest.mark.parametrize("shape", [((1, 1, 5, 5), (2, 1, 3, 3)), ((1, 2, 5, 7), (3, 2, 3, 3))])
@pytest.mark.parametrize("layout", ["dense", "padded", "offset"])
@pytest.mark.skipif(not mkl_sse_small_kernel(), reason="source CPU uses another BLAS kernel")
def test_convolution_weight_gradient_native_vectors_and_tails(tmp_path, shape, layout):
    g = torch.Generator().manual_seed(118)
    x = torch.randn(shape[0], generator=g) * 2. ** torch.randint(-10, 20, shape[0], generator=g)
    w = torch.randn(shape[1], generator=g)
    dy = torch.randn((shape[0][0], shape[1][0], shape[0][2] - 2, shape[0][3] - 2), generator=g)
    if layout == "padded":
        x = padded(x)
    elif layout == "offset":
        storage = torch.empty(dy.numel() + 1)
        storage[1:] = dy.flatten()
        dy = storage[1:].reshape(dy.shape)
    def fn(dy, x, w):
        return torch.ops.aten.convolution_backward.default(
            dy, x, w, [w.shape[0]], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])[1]
    assert_native_bits(tmp_path, fn, (dy, x, w))
