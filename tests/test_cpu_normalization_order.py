"""CPU reductions match native bits across random rows, SIMD boundaries and layouts."""

import numpy as np
import pytest
import torch
from native_execution import padded, run_native


def assert_native_bits(tmp_path, fn, inputs):
    actual = run_native(tmp_path, fn, inputs, convert_options={"fully_standard": True})
    expected = torch.utils._pytree.tree_leaves(fn(*inputs))
    for got, want in zip(actual, expected, strict=True):
        value = torch.from_numpy(got)
        if want.dtype == torch.bfloat16:
            value = value.view(torch.bfloat16)
        missing = torch.isnan(want)
        assert torch.equal(torch.isnan(value), missing)
        storage = {torch.float16: torch.int16, torch.bfloat16: torch.int16,
                   torch.float32: torch.int32, torch.float64: torch.int64}[want.dtype]
        assert torch.equal(value.contiguous().view(storage)[~missing], want.contiguous().view(storage)[~missing])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("columns", [3, 8, 15, 16, 17, 128, 257, 768, 4096])
def test_layer_norm_native_random_moments_and_affine(tmp_path, dtype, columns):
    generator = torch.Generator().manual_seed(112)
    x = (torch.randn((9, columns), generator=generator) * 20 + 4).to(dtype)
    weight = torch.randn(columns, generator=generator).to(dtype)
    bias = torch.randn(columns, generator=generator).to(dtype)
    def fn(x, w, b):
        return torch.ops.aten.native_layer_norm.default(x, [columns], w, b, 0.001)
    assert_native_bits(tmp_path, fn, (x, weight, bias))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("nonfinite", [False, True])
def test_layer_norm_empty_simd_lanes_and_scalar_tail(tmp_path, dtype, nonfinite):
    limit = float("inf") if nonfinite else torch.finfo(dtype).max
    x = torch.tensor([[-limit, -1., 0.], [1., limit, -limit]], dtype=dtype)
    def fn(x):
        return torch.ops.aten.native_layer_norm.default(x, [3], None, None, 0.001)
    assert_native_bits(tmp_path, fn, (x,))


@pytest.mark.parametrize("dtype,shape,channels_last,seed", [
    *((dtype, shape, channels_last, 113) for dtype in (torch.float16, torch.bfloat16, torch.float32)
      for shape in [(2, 4, 2, 2), (2, 20, 3, 7), (1, 34, 2, 17)] for channels_last in (False, True)),
    # The large-spatial channels-last policy boundary.
    *((dtype, (1, 4, 1, spatial), True, 119) for spatial in (1023, 1024, 1025)
      for dtype in (torch.float32, torch.bfloat16))])
def test_group_norm_native_random_moments_affine_and_layout(tmp_path, dtype, shape, channels_last, seed):
    generator = torch.Generator().manual_seed(seed)
    x = (torch.randn(shape, generator=generator) * 7 + 10).to(dtype)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    w = torch.randn(shape[1], generator=generator).to(dtype)
    b = torch.randn(shape[1], generator=generator).to(dtype)
    def fn(x, w, b):
        return torch.ops.aten.native_group_norm.default(x, w, b, shape[0], shape[1],
                                                      shape[2] * shape[3], 2, 0.001)
    assert_native_bits(tmp_path, fn, (x, w, b))


def test_mean_bfloat16_cancellation_vector_tail(tmp_path):
    limit = torch.finfo(torch.bfloat16).max
    x = torch.tensor([-limit, -1., 0., 1., limit] * 2, dtype=torch.bfloat16).reshape(2, 1, 5)
    def fn(x): return x.mean([], True, dtype=torch.bfloat16)
    assert_native_bits(tmp_path, fn, (x,))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [[], [0], [2]])
@pytest.mark.parametrize("data,layout", [*(("random", layout) for layout in ("dense", "padded", "transpose")),
                                         *(("wide", layout) for layout in ("dense", "padded", "transpose", "expanded"))])
def test_mean_native_sum_traversal_wide_exponents_and_cascade_boundaries(tmp_path, dtype, dim, data, layout):
    if data == "random":
        x = (torch.randn((3, 4, 19), generator=torch.Generator().manual_seed(114)) * 10).to(dtype)
    else:
        generator = torch.Generator().manual_seed(120)
        shape = (33, 3, 67)
        low, high = (-20, 15) if dtype == torch.float16 else (-130, 127)
        significands = torch.randn(shape, generator=generator)
        x = torch.ldexp(significands, torch.randint(low, high, shape, generator=generator)).to(dtype)
    if layout == "padded":
        x = padded(x)
    elif layout == "transpose":
        x = x.transpose(0, 2)
    elif layout == "expanded":
        x = x[:1].expand(x.shape)
    def fn(x): return x.mean(dim, True, dtype=dtype)
    assert_native_bits(tmp_path, fn, (x,))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["dense", "channels_last", "padded", "expanded", "transpose"])
@pytest.mark.parametrize("shape", [(2, 4, 2, 2), (2, 17, 2, 17)])
def test_batch_norm_native_statistics_and_running_state(tmp_path, dtype, layout, shape):
    generator = torch.Generator().manual_seed(115)
    x = (torch.randn(shape, generator=generator) * 3 + 10).to(dtype)
    if layout == "channels_last":
        x = x.contiguous(memory_format=torch.channels_last)
    elif layout == "padded":
        x = padded(x)
    elif layout == "expanded":
        x = x[:1].expand(2, -1, -1, -1)
    elif layout == "transpose":
        x = x.transpose(-1, -2)
    w = torch.randn(shape[1], generator=generator).to(dtype)
    b = torch.randn(shape[1], generator=generator).to(dtype)
    mean, variance = torch.zeros(shape[1], dtype=dtype), torch.ones(shape[1], dtype=dtype)
    def fn(x, w, b, mean, variance):
        return torch.ops.aten._native_batch_norm_legit_functional.default(
            x, w, b, mean, variance, True, .1, .001)
    assert_native_bits(tmp_path, fn, (x, w, b, mean, variance))
