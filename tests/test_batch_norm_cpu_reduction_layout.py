"""CPU BatchNorm mean grouping across dense and strided statistics paths."""

import pytest
import torch
from native_execution import assert_native_matches_eager


def layout_tensor(value, layout):
    if layout == "channels_last":
        return value.contiguous(memory_format=torch.channels_last)
    if layout == "offset":
        base = torch.empty(value.numel() + 1, dtype=value.dtype)
        result = base[1:].reshape(value.shape)
        result.copy_(value)
        return result
    if layout == "padded":
        base = torch.zeros((*value.shape[:-1], value.shape[-1] * 2), dtype=value.dtype)
        result = base[..., ::2]
        result.copy_(value)
        return result
    if layout == "expanded":
        return value[:1].clone().expand(value.shape)
    return value


@pytest.mark.parametrize("operation", ["functional", "no_stats"])
@pytest.mark.parametrize("layout", ["dense", "channels_last", "offset", "padded", "expanded"])
@pytest.mark.parametrize("pattern", ["ordinary", "finite_extrema", "nonfinite"])
def test_cpu_batch_norm_layout_statistics_and_running_state(tmp_path, operation, layout, pattern):
    dtype = torch.float32
    shape = (2, 4, 2, 2)
    maximum = torch.finfo(dtype).max
    values = {
        "ordinary": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "finite_extrema": [-maximum, -1.0, 0.0, 1.0, maximum],
        "nonfinite": [float("nan"), -1.0, float("inf"), 1.0, -float("inf")],
    }[pattern]
    value = torch.tensor(values, dtype=dtype).repeat(8)[:32].reshape(shape)
    x = layout_tensor(value, layout)
    weight = torch.tensor([1.0, 0.5, -1.0, 0.0], dtype=dtype)
    bias = torch.tensor([0.0, 0.25, 0.0, -0.5], dtype=dtype)
    mean, variance = torch.zeros(4, dtype=dtype), torch.ones(4, dtype=dtype)

    def fn(a, w, b, mean, variance):
        if operation == "no_stats":
            return torch.ops.aten._native_batch_norm_legit.no_stats(a, w, b, True, 0.1, 1e-5)
        return torch.ops.aten._native_batch_norm_legit_functional.default(a, w, b, mean, variance, True, 0.1, 1e-5)

    assert_native_matches_eager(tmp_path, fn, (x, weight, bias, mean, variance), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", [(3, 4, 3), (1, 3, 2, 2, 3)])
@pytest.mark.parametrize("layout", ["dense", "padded", "expanded"])
def test_cpu_batch_norm_other_spatial_ranks(tmp_path, shape, layout):
    generator = torch.Generator().manual_seed(1049)
    x = layout_tensor(torch.randn(shape, generator=generator), layout)

    def fn(a):
        return torch.ops.aten._native_batch_norm_legit.no_stats(a, None, None, True, 0.1, 1e-5)

    assert_native_matches_eager(tmp_path, fn, (x,), rtol=1e-5, atol=1e-6)
