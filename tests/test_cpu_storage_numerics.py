"""Execute storage semantics and portable arithmetic under the Core ATen comparison policy."""

import pytest
import torch
from native_execution import assert_native_matches_eager

CLOSE = dict(rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape,dim", [((2, 3), 0), ((1, 2, 3, 5), 2), ((2, 3, 5), -1)])
@pytest.mark.parametrize("special", [False, True])
def test_softmax_scratch_storage(tmp_path, dtype, shape, dim, special):
    x = torch.tensor([-1.0, 0.0, 1.0, 4.0, -2.0], dtype=dtype).repeat(20)[: torch.tensor(shape).prod()].reshape(shape)
    if special:
        x.flatten()[0] = float("nan")
    assert_native_matches_eager(tmp_path, lambda a: torch.ops.aten._softmax.default(a, dim, False), (x,), **CLOSE)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("special", [False, True])
def test_adaptive_pool3d_scalar_accumulator(tmp_path, dtype, special):
    x = torch.linspace(-0.75, 1.25, 64).to(dtype).reshape(1, 1, 4, 4, 4)
    if special:
        x.flatten()[:4] = torch.tensor([torch.finfo(dtype).max, torch.finfo(dtype).max, -1.0, 0.0], dtype=dtype)
    assert_native_matches_eager(
        tmp_path, lambda a: torch.ops.aten._adaptive_avg_pool3d.default(a, [2, 2, 2]), (x,), **CLOSE
    )


@pytest.mark.parametrize("value", [2.0, -2.0, 0.5, float("nan")])
def test_boolean_padding_truth_conversion(tmp_path, value):
    def fn(a):
        return torch.ops.aten.constant_pad_nd.default(a, [1, 2], value)

    assert_native_matches_eager(tmp_path, fn, (torch.tensor([[False, True]]),), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("special", [False, True])
def test_group_norm_portable_moments_across_layouts(tmp_path, dtype, channels_last, special):
    x = torch.linspace(-0.75, 1.25, 32).to(dtype).reshape(2, 4, 2, 2)
    if special:
        row = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)
        x = row.repeat(7)[:32].reshape(2, 4, 2, 2)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    w, b = torch.tensor([1.0, 0.5, 0.0, -1.0], dtype=dtype), torch.tensor([0.0, 1.0, -0.5, 0.25], dtype=dtype)

    def fn(a, w, b):
        return torch.ops.aten.native_group_norm.default(a, w, b, 2, 4, 4, 2, 1e-5)

    assert_native_matches_eager(tmp_path, fn, (x, w, b), **CLOSE)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("size", [3, 8, 17, 33])
def test_layer_norm_portable_moments(tmp_path, dtype, size):
    row = torch.tensor([2.0, -2.0, 1.0, -1.0, 0.0], dtype=dtype).repeat((size + 4) // 5)[:size]
    x = torch.stack([row, row.flip(0)])

    def fn(a):
        return torch.ops.aten.native_layer_norm.default(a, [size], None, None, 1e-5)

    assert_native_matches_eager(tmp_path, fn, (x,), **CLOSE)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size", [1, 3, 31, 32, 33, 64, 65])
def test_rsqrt_portable_opmath_across_sizes(tmp_path, dtype, size):
    x = torch.tensor([0.25, 1.0, 4.0], dtype=dtype).repeat((size + 2) // 3)[:size]
    assert_native_matches_eager(tmp_path, torch.rsqrt, (x,), **CLOSE)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), -float("inf")])
def test_nearest_grid_nonfinite_half_coordinates(tmp_path, dtype, coordinate):
    x = torch.tensor([5.0, -1.0, 0.0, 1.0], dtype=dtype).reshape(1, 1, 2, 2)
    g = torch.tensor([[[[coordinate, -1.0], [-1.0, coordinate]], [[coordinate, 1.0], [1.0, coordinate]]]], dtype=dtype)

    def fn(a, b):
        return torch.ops.aten.grid_sampler_2d.default(a, b, 1, 0, True)

    assert_native_matches_eager(tmp_path, fn, (x, g), **CLOSE)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_bfloat16_embedding_bag_portable_conversion(tmp_path, sign):
    w = torch.tensor([[0.125, 0.25], [-0.5, -0.125]], dtype=torch.bfloat16) * sign
    i, o = torch.tensor([0, 1]), torch.tensor([0])

    def fn(w, i, o):
        return torch.ops.aten._embedding_bag.default(w, i, o, False, 0, False, None, False, -1)

    assert_native_matches_eager(tmp_path, fn, (w, i, o), **CLOSE)


@pytest.mark.parametrize("hole", [0.0, 2.0])
def test_layer_norm_backward_reads_initialized_physical_statistics(tmp_path, hole):
    x = torch.tensor([[-0.75, -0.35, 0.05], [0.45, 0.85, 1.25]])
    m = torch.tensor([-0.35, hole, 0.85]).as_strided((2, 1), (2, 2))
    r = torch.tensor([3.06, hole, 3.06]).as_strided((2, 1), (2, 2))

    def fn(g, x, m, r):
        return torch.ops.aten.native_layer_norm_backward.default(g, x, [3], m, r, None, None, [True, False, False])[0]

    assert_native_matches_eager(tmp_path, fn, (torch.ones_like(x), x, m, r), **CLOSE)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("size", [3, 8, 10, 17, 31])
def test_mean_portable_reduction_across_sizes(tmp_path, dtype, size):
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype).repeat((size + 4) // 5)[:size]
    assert_native_matches_eager(tmp_path, torch.mean, (x,), **CLOSE)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_mean_larger_ordinary_reduction(tmp_path, dtype):
    x = torch.linspace(-.75, 1.25, 63).to(dtype)
    assert_native_matches_eager(tmp_path, torch.mean, (x,), **CLOSE)


def _initialized_statistics(stride, hidden):
    # Give the raw CPU row reads real backing cells, including expanded views.
    backing = torch.full((max(3, 2 * stride + 1),), hidden)
    statistic = backing.as_strided((3, 1), (stride, stride))
    for row in range(3):
        backing[row * stride] = 1.0
    return statistic


@pytest.mark.parametrize("stride", [0, 3])
def test_layer_norm_backward_hidden_statistic_cells_are_observable(stride):
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 8
    first = _initialized_statistics(stride, 0.0)
    second = _initialized_statistics(stride, 2.0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)

    def backward(statistic):
        return torch.ops.aten.native_layer_norm_backward.default(
            torch.ones_like(x), x, [4], statistic, statistic, torch.ones(4), torch.zeros(4), [True, True, True]
        )

    for index in (0, 1):
        assert not torch.allclose(backward(first)[index], backward(second)[index])
