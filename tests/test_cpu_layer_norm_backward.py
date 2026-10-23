"""Native LayerNorm dgamma preserves affine FMAs and row-buffer rounding."""

import itertools

import pytest
import torch

from test_cpu_normalization_order import assert_native_bits


DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def dgamma(g, x, m, r, w):
    return torch.ops.aten.native_layer_norm_backward.default(
        g, x, list(x.shape[1:]), m, r, w, None, [False, True, False]
    )[1]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_layer_norm_backward_infinite_affine_all_outputs(tmp_path, dtype, sign):
    # Row zero's affine shift is -inf. For negative rstd, x=-1 cancels an
    # infinite product; centering first incorrectly produces an infinity.
    x = torch.tensor([float("inf") * sign, -1.0, 0.0], dtype=dtype).expand(2, 3)
    mean = torch.tensor([[float("inf") * sign], [0.0]], dtype=dtype)
    rstd = mean.clone()
    weight = torch.full((1,), float("inf") * sign, dtype=dtype).expand(3)

    def fn(g, x, m, r, w):
        return torch.ops.aten.native_layer_norm_backward.default(
            g, x, [3], m, r, w, w, [True, True, True]
        )

    middle = fn(x, x, mean, rstd, weight)[1][1]
    assert torch.isnan(middle) if sign < 0 else torch.isposinf(middle)
    assert_native_bits(tmp_path, fn, (x, x, mean, rstd, weight))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mean,rstd", [
    (float("inf"), float("inf")), (-float("inf"), -float("inf")),
    (float("inf"), 0.0), (0.0, float("inf")),
    (-0.0, -float("inf")), (float("nan"), 1.0),
    (1.0, float("nan")), (0.0, -0.0),
])
def test_layer_norm_backward_nonfinite_affine_cross_product(tmp_path, dtype, mean, rstd):
    # Every x/dy combination remains a separate dgamma element; a sum over
    # different scenarios would hide wrong infinities behind unrelated NaNs.
    values = [-float("inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan")]
    pairs = list(itertools.product(values, repeat=2))
    x, grad = torch.tensor(pairs, dtype=dtype).T.reshape(2, 1, -1)
    m, r = torch.tensor([[mean]], dtype=dtype), torch.tensor([[rstd]], dtype=dtype)
    w = torch.ones(x.shape[-1], dtype=dtype)
    assert_native_bits(tmp_path, dgamma, (grad, x, m, r, w))


@pytest.mark.parametrize("dtype,parameter_dtype,value,scale", [
    (torch.float32, torch.float32, 2.0**100, 2.0**100),
    (torch.float64, torch.float64, 2.0**700, 2.0**700),
    (torch.bfloat16, torch.bfloat16, 2.0**100, 2.0**100),
    (torch.float16, torch.float32, 60000.0, 2.0**120),
])
def test_layer_norm_backward_finite_inputs_overflow_affine_shift(tmp_path, dtype, parameter_dtype, value, scale):
    # The separately rounded bias overflows. FMA(a, x, -inf) stays -inf,
    # whereas centering the equal finite x and mean would produce zero.
    x = torch.full((1, 17), value, dtype=dtype)
    m = torch.tensor([[value]], dtype=parameter_dtype)
    r = torch.tensor([[scale]], dtype=parameter_dtype)
    w = torch.ones(17, dtype=parameter_dtype)
    g = torch.ones_like(x)
    assert torch.isneginf(dgamma(g, x, m, r, w)).all()
    assert_native_bits(tmp_path, dgamma, (g, x, m, r, w))


@pytest.mark.parametrize("dtype,mixed", [(t, False) for t in DTYPES] + [
    (torch.float16, True), (torch.bfloat16, True),
])
@pytest.mark.parametrize("layout", ["dense", "transpose", "padded", "expanded"])
def test_layer_norm_backward_random_affine_and_row_rounding(tmp_path, dtype, mixed, layout):
    rng = torch.Generator().manual_seed(121 + ["dense", "transpose", "padded", "expanded"].index(layout))
    # Independent random columns cover full vectors, scalar tails, multiple
    # normalized dimensions and enough rows to expose buffer-rounding errors.
    shape = (17, 4, 19)
    x = (torch.randn(shape, generator=rng, dtype=torch.float64) * 13).to(dtype)
    grad = (torch.randn(shape, generator=rng, dtype=torch.float64) * 7).to(dtype)
    if layout == "transpose":
        x, grad = x.transpose(1, 2), grad.transpose(1, 2)
    elif layout == "padded":
        def pad(t):
            backing = torch.zeros((*t.shape[:-1], t.shape[-1] * 2), dtype=t.dtype)
            backing[..., ::2] = t
            return backing[..., ::2]
        x, grad = pad(x), pad(grad)
    elif layout == "expanded":
        x, grad = x[:1].expand(shape), grad[:1].expand(shape)
    parameter_dtype = torch.float32 if mixed else dtype
    m = torch.randn((17, 1, 1), generator=rng, dtype=torch.float64).to(parameter_dtype)
    r = torch.randn((17, 1, 1), generator=rng, dtype=torch.float64).to(parameter_dtype)
    w = torch.ones(x.shape[1:], dtype=parameter_dtype)
    assert_native_bits(tmp_path, dgamma, (grad, x, m, r, w))


@pytest.mark.parametrize("dtype", DTYPES)
def test_layer_norm_backward_physical_statistics_and_bias_slot(tmp_path, dtype):
    rng = torch.Generator().manual_seed(125)
    x = torch.randn((5, 17), generator=rng).to(dtype)
    grad = torch.randint(-8, 8, x.shape, generator=rng).to(dtype) / 8
    # Saved statistics are raw pointer reads, including the initialized cells
    # between logical rows. The dgamma correction must retain that convention.
    m = torch.randn(10, generator=rng).to(dtype)[::2].reshape(5, 1)
    r = torch.randn(10, generator=rng).to(dtype)[::2].reshape(5, 1)
    w = torch.ones(17, dtype=dtype)

    def fn(g, x, m, r, w):
        out = torch.ops.aten.native_layer_norm_backward.default(
            g, x, [17], m, r, w, w, output_mask=[False, True, True]
        )
        return out[1], out[2]

    assert_native_bits(tmp_path, fn, (grad, x, m, r, w))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(0, 3), (2, 0)])
def test_layer_norm_backward_empty_parameter_gradient(tmp_path, dtype, shape):
    x, w = torch.empty(shape, dtype=dtype), torch.ones(shape[1], dtype=dtype)
    m, r = torch.zeros((shape[0], 1), dtype=dtype), torch.ones((shape[0], 1), dtype=dtype)
    assert_native_bits(tmp_path, lambda x, m, r, w: dgamma(x, x, m, r, w), (x, m, r, w))
