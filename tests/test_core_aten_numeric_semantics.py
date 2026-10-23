"""Independent public probes for storage rounding and absent arithmetic edges."""

import pytest
import torch
from m2m.ir.torchmlir_decomps import _batch_norm_no_stats_cpu, _log_softmax_cpu
from native_execution import assert_native_matches_eager, run_native

CLOSE = dict(atol=1e-6, rtol=1e-5)
BN_TRAINING = lambda x, w, b, m, v: torch.ops.aten._native_batch_norm_legit.default(x, w, b, m, v, True, 0.1, 1e-5)
BN_INFERENCE = lambda x, w, b, m, v: torch.ops.aten._native_batch_norm_legit_no_training.default(x, w, b, m, v, 0.1, 1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("operation", ["sigmoid", "mean", "var", "pool", "adaptive_pool", "addmm"])
def test_half_opmath_native(tmp_path, dtype, operation):
    x = torch.tensor([[-1.0, -0.5, 0.125], [0.5, 1.0, 1.5]], dtype=dtype)
    if operation == "sigmoid":
        fn, inputs = torch.sigmoid, (x,)
    elif operation == "mean":
        fn, inputs = lambda x: x.mean(1, keepdim=True), (x,)
    elif operation == "var":
        fn, inputs = lambda x: x.var(dim=1, unbiased=False), (x,)
    elif operation in ("pool", "adaptive_pool"):
        x = torch.tensor([60000.0, 60000.0, -60000.0, -60000.0], dtype=dtype).reshape(1, 1, 2, 2)
        fn = (
            (lambda x: torch.ops.aten.avg_pool2d.default(x, [2, 2]))
            if operation == "pool"
            else (lambda x: torch.ops.aten._adaptive_avg_pool2d.default(x, [1, 1]))
        )
        inputs = (x,)
    else:
        fn = torch.addmm
        inputs = (
            torch.full((2, 2), -60000.0, dtype=dtype),
            torch.full((2, 2), 200.0, dtype=dtype),
            torch.full((2, 2), 200.0, dtype=dtype),
        )
    assert_native_matches_eager(tmp_path, fn, inputs, **CLOSE)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("nearest", [True, False])
@pytest.mark.parametrize("scales", [False, True])
def test_resize_local_samples_and_scale(tmp_path, dtype, nearest, scales):
    x = torch.tensor([float("inf"), -1.0, 0.0, 1.0, 2.0, float("nan"), 3.0, 4.0, 5.0], dtype=dtype).reshape(1, 1, 3, 3)
    if nearest:
        fn = lambda x: torch.ops.aten.upsample_nearest2d.vec(
            x, None if scales else [5, 7], [1.7, 2.4] if scales else None
        )
    else:
        fn = lambda x: torch.ops.aten.upsample_bilinear2d.vec(
            x, None if scales else [5, 7], False, [1.7, 2.4] if scales else None
        )
    assert_native_matches_eager(tmp_path, fn, (x,), **CLOSE)


@pytest.mark.parametrize("dtype,values", [
    *((dtype, values) for dtype in (torch.float16, torch.bfloat16, torch.float32) for values in ("finite", "nonfinite")),
    # Column storage rounding.
    (torch.float16, "linspace"), (torch.bfloat16, "linspace")])
def test_convolution_backward_missing_edges_and_column_storage(tmp_path, dtype, values):
    if values == "linspace":
        grad = torch.linspace(-0.625, 0.875, 8).reshape(1, 2, 2, 2).to(dtype)
        source = torch.linspace(-0.875, 1.125, 25).reshape(1, 1, 5, 5).to(dtype)
        weight = torch.linspace(-0.875, 1.125, 18).reshape(2, 1, 3, 3).to(dtype)
    else:
        grad = torch.tensor([1.0, -0.5, 0.25, 0.5, 0.0, 1.0, 0.5, -1.0], dtype=dtype).reshape(1, 2, 2, 2)
        source = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5).to(dtype) / 8
        weight = torch.arange(18, dtype=torch.float32).reshape(2, 1, 3, 3).to(dtype) / 8
    if values == "nonfinite":
        weight[0, 0, 0, 0] = float("inf")
    fn = lambda g, x, w: torch.ops.aten.convolution_backward.default(
        g, x, w, [2], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]
    )
    assert_native_matches_eager(tmp_path, fn, (grad, source, weight), **CLOSE)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_log_softmax_scratch_rounding(tmp_path, dtype):
    x = torch.tensor([[-0.75, -0.35, 0.05], [0.45, 0.85, 1.25]], dtype=dtype)
    fn = lambda x: torch.ops.aten._log_softmax.default(x, 1, False)
    torch.testing.assert_close(_log_softmax_cpu(x, 1), fn(x), atol=0, rtol=0)
    assert_native_matches_eager(tmp_path, fn, (x,), **CLOSE)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("eps", [0.0, 1e-3])
def test_batch_norm_saved_storage_and_zero_eps(tmp_path, dtype, eps):
    x = (
        torch.full((2, 3, 2, 2), 0.5, dtype=dtype)
        if eps == 0
        else torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2).to(dtype) / 8
    )
    weight = torch.tensor([1.0, 0.5, -1.0], dtype=dtype)
    bias = torch.tensor([0.5, -1.0, 0.25], dtype=dtype)
    fn = lambda x, w, b: torch.ops.aten._native_batch_norm_legit.no_stats(x, w, b, True, 0.1, eps)
    for actual, expected in zip(_batch_norm_no_stats_cpu(x, weight, bias, True, 0.1, eps), fn(x, weight, bias)):
        torch.testing.assert_close(actual, expected, equal_nan=True, **CLOSE)
    assert_native_matches_eager(tmp_path, fn, (x, weight, bias), **CLOSE)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_irfft_structural_zero_bins(tmp_path, nonfinite):
    spectrum = torch.tensor([[1 + 3j, complex(nonfinite, 0), -1 + 2j]], dtype=torch.complex64)
    # Native helper's ABI uses real storage for the complex pair dimension.
    fn = lambda x: torch.ops.aten._fft_c2r.default(torch.view_as_complex(x), [1], 0, 4)
    assert_native_matches_eager(tmp_path, fn, (torch.view_as_real(spectrum),), **CLOSE)


def test_embedding_bag_auxiliary_values(tmp_path):
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4).t() / 8
    indices = torch.tensor([0, 2, 1, 3], dtype=torch.int64)
    offsets = torch.tensor([0, 2], dtype=torch.int64)
    fn = lambda w, i, o: torch.ops.aten._embedding_bag.default(w, i, o, False, 0, False, None, False, -1)
    assert_native_matches_eager(tmp_path, fn, (weight, indices, offsets), **CLOSE)


def test_layer_norm_backward_contiguous_statistics(tmp_path):
    x = torch.tensor([[-0.5, 0.0, 0.5], [0.5, 0.0, -0.5]])
    grad = torch.ones_like(x)
    mean = torch.zeros(2, 1)
    rstd = torch.ones(2, 1)
    weight, bias = torch.ones(3), torch.zeros(3)
    fn = lambda g, x, m, r, w, b: torch.ops.aten.native_layer_norm_backward.default(
        g, x, [3], m, r, w, b, [True, True, True]
    )
    assert_native_matches_eager(tmp_path, fn, (grad, x, mean, rstd, weight, bias), **CLOSE)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("padding", [0, 1, 2])
def test_grid_sample_rounding_and_masked_neighbors(tmp_path, dtype, mode, padding):
    image = torch.tensor([float("nan"), -1.0, 0.0, 1.0], dtype=dtype).reshape(1, 1, 2, 2)
    grid = torch.tensor([[[[-0.75, -0.25], [0.0, 1.0]], [[1.25, -1.25], [0.375, 0.625]]]], dtype=dtype)
    fn = lambda a, g: torch.ops.aten.grid_sampler_2d.default(a, g, mode, padding, True)
    assert_native_matches_eager(tmp_path, fn, (image, grid), **CLOSE)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("training", [True, False])
def test_batch_norm_affine_order(tmp_path, dtype, training):
    maximum = torch.finfo(dtype).max
    x = torch.tensor([-maximum, -1.0, 0.0, 1.0, maximum, -maximum, -1.0, 0.0], dtype=dtype).reshape(2, 1, 2, 2)
    weight, bias, mean, variance = [torch.tensor([value], dtype=dtype) for value in (1.0, -1.0, -maximum / 8, maximum)]
    fn = BN_TRAINING if training else BN_INFERENCE
    assert_native_matches_eager(tmp_path, fn, (x, weight, bias, mean, variance), **CLOSE)


def test_convolution_backward_grouped_dilated_geometry(tmp_path):
    source = torch.arange(120, dtype=torch.float32).reshape(1, 4, 5, 6) / 16
    weight = torch.arange(72, dtype=torch.float32).reshape(6, 2, 2, 3) / 32
    grad = torch.arange(72, dtype=torch.float32).reshape(1, 6, 3, 4) / 8
    fn = lambda g, x, w: torch.ops.aten.convolution_backward.default(
        g, x, w, [6], [2, 1], [1, 0], [2, 1], False, [0, 0], 2, [True, True, True]
    )
    assert_native_matches_eager(tmp_path, fn, (grad, source, weight), **CLOSE)


def test_batch_norm_running_mean_retains_unrounded_half_statistic(tmp_path):
    x = torch.tensor(
        [0.470458984375, 1.65625, 0.51513671875, -0.2744140625, -2.259765625, 1.2275390625, 0.79296875, 0.623046875],
        dtype=torch.float16,
    ).reshape(2, 1, 2, 2)
    mean, variance = torch.tensor([0.35], dtype=x.dtype), torch.ones(1, dtype=x.dtype)

    def fn(x, mean, variance):
        result = torch.ops.aten._native_batch_norm_legit.default(x, None, None, mean, variance, True, 0.1, 1e-5)
        return *result, mean, variance

    assert_native_matches_eager(tmp_path, fn, (x, mean, variance), **CLOSE)


def test_batch_norm_noncontiguous_variance_uses_cpu_accumulator(tmp_path):
    maximum = torch.finfo(torch.float32).max
    x = torch.tensor([-maximum, maximum, maximum, -maximum, -1.0, 1.0, 1.0, -1.0]).reshape(2, 1, 2, 2).transpose(2, 3)
    weight, bias = torch.ones(1), torch.zeros(1)
    mean, variance = torch.zeros(1), torch.ones(1)

    def fn(x, w, b, m, v):
        result = torch.ops.aten._native_batch_norm_legit.default(x, w, b, m, v, True, 0.0, 1e-5)
        return *result, m, v

    assert_native_matches_eager(tmp_path, fn, (x, weight, bias, mean, variance), **CLOSE)


@pytest.mark.parametrize("operation", ["sum", "mean", "amin", "amax"])
@pytest.mark.parametrize("keepdim", [True, False])
def test_empty_dimension_list_reduces_all_axes(tmp_path, operation, keepdim):
    x = torch.tensor([[-0.5, 1.0, 0.0], [2.0, -1.0, 0.5]])
    fn = lambda x: getattr(torch, operation)(x, dim=[], keepdim=keepdim)
    assert_native_matches_eager(tmp_path, fn, (x,), **CLOSE)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cpu_softmax_scalar_tail_storage_is_observable(dtype):
    x = torch.tensor([[-1.0, -0.5, 0.0, 0.25, 0.5], [0.0, 0.75, 1.5, 2.0, 2.5]], dtype=dtype).reshape(2, 1, 5)
    wide = x.float()
    exponentials = (wide - wide.amax(0, keepdim=True)).exp()
    denominator = exponentials.sum(0, keepdim=True)
    native = torch.ops.aten._softmax.default(x, 0, False)
    stored_tail = (exponentials.to(dtype).float() / denominator).to(dtype)
    torch.testing.assert_close(native, stored_tail, atol=0, rtol=0)
    assert not torch.equal(native, (exponentials / denominator).to(dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_group_norm_ordinary_statistics(tmp_path, dtype):
    x = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2).to(dtype) / 8
    fn = lambda x, w, b: torch.ops.aten.native_group_norm.default(x, w, b, 1, 4, 4, 2, 1e-3)
    assert_native_matches_eager(tmp_path, fn, (x, torch.ones(4, dtype=dtype), torch.zeros(4, dtype=dtype)), **CLOSE)


def test_layer_norm_ordinary_statistics(tmp_path):
    x = torch.tensor([[0.5, -0.5, 1.0, -1.0], [-0.25, 0.25, -0.5, 0.5]])
    fn = lambda x, w, b: torch.ops.aten.native_layer_norm.default(x, [4], w, b, 1e-3)
    assert_native_matches_eager(tmp_path, fn, (x, torch.ones(4), torch.zeros(4)), **CLOSE)


def test_convolution_ordinary_contraction(tmp_path):
    x = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5) / 8
    w = torch.arange(12, dtype=torch.float32).reshape(2, 1, 2, 3) / 8
    fn = lambda x, w: torch.ops.aten.convolution.default(x, w, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    assert_native_matches_eager(tmp_path, fn, (x, w), **CLOSE)


def test_rfft_absent_basis_products(tmp_path):
    x = torch.tensor([[1.0, float("nan"), -1.0, 2.0]])
    fn = lambda x: torch.view_as_real(torch.ops.aten._fft_r2c.default(x, [1], 0, True))
    assert_native_matches_eager(tmp_path, fn, (x,), **CLOSE)


@pytest.mark.parametrize("variant", ["training", "no_stats", "inference"])
def test_batch_norm_strided_affine_parameters(tmp_path, variant):
    x = torch.tensor([100000008.0, 0.0, 99999992.0, 1.0]).reshape(2, 2, 1, 1)
    weight = torch.tensor([0.3, 0.0, 1.0, 0.0])[::2]
    bias = torch.zeros(2)
    mean, variance = torch.tensor([1e8, 0.5]), torch.tensor([64.0, 0.25])
    if variant == "no_stats":
        fn = lambda x, w, b: torch.ops.aten._native_batch_norm_legit.no_stats(x, w, b, True, 0.1, 1e-5)
        inputs = x, weight, bias
        expected = fn(*inputs)
    else:
        fn = BN_TRAINING if variant == "training" else BN_INFERENCE
        inputs = x, weight, bias, mean, variance
        expected = fn(x, weight, bias, mean.clone(), variance.clone())
    # Eager must retain the weight's physical layout. Cloning this padded
    # vector would make it contiguous and select a different native formula.
    actual = run_native(tmp_path, fn, inputs)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(torch.from_numpy(got), want, equal_nan=True, **CLOSE)
