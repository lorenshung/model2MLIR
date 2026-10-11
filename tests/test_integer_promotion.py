"""Promotion and exact integral arithmetic, including runtime overflow edges."""
import pytest
import torch

import m2m
from tests.native_execution import assert_native_matches_eager


class Call(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def convert(fn, *inputs):
    result = m2m.convert(Call(fn), inputs, backend='fx_importer')
    assert result.ok, result.diagnostics
    assert 'func.call' not in result.mlir_text, result.diagnostics
    result.module.verify()
    return result.mlir_text


DTYPES = [torch.int8, torch.int32, torch.int64, torch.bool]
FLOAT_UNARY = ['acos', 'asin', 'atan', 'acosh', 'asinh', 'atanh', 'sin', 'cos',
               'tan', 'sinh', 'cosh', 'tanh', 'exp', 'expm1', 'log', 'log1p',
               'log2', 'log10', 'sqrt', 'rsqrt', 'sigmoid', 'erf']


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('name', FLOAT_UNARY)
def test_floating_unary_promotes_integer_and_bool(name, dtype):
    x = torch.tensor([0, 1], dtype=dtype)
    fn = getattr(torch.ops.aten, name).default
    text = convert(fn, x)
    assert 'tensor<2xf32>' in text
    assert ('arith.uitofp' if dtype == torch.bool else 'arith.sitofp') in text
    assert f'prov.aten = "aten.{name}.default"' in text


@pytest.mark.parametrize('dtype', DTYPES)
def test_default_float_is_capture_time_dtype(dtype):
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        text = convert(torch.sin, torch.tensor([0, 1], dtype=dtype))
        assert 'tensor<2xf64>' in text
    finally:
        torch.set_default_dtype(old)


@pytest.mark.parametrize('dtype', DTYPES[:-1])
@pytest.mark.parametrize('name', ['neg', 'floor', 'ceil', 'round', 'trunc'])
def test_integral_unary_exact_at_extrema(tmp_path, dtype, name):
    limits = torch.iinfo(dtype)
    x = torch.tensor([limits.min, limits.min + 1, -1, 0, 1, limits.max], dtype=dtype)
    fn = getattr(torch.ops.aten, name).default
    text = convert(fn, x)
    assert 'tofp' not in text
    assert_native_matches_eager(tmp_path, fn, (x,), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', DTYPES[:-1])
def test_integer_tensor_pow_negative_exponents_and_wrap(tmp_path, dtype):
    limits = torch.iinfo(dtype)
    x = torch.tensor([-2, -1, -1, 0, 1, 2, 3, limits.max, limits.min], dtype=dtype)
    y = torch.tensor([-3, limits.min, -3, -1, -3, 7, 10, 2, 3], dtype=dtype)
    text = convert(torch.pow, x, y)
    assert 'math.powf' not in text
    assert 'scf.while' in text
    assert_native_matches_eager(tmp_path, torch.pow, (x, y), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('base', [2, 2.5])
def test_scalar_base_pow_uses_result_type(tmp_path, dtype, base):
    x = torch.tensor([0, 1] if dtype == torch.bool else [-3, -1, 0, 1, 9], dtype=dtype)
    fn = lambda x: torch.ops.aten.pow.Scalar(base, x)
    assert fn(x).dtype == torch.result_type(base, x)
    convert(fn, x)
    assert_native_matches_eager(tmp_path, fn, (x,))


@pytest.mark.parametrize('exponent', [0, 2, 7, 2.5])
def test_scalar_exponent_pow(tmp_path, exponent):
    x = torch.tensor([-128, -2, -1, 0, 1, 3, 127], dtype=torch.int8)
    fn = lambda x: torch.ops.aten.pow.Tensor_Scalar(x, exponent)
    convert(fn, x)
    assert_native_matches_eager(tmp_path, fn, (x,), rtol=1e-5, atol=1e-6)


def test_negative_integer_scalar_exponent_stays_eager_invalid():
    x = torch.tensor([1, 2], dtype=torch.int64)
    with pytest.raises(RuntimeError, match='negative integer powers'):
        torch.pow(x, -1)
    result = m2m.convert(Call(lambda x: torch.pow(x, -1)), (x,), backend='fx_importer')
    assert not result.ok or 'func.call' in result.mlir_text


@pytest.mark.parametrize('other', [torch.tensor([2, 3], dtype=torch.int64),
                                   torch.tensor(2, dtype=torch.int64),
                                   torch.tensor([2., 3.], dtype=torch.float64)])
def test_pow_tensor_rank_and_dtype_promotion(tmp_path, other):
    x = torch.tensor([3, 5], dtype=torch.int8)
    assert torch.pow(x, other).dtype == torch.result_type(x, other)
    convert(torch.pow, x, other)
    assert_native_matches_eager(tmp_path, torch.pow, (x, other), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('dim', [False, True])
def test_mean_explicit_floating_dtype(tmp_path, dtype, dim):
    x = torch.tensor([[0, 1, 1], [1, 1, 0]], dtype=dtype)
    fn = (lambda x: torch.ops.aten.mean.dim(x, [1], True, dtype=torch.float64)) if dim else (
        lambda x: torch.ops.aten.mean.default(x, dtype=torch.float64))
    convert(fn, x)
    assert_native_matches_eager(tmp_path, fn, (x,))


@pytest.mark.parametrize('alpha', [0, 1, 2, -2])
def test_boolean_add_uses_truth_and_or(tmp_path, alpha):
    x, y = torch.tensor([False, True, True]), torch.tensor([True, False, True])
    fn = lambda x, y: torch.ops.aten.add.Tensor(x, y, alpha=alpha)
    assert_native_matches_eager(tmp_path, fn, (x, y), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', DTYPES[:-1])
@pytest.mark.parametrize('mode', [None, 'floor', 'trunc'])
def test_division_mixed_integer_widths(tmp_path, dtype, mode):
    x = torch.tensor([-127, -7, -1, 0, 1, 7, 127], dtype=dtype)
    y = torch.tensor([2, -2, 3, -3, -2, 2, 1], dtype=torch.int64)
    fn = lambda x, y: torch.ops.aten.div.Tensor_mode(x, y, rounding_mode=mode)
    assert_native_matches_eager(tmp_path, fn, (x, y), rtol=0, atol=0)


@pytest.mark.parametrize('mode', [None, 'floor', 'trunc'])
@pytest.mark.parametrize('scalar', [True, 0., .1, -.1, 2.])
def test_division_integer_with_promoting_scalar(tmp_path, mode, scalar):
    x = torch.tensor([-3, -1, 0, 1, 3], dtype=torch.int64)
    fn = lambda x: torch.ops.aten.div.Scalar_mode(x, scalar, rounding_mode=mode)
    assert_native_matches_eager(tmp_path, fn, (x,), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', DTYPES[:-1])
def test_floor_division_wraps_minimum(tmp_path, dtype):
    x = torch.tensor([torch.iinfo(dtype).min, -7, 7], dtype=dtype)
    fn = lambda x: torch.ops.aten.div.Scalar_mode(x, -1, rounding_mode='floor')
    assert_native_matches_eager(tmp_path, fn, (x,), rtol=0, atol=0)


@pytest.mark.parametrize('shape,dims', [((0, 3), None), ((2, 3), []), ((), [])])
def test_integer_mean_empty_and_scalar(tmp_path, shape, dims):
    x = torch.ones(shape, dtype=torch.int32)
    fn = lambda x: torch.ops.aten.mean.dim(x, dims, False, dtype=torch.float32)
    assert_native_matches_eager(tmp_path, fn, (x,))


@pytest.mark.parametrize('name', ['sum', 'prod', 'amin', 'amax'])
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('scalar', [False, True])
def test_integer_reductions_use_exact_result_dtype(tmp_path, name, dtype, scalar):
    x = torch.tensor(1 if scalar else [[0, 1, 1], [1, 1, 0]], dtype=dtype)
    if name in ('sum', 'prod'):
        fn = getattr(torch.ops.aten, name).default
    else:
        fn = lambda x: getattr(torch.ops.aten, name).default(x, [], False)
    assert_native_matches_eager(tmp_path, fn, (x,), rtol=0, atol=0)
