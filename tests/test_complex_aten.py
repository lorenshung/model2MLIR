"""Complex component ABI and IEEE edge semantics against eager CPU PyTorch."""
import numpy as np
import pytest
import torch
from xdsl.dialects.builtin import Float32Type, Float64Type
from xdsl.ir import Block

import m2m
from m2m.coverage import opaque_report
from m2m.ir.complex_aten import _Scalar
from complex_scalar_interpreter import evaluate


def _evaluate(build, components, scalar=Float64Type()):
    """Evaluate ``build(builder, args)`` on f32 component arrays cast to ``scalar``."""
    block = Block(arg_types=[Float32Type()] * len(components))
    builder = _Scalar(scalar)
    results = build(builder, [builder.cast(arg) for arg in block.args])
    return evaluate(builder.ops, block.args, components, results)


@pytest.mark.parametrize('op', [
    'abs', 'exp', 'expm1', 'log', 'log1p', 'log2', 'log10', 'sqrt',
    'rsqrt', 'reciprocal', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh',
    'asin', 'acos', 'atan', 'asinh', 'acosh', 'atanh', 'sigmoid',
])
def test_scalar_ieee_edges(op):
    values = np.array([
        -np.inf, -np.finfo(np.float32).max, -2, -1, -.75, -0., 0.,
        np.nextafter(np.float32(0), np.float32(1)), .25, 1, 2,
        np.finfo(np.float32).max, np.inf, np.nan,
    ], dtype=np.float32)
    x, y = np.meshgrid(values, values)
    z = np.empty(x.shape, np.complex64)
    z.real, z.imag = x, y
    reference = getattr(torch, op)(torch.from_numpy(z)).numpy()
    actual = _evaluate(lambda builder, args: builder.unary(op, args), [x, y],
                       Float32Type() if op in ('expm1', 'log1p', 'sigmoid') else Float64Type())
    with np.errstate(all='ignore'):
        np.testing.assert_allclose(np.asarray(actual[0], np.float32), reference.real,
                                   rtol=1e-5, atol=1e-6, equal_nan=True)
        if len(actual) == 2:
            np.testing.assert_allclose(np.asarray(actual[1], np.float32), reference.imag,
                                       rtol=1e-5, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize('shape', [(), (0,), (2, 1, 3)])
@pytest.mark.parametrize('fn', [
    lambda x: x.abs(), lambda x: x.exp(), lambda x: x.sqrt(),
    lambda x: x.sum(), lambda x: x.mean(), lambda x: x.prod(),
    lambda x: x + 2, lambda x: x * 2, lambda x: x / 2,
    lambda x: x == 2, lambda x: x != 2,
])
def test_capture_component_abi(shape, fn):
    class Model(torch.nn.Module):
        def forward(self, x): return fn(x)
    result = m2m.convert(Model(), (torch.ones(shape, dtype=torch.complex64),), backend='fx_importer')
    assert result.ok, result.diagnostics
    assert not opaque_report(result.mlir_text)
    result.module.verify()


def test_complex128_still_refuses_new_lowerings():
    class Model(torch.nn.Module):
        def forward(self, x): return x.sin()
    result = m2m.convert(Model(), (torch.ones(2, dtype=torch.complex128),), backend='fx_importer')
    assert opaque_report(result.mlir_text)


@pytest.mark.parametrize('op', ['asin', 'acos', 'asinh', 'acosh', 'atan', 'atanh'])
def test_near_branch_points(op):
    z = np.array([complex(x, y) for x in (-1., 0., 1.)
                  for y in (-1., -1e-8, -1e-15, 1e-15, 1e-8, 1.)], np.complex64)
    actual = _evaluate(lambda builder, args: builder.unary(op, args), [z.real, z.imag])
    reference = getattr(torch, op)(torch.from_numpy(z)).numpy()
    np.testing.assert_allclose(actual[0], reference.real, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(actual[1], reference.imag, rtol=1e-5, atol=1e-12)


def test_scaled_division_matches_c10_nonfinite_rules():
    z = np.array([0, complex(-0., -0.), 1, 1j, complex(np.inf, 0),
                  complex(0, np.inf), complex(np.inf, np.inf),
                  complex(np.nan, 0), complex(3.4028235e38, 3.4028235e38),
                  complex(1e-45, 0)], np.complex64)
    numerator, denominator = np.meshgrid(z, z)
    actual = _evaluate(lambda builder, args: builder.cdiv(args[:2], args[2:]),
                       [numerator.real, numerator.imag, denominator.real, denominator.imag])
    reference = (torch.from_numpy(numerator) / torch.from_numpy(denominator)).numpy()
    np.testing.assert_allclose(actual[0], reference.real, rtol=1e-5, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(actual[1], reference.imag, rtol=1e-5, atol=1e-6, equal_nan=True)


def test_as_strided_with_offset_uses_logical_storage_units():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.as_strided(x, (2, 2), (4, 1), 1)
    base = torch.arange(13, dtype=torch.float32).to(torch.complex64)
    value = base.as_strided((3, 4), (4, 1), 1)
    result = m2m.convert(Model(), (value,), backend='fx_importer')
    assert result.ok and not opaque_report(result.mlir_text)
    result.module.verify()


@pytest.mark.parametrize('dtype', [Float32Type(), Float64Type()])
def test_sign_transfer_preserves_signed_zero(dtype):
    values = np.array([0., -0., 1., -1., np.inf, -np.inf, np.nan],
                      dtype=np.float32 if isinstance(dtype, Float32Type) else np.float64)
    block = Block(arg_types=[dtype] * 2)
    builder = _Scalar(dtype)
    result = builder.sign(*block.args)
    actual = evaluate(builder.ops, block.args, [values, values[::-1]], [result])[0]
    expected = np.copysign(values, values[::-1])
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert all(op.name != 'math.copysign' for op in builder.ops)


def test_power_rounds_the_phase_on_the_branch_cut():
    base = np.array([complex(-1, -0.), complex(-1, 0.), -.75-.75j, 2+1j], np.complex64)
    exponent = np.array([2+1j, 2+1j, -.75-.75j, .5+0j], np.complex64)
    actual = _evaluate(lambda builder, args: builder.cpow(args[:2], args[2:]),
                       [base.real, base.imag, exponent.real, exponent.imag])
    reference = torch.pow(torch.from_numpy(base), torch.from_numpy(exponent)).numpy()
    np.testing.assert_allclose(actual[0], reference.real, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual[1], reference.imag, rtol=1e-5, atol=1e-6)
