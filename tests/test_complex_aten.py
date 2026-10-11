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
    block = Block(arg_types=[Float32Type()] * 2)
    builder = _Scalar(Float32Type() if op in ('expm1', 'log1p', 'sigmoid') else Float64Type())
    results = builder.unary(op, [builder.cast(arg) for arg in block.args])
    actual = evaluate(builder.ops, block.args, [x, y], results)
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
    block = Block(arg_types=[Float32Type()] * 2)
    builder = _Scalar(Float64Type())
    results = builder.unary(op, [builder.cast(arg) for arg in block.args])
    actual = evaluate(builder.ops, block.args, [z.real, z.imag], results)
    reference = getattr(torch, op)(torch.from_numpy(z)).numpy()
    np.testing.assert_allclose(actual[0], reference.real, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(actual[1], reference.imag, rtol=1e-5, atol=1e-12)


def test_scaled_division_matches_c10_nonfinite_rules():
    z = np.array([0, complex(-0., -0.), 1, 1j, complex(np.inf, 0),
                  complex(0, np.inf), complex(np.inf, np.inf),
                  complex(np.nan, 0), complex(3.4028235e38, 3.4028235e38),
                  complex(1e-45, 0)], np.complex64)
    numerator, denominator = np.meshgrid(z, z)
    block = Block(arg_types=[Float32Type()] * 4)
    builder = _Scalar(Float64Type())
    args = [builder.cast(arg) for arg in block.args]
    results = builder.cdiv(args[:2], args[2:])
    actual = evaluate(builder.ops, block.args,
                      [numerator.real, numerator.imag, denominator.real, denominator.imag], results)
    reference = (torch.from_numpy(numerator) / torch.from_numpy(denominator)).numpy()
    np.testing.assert_allclose(actual[0], reference.real, rtol=1e-5, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(actual[1], reference.imag, rtol=1e-5, atol=1e-6, equal_nan=True)
