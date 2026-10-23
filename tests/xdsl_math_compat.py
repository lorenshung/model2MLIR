"""``MathFunctions`` for the xDSL interpreter, independent of the xDSL release.

xDSL added ``xdsl.interpreters.math`` after 0.65, the version ``uv.lock`` pins.
Tests that execute captured IR in the interpreter import ``MathFunctions`` from
here: the upstream implementation when it exists, otherwise this minimal
equivalent over Python floats (the same value domain the upstream interpreter
uses). Only scalar float/int ``math`` operations are covered. ``math.absf``,
``rsqrt``, ``log1p``, ``sin`` and ``cos`` are left to the tests' own extra
interpreter functions, as with the upstream module, so registrations never clash.
"""

from __future__ import annotations

try:  # pragma: no cover - depends on the installed xDSL
    from xdsl.interpreters.math import MathFunctions  # type: ignore[attr-defined]
except ModuleNotFoundError:  # xDSL <= 0.65
    import math as _m

    from xdsl.dialects import math as _math
    from xdsl.interpreter import _IMPL_DICT_KEY, Interpreter, InterpreterFunctions, PythonValues, impl, register_impls

    def _round_half_away(value: float) -> float:
        if not _m.isfinite(value):
            return value
        return _m.copysign(_m.floor(abs(value) + 0.5), value)

    def _erf(value: float) -> float:
        return _m.erf(value)

    def _safe(fn):
        def run(value: float) -> float:
            try:
                return fn(value)
            except (ValueError, OverflowError):
                if _m.isnan(value):
                    return value
                if fn in (_m.exp, _m.expm1, _m.exp2 if hasattr(_m, "exp2") else _m.exp, _m.cosh, _m.sinh):
                    return _m.inf if value > 0 or fn is _m.cosh else (-_m.inf if fn is _m.sinh else 0.0)
                if fn in (_m.log, _m.log2, _m.log10):
                    return -_m.inf if value == 0 else _m.nan
                if fn is _m.log1p:
                    return -_m.inf if value == -1 else _m.nan
                return _m.nan
        return run

    _UNARY = {
        _math.CeilOp: lambda v: float(_m.ceil(v)) if _m.isfinite(v) else v,
        _math.FloorOp: lambda v: float(_m.floor(v)) if _m.isfinite(v) else v,
        _math.TruncOp: lambda v: float(_m.trunc(v)) if _m.isfinite(v) else v,
        _math.RoundOp: _round_half_away,
        _math.RoundEvenOp: lambda v: float(round(v)) if _m.isfinite(v) else v,
        _math.SqrtOp: _safe(_m.sqrt),
        _math.ExpOp: _safe(_m.exp),
        _math.Exp2Op: _safe(lambda v: 2.0 ** v),
        _math.ExpM1Op: _safe(_m.expm1),
        _math.LogOp: _safe(_m.log),
        _math.Log2Op: _safe(_m.log2),
        _math.Log10Op: _safe(_m.log10),
        _math.TanhOp: _m.tanh,
        _math.TanOp: _safe(_m.tan),
        _math.SinhOp: _safe(_m.sinh),
        _math.CoshOp: _safe(_m.cosh),
        _math.AsinOp: _safe(_m.asin),
        _math.AcosOp: _safe(_m.acos),
        _math.AtanOp: _m.atan,
        _math.AsinhOp: _m.asinh,
        _math.AcoshOp: _safe(_m.acosh),
        _math.AtanhOp: _safe(_m.atanh),
        _math.CbrtOp: lambda v: _m.copysign(abs(v) ** (1.0 / 3.0), v),
        _math.ErfOp: _erf,
    }

    @register_impls
    class MathFunctions(InterpreterFunctions):  # type: ignore[no-redef]
        @impl(_math.AbsIOp)
        def run_absi(self, interpreter: Interpreter, op, args: PythonValues):
            return (abs(args[0]),)

        @impl(_math.PowFOp)
        def run_powf(self, interpreter: Interpreter, op, args: PythonValues):
            try:
                return (float(args[0]) ** float(args[1]),)
            except (ValueError, ZeroDivisionError, OverflowError):
                return (_m.nan if args[0] < 0 else _m.inf,)

        @impl(_math.FPowIOp)
        def run_fpowi(self, interpreter: Interpreter, op, args: PythonValues):
            try:
                return (float(args[0]) ** int(args[1]),)
            except ZeroDivisionError:
                return (_m.inf,)

        @impl(_math.FmaOp)
        def run_fma(self, interpreter: Interpreter, op, args: PythonValues):
            return (args[0] * args[1] + args[2],)

        @impl(_math.CopySignOp)
        def run_copysign(self, interpreter: Interpreter, op, args: PythonValues):
            return (_m.copysign(args[0], args[1]),)

        @impl(_math.Atan2Op)
        def run_atan2(self, interpreter: Interpreter, op, args: PythonValues):
            return (_m.atan2(args[0], args[1]),)

    def _unary_impl(fn):
        def run(self, interpreter: Interpreter, op, args: PythonValues):
            return (fn(args[0]),)
        return run

    # register_impls has already collected the decorated methods; add the
    # table-driven unary ops to the same registry.
    for _op_type, _fn in _UNARY.items():
        getattr(MathFunctions, _IMPL_DICT_KEY)[_op_type] = impl(_op_type)(_unary_impl(_fn))



def _static_slice(op, name):
    values = getattr(op, f"static_{name}").get_values()
    if any(v < 0 for v in values):
        raise NotImplementedError(f"dynamic {name} in {op.name}")
    return list(values)


def _slice_indices(op):
    from itertools import product as _product

    offsets, sizes, strides = (_static_slice(op, name) for name in ("offsets", "sizes", "strides"))
    for index in _product(*(range(size) for size in sizes)):
        yield index, [o + i * s for o, i, s in zip(offsets, index, strides)]


def _register_slice_functions():
    from xdsl.dialects import tensor as _tensor
    from xdsl.interpreter import InterpreterFunctions as _Functions, impl as _impl, register_impls as _register
    from xdsl.interpreters.shaped_array import ShapedArray as _Array
    from xdsl.interpreters.utils.ptr import TypedPtr as _Ptr

    @_register
    class SliceFunctions(_Functions):
        """Static extract_slice / insert_slice / concat for the xDSL interpreter."""

        @_impl(_tensor.ExtractSliceOp)
        def extract_slice(self, interpreter, op, args):
            source = args[0]
            sizes = _static_slice(op, "sizes")
            result_shape = list(op.result.type.get_shape())
            values = [source.load(where) for _, where in _slice_indices(op)]
            assert len(values) == max(1, __import__("math").prod(result_shape)) or not values
            out = _Array(_Ptr.new(values, xtype=source.element_type), result_shape or sizes)
            return (out,)

        @_impl(_tensor.InsertSliceOp)
        def insert_slice(self, interpreter, op, args):
            source, dest = args[0], args[1].copy()
            flat = source.data
            for position, (_, where) in enumerate(_slice_indices(op)):
                dest.store(where, flat[position])
            return (dest,)

        @_impl(_tensor.ConcatOp)
        def concat(self, interpreter, op, args):
            dim = op.dim.value.data
            shape = list(op.result.type.get_shape())
            result = _Array(_Ptr.new([args[0].data[0]] * __import__("math").prod(shape),
                                     xtype=args[0].element_type), shape)
            offset = 0
            for value in args:
                for index in value.indices():
                    where = list(index)
                    where[dim] += offset
                    result.store(where, value.load(index))
                offset += value.shape[dim]
            return (result,)

    return SliceFunctions


SliceFunctions = _register_slice_functions()

__all__ = ["MathFunctions", "SliceFunctions"]
