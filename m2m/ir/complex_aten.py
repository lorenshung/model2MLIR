"""Core ATen complex64 lowerings for the interleaved real-component ABI.

Logical tensor dimensions never include the component axis. Complex arithmetic
is expressed with scalar arith/math inside linalg, so consumers do not need a
complex dialect conversion pass. No tensor data is read during lowering.
"""
from __future__ import annotations

import math as pymath
from types import SimpleNamespace

import torch
from xdsl.dialects import arith, math, tensor
from xdsl.dialects.builtin import (
    AffineMapAttr, FloatAttr, Float32Type, Float64Type, IndexType,
    IntegerAttr, IntegerType, StringAttr, TensorType, i1, i64,
)
from xdsl.dialects.linalg import GenericOp, IndexOp, IteratorType, IteratorTypeAttr, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineExpr, AffineMap

from m2m.ir.builder import Builder
from m2m.ir.structural import generate
from m2m.ir.decompositions import (
    DECOMPOSITION_TABLE, DecompResult, _broadcast_map, _cast_scalar_arg,
    _elementwise, _fx_arg, _next_region_id, _splat_scalar,
)


def _complex(value):
    return getattr(value, 'dtype', None) == torch.complex64


def _split(source):
    """Project components without reshape (including scalars and zero extents)."""
    shape = list(source.type.get_shape())[:-1]
    rank = len(shape)
    ops, planes = [], []
    for component in range(2):
        mapping = AffineMap(rank, 0, tuple(
            [AffineExpr.dimension(i) for i in range(rank)] + [AffineExpr.constant(component)]))
        emitted = _elementwise([source], TensorType(Float32Type(), shape),
                               lambda args, elem: ([], args[0]), input_maps=[mapping])
        ops += emitted[0]
        planes.append(emitted[1])
    return ops, planes


def _pack(real, imag):
    shape = list(real.type.get_shape())
    rank = len(shape) + 1
    mapping = AffineMap(rank, 0, tuple(AffineExpr.dimension(i) for i in range(rank - 1)))

    def body(args, elem):
        index = IndexOp(rank - 1)
        zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
        is_real = arith.CmpiOp(index.result, zero.result, 'eq')
        selected = arith.SelectOp(is_real.result, args[0], args[1])
        return [index, zero, is_real, selected], selected.result

    return _elementwise([real, imag], TensorType(real.type.element_type, [*shape, 2]),
                        body, input_maps=[mapping, mapping])


class _Scalar(Builder):
    """Scalar float IR; operations carry no fast-math assumptions."""
    def __init__(self, elem=Float32Type()):
        super().__init__(elem=elem)

    def c(self, value):
        if (type(value) is int and isinstance(self.elem, Float32Type)
                and abs(value) % (1 << max(0, abs(value).bit_length() - 24))):
            if not -(1 << 63) <= value < (1 << 64):
                raise ValueError('complex CPU coefficient requires a 64-bit Scalar')
            unsigned = value >= (1 << 63)
            integer = self.put(arith.ConstantOp(IntegerAttr(value - (1 << 64) if unsigned else value, i64)))
            return self.put((arith.UIToFPOp if unsigned else arith.SIToFPOp)(integer, self.elem))
        return self.put(arith.ConstantOp(FloatAttr(float(value), self.elem)))

    def cast(self, value, elem=None):
        ops, result = _cast_scalar_arg(value, elem or self.elem)
        self.ops += ops
        return result

    def neg(self, a): return self.put(arith.NegfOp(a))
    def abs(self, a): return self.put(math.AbsFOp(a))
    def sqrt(self, a): return self.put(math.SqrtOp(a))
    def log(self, a): return self.put(math.LogOp(a))
    def exp(self, a): return self.put(math.ExpOp(a))
    def sin(self, a): return self.put(math.SinOp(a))
    def cos(self, a): return self.put(math.CosOp(a))
    def sign(self, a, b):
        # Merlin's math lowering does not currently legalize math.copysign.
        # Express the IEEE sign-bit transfer with standard arith instead.
        width = 32 if isinstance(a.type, Float32Type) else 64
        integer = IntegerType(width)
        magnitude = self.put(arith.BitcastOp(a, integer))
        sign = self.put(arith.BitcastOp(b, integer))
        magnitude_mask = self.put(arith.ConstantOp(IntegerAttr((1 << (width - 1)) - 1, integer)))
        sign_mask = self.put(arith.ConstantOp(IntegerAttr(-(1 << (width - 1)), integer)))
        magnitude = self.put(arith.AndIOp(magnitude, magnitude_mask))
        sign = self.put(arith.AndIOp(sign, sign_mask))
        bits = self.put(arith.OrIOp(magnitude, sign))
        return self.put(arith.BitcastOp(bits, a.type))
    def cmp(self, a, b, pred='oeq'): return self.put(arith.CmpfOp(a, b, pred))
    def select(self, pred, yes, no): return self.put(arith.SelectOp(pred, yes, no))
    def both(self, a, b): return self.put(arith.AndIOp(a, b))
    def either(self, a, b): return self.put(arith.OrIOp(a, b))
    def isnan(self, a): return self.cmp(a, a, 'uno')
    def isinf(self, a): return self.cmp(self.abs(a), self.c(float('inf')))

    def hypot(self, a, b):
        # Inputs are widened to f64 for range-sensitive operations. All finite
        # f32 squares fit, even at the subnormal and maximum finite endpoints.
        h = self.sqrt(self.add(self.mul(a, a), self.mul(b, b)))
        return self.select(self.either(self.isinf(a), self.isinf(b)), self.c(float('inf')), h)

    def cmul(self, a, b):
        x, y = a; u, v = b
        return self.sub(self.mul(x, u), self.mul(y, v)), self.add(self.mul(x, v), self.mul(y, u))

    def cmul_f32(self, a, b):
        s = _Scalar(Float32Type())
        values = [s.cast(value) for value in (*a, *b)]
        re, im = s.cmul(values[:2], values[2:])
        self.ops += s.ops
        return self.cast(re), self.cast(im)

    def cdiv(self, a, b):
        # ATen CPU uses c10::complex's NumPy-style scaled f32 division.
        # Preserve its intermediate rounding, including 0*inf -> NaN.
        s = _Scalar(Float32Type())
        x, y, u, v = (s.cast(value) for value in (*a, *b))
        au, av = s.abs(u), s.abs(v)
        first = s.cmp(au, av, 'oge')
        ratio_u, ratio_v = s.div(v, u), s.div(u, v)
        scale_u = s.div(s.c(1), s.add(u, s.mul(v, ratio_u)))
        scale_v = s.div(s.c(1), s.add(v, s.mul(u, ratio_v)))
        re = s.select(first, s.mul(s.add(x, s.mul(y, ratio_u)), scale_u),
                      s.mul(s.add(s.mul(x, ratio_v), y), scale_v))
        im = s.select(first, s.mul(s.sub(y, s.mul(x, ratio_u)), scale_u),
                      s.mul(s.sub(s.mul(y, ratio_v), x), scale_v))
        zero = s.both(s.cmp(au, s.c(0)), s.cmp(av, s.c(0)))
        re = s.select(zero, s.div(x, au), re)
        im = s.select(zero, s.div(y, av), im)
        self.ops += s.ops
        return self.cast(re), self.cast(im)

    def clog(self, z):
        x, y = z
        if isinstance(self.elem, Float32Type):
            wide = _Scalar(Float64Type())
            re, im = wide.clog((wide.cast(x), wide.cast(y)))
            self.ops += wide.ops
            return self.cast(re), self.cast(im)
        return self.log(self.hypot(x, y)), self.put(math.Atan2Op(y, x))

    def cpow(self, base, exponent):
        # std::pow(complex<float>, complex<float>) rounds log and the
        # exponent product to f32 before cexp. Retain that rounding near
        # the negative-real branch cut (notably phase multiples of pi).
        logarithm = self.clog(base)
        product = self.cmul_f32(exponent, logarithm)
        return self.cexp(product)

    def cexp(self, z):
        x, y = z
        ex = self.exp(x)
        re, im = self.mul(ex, self.cos(y)), self.mul(ex, self.sin(y))
        # C99 cexp(x +/- 0i) preserves a signed imaginary zero even for NaN/inf x.
        im = self.select(self.cmp(y, self.c(0)), y, im)
        neg_inf = self.cmp(x, self.c(float('-inf')))
        re = self.select(neg_inf, self.sign(self.c(0), self.cos(y)), re)
        im = self.select(neg_inf, self.sign(self.c(0), self.sin(y)), im)
        re = self.select(self.both(self.cmp(x, self.c(float('inf'))),
                                  self.cmp(self.abs(y), self.c(float('inf')), 'uge')), self.c(float('inf')), re)
        return re, im

    def csqrt(self, z):
        x, y = z
        zero, inf, nan = self.c(0), self.c(float('inf')), self.c(float('nan'))
        h = self.hypot(x, y)
        t = self.sqrt(self.mul(self.add(h, self.abs(x)), self.c(0.5)))
        re = self.select(self.cmp(x, zero, 'oge'), t, self.div(self.abs(y), self.mul(self.c(2), t)))
        im = self.sign(self.select(self.cmp(x, zero, 'oge'), self.div(self.abs(y), self.mul(self.c(2), t)), t), y)
        iszero = self.cmp(h, zero)
        re = self.select(iszero, zero, re)
        im = self.select(iszero, y, im)
        posinf, neginf = self.cmp(x, inf), self.cmp(x, self.c(float('-inf')))
        re = self.select(neginf, self.select(self.isnan(y), nan, zero), re)
        im = self.select(neginf, self.sign(inf, y), im)
        re = self.select(posinf, inf, re)
        im = self.select(posinf, self.select(self.isnan(y), nan, self.sign(zero, y)), im)
        re = self.select(self.isinf(y), inf, re)
        im = self.select(self.isinf(y), y, im)
        return re, im

    def csinh(self, z):
        x, y = z
        sh = self.div(self.sub(self.exp(x), self.exp(self.neg(x))), self.c(2))
        ch = self.div(self.add(self.exp(x), self.exp(self.neg(x))), self.c(2))
        re, im = self.mul(sh, self.cos(y)), self.mul(ch, self.sin(y))
        re = self.select(self.cmp(x, self.c(0)), self.sign(self.c(0), x), re)
        im = self.select(self.cmp(y, self.c(0)), y, im)
        nonfinite_angle = self.cmp(self.abs(y), self.c(float('inf')), 'uge')
        re = self.select(self.both(self.isinf(x), nonfinite_angle), self.c(float('inf')), re)
        return re, im

    def ccosh(self, z):
        x, y = z
        sh = self.div(self.sub(self.exp(x), self.exp(self.neg(x))), self.c(2))
        ch = self.div(self.add(self.exp(x), self.exp(self.neg(x))), self.c(2))
        re, im = self.mul(ch, self.cos(y)), self.mul(sh, self.sin(y))
        im = self.select(self.cmp(y, self.c(0)), self.sign(y, self.mul(x, y)), im)
        im = self.select(self.cmp(x, self.c(0)), self.sign(self.c(0), self.mul(x, y)), im)
        nonfinite_angle = self.cmp(self.abs(y), self.c(float('inf')), 'uge')
        re = self.select(self.both(self.isinf(x), nonfinite_angle), self.c(float('inf')), re)
        return re, im

    def ctanh(self, z):
        x, y = z
        # Saturation avoids exp(2*abs(x)) overflow and retains the imaginary
        # signed zero as the real argument tends to either infinity.
        ax = self.abs(x)
        e = self.exp(self.mul(self.c(-2), ax))
        e2 = self.mul(e, e)
        den = self.add(self.add(self.c(1), e2), self.mul(self.mul(self.c(2), e), self.cos(self.mul(self.c(2), y))))
        re = self.sign(self.div(self.sub(self.c(1), e2), den), x)
        im = self.div(self.mul(self.mul(self.c(2), e), self.sin(self.mul(self.c(2), y))), den)
        re = self.select(self.cmp(x, self.c(0)), x, re)
        im = self.select(self.cmp(y, self.c(0)), y, im)
        re = self.select(self.isinf(x), self.sign(self.c(1), x), re)
        im = self.select(self.isinf(x), self.sign(self.c(0), y), im)
        return re, im

    def unary(self, op, z):
        x, y = z
        if op == 'neg': return self.neg(x), self.neg(y)
        if op == 'abs': return (self.hypot(x, y),)
        if op == 'isnan': return (self.either(self.isnan(x), self.isnan(y)),)
        if op == 'isinf': return (self.either(self.isinf(x), self.isinf(y)),)
        if op == 'exp': return self.cexp(z)
        if op == 'expm1':
            re, im = self.cexp(z)
            # expm1(x)*cos(y) - 2*sin(y/2)^2 avoids cancellation near zero.
            em = self.put(math.ExpM1Op(x))
            sy = self.sin(self.div(y, self.c(2)))
            re = self.sub(self.mul(em, self.cos(y)), self.mul(self.c(2), self.mul(sy, sy)))
            im = self.mul(self.exp(x), self.sin(y))
            return re, im
        if op == 'log1p':
            # CPU c10 log1p uses a correction when rounded 1+z loses bits.
            # Follow its equality tests and f32 complex multiply/divide.
            u = self.add(x, self.c(1)), y
            log_u = self.clog(u)
            recovered = self.sub(u[0], self.c(1)), y
            correction = self.cmul(log_u, self.cdiv(z, recovered))
            exact = self.both(self.cmp(recovered[0], x), self.cmp(recovered[1], y))
            rounded_one = self.both(self.cmp(u[0], self.c(1)), self.cmp(y, self.c(0)))
            return tuple(self.select(rounded_one, original, self.select(exact, log, corrected))
                         for original, log, corrected in zip(z, log_u, correction))
        if op in ('log', 'log2', 'log10'):
            re, im = self.clog((x, y))
            if op in ('log2', 'log10'):
                # libstdc++ returns float components from clogf, then divides
                # them by a float logarithm. Nonfinite complex comparisons
                # also observe the finite component's exact f32 rounding.
                s = _Scalar(Float32Type())
                d = s.c(pymath.log(2 if op == 'log2' else 10))
                re, im = s.div(s.cast(re), d), s.div(s.cast(im), d)
                self.ops += s.ops
                re, im = self.cast(re), self.cast(im)
            return re, im
        if op == 'sqrt': return self.csqrt(z)
        if op in ('reciprocal', 'rsqrt'):
            if op == 'rsqrt': z = self.csqrt(z)
            return self.cdiv((self.c(1), self.c(0)), z)
        if op == 'sinh': return self.csinh(z)
        if op == 'cosh': return self.ccosh(z)
        if op == 'tanh': return self.ctanh(z)
        if op == 'sin':
            re, im = self.csinh((self.neg(y), x))
            imag = self.neg(re)
            imag = self.select(self.both(self.isinf(y), self.cmp(self.abs(x), self.c(float('inf')), 'uge')),
                               self.c(float('inf')), imag)
            return im, imag
        if op == 'cos': return self.ccosh((self.neg(y), x))
        if op == 'tan':
            re, im = self.ctanh((self.neg(y), x))
            return im, self.neg(re)
        if op == 'sigmoid':
            re, im = self.cexp((self.neg(x), self.neg(y)))
            return self.cdiv((self.c(1), self.c(0)), (self.add(re, self.c(1)), im))
        if op in ('asin', 'acos', 'asinh', 'acosh'):
            if op == 'asinh':
                re, im = self.unary('asin', (self.neg(y), x))
                return im, self.neg(re)
            # Distances to the branch points +/-1 give principal asin/acos
            # without subtracting nearly equal complex square roots.
            one, zero, inf = self.c(1), self.c(0), self.c(float('inf'))
            ax = self.abs(x)
            plus, minus = self.add(ax, one), self.abs(self.sub(ax, one))
            r, s = self.hypot(plus, y), self.hypot(minus, y)
            a = self.div(self.add(r, s), self.c(2))
            # Rationalize hypot(t,y)-t, retaining tiny distances to the
            # branch points that disappear in (r+s)/2 - 1.
            yy = self.mul(y, y)
            delta = self.mul(self.c(0.5), self.add(self.div(yy, self.add(r, plus)),
                                                  self.div(yy, self.add(s, minus))))
            delta = self.select(self.cmp(y, zero), zero, delta)
            outside = self.cmp(ax, one, 'ogt')
            am1 = self.select(outside, self.add(self.sub(ax, one), delta), delta)
            amx = self.select(outside, delta, self.add(self.sub(one, ax), delta))
            radical = self.sqrt(self.mul(amx, self.add(a, ax)))
            asin_re = self.put(math.Atan2Op(x, radical))
            acos_re = self.put(math.Atan2Op(radical, x))
            growth = self.put(math.Log1pOp(self.add(am1, self.sqrt(self.mul(am1, self.add(a, one))))))
            infinite = self.either(self.isinf(x), self.isinf(y))
            asin_re = self.select(infinite, self.put(math.Atan2Op(x, self.abs(y))), asin_re)
            acos_re = self.select(infinite, self.put(math.Atan2Op(self.abs(y), x)), acos_re)
            growth = self.select(infinite, inf, growth)
            asin_re = self.select(self.cmp(x, zero), x, asin_re)
            acos_re = self.select(self.cmp(x, zero), self.c(pymath.pi / 2), acos_re)
            if op == 'acosh': return growth, self.sign(acos_re, y)
            if op == 'acos': return acos_re, self.neg(self.sign(growth, y))
            return asin_re, self.sign(growth, y)
        if op in ('atan', 'atanh'):
            if op == 'atan':
                re, im = self.unary('atanh', (self.neg(y), x))
                return im, self.neg(re)
            one = self.c(1)
            denominator = self.add(self.mul(self.sub(one, x), self.sub(one, x)), self.mul(y, y))
            # Use positive abs(x) then restore its sign; this avoids losing
            # tiny imaginary displacements at the -1 branch point.
            ax = self.abs(x)
            denominator = self.add(self.mul(self.sub(one, ax), self.sub(one, ax)), self.mul(y, y))
            re = self.sign(self.mul(self.c(0.25), self.put(math.Log1pOp(self.div(self.mul(self.c(4), ax), denominator)))), x)
            im = self.mul(self.c(0.5), self.put(math.Atan2Op(self.mul(self.c(2), y), self.sub(one, self.add(self.mul(x, x), self.mul(y, y))))))
            re = self.select(self.cmp(x, self.c(0)), x, re)
            infinite = self.either(self.isinf(x), self.isinf(y))
            re = self.select(infinite, self.sign(self.c(0), x), re)
            im = self.select(infinite, self.sign(self.c(pymath.pi / 2), y), im)
            im = self.select(self.isnan(y), y, im)
            return re, im
        raise ValueError(f'complex64 scalar operation {op} is unsupported')


_POINTWISE = {
    f'aten.{op}.default' for op in (
        'abs', 'neg', 'isnan', 'isinf', 'exp', 'expm1', 'log', 'log1p', 'log2',
        'log10', 'sqrt', 'rsqrt', 'reciprocal', 'sin', 'cos', 'tan', 'sinh',
        'cosh', 'tanh', 'sigmoid', 'asin', 'acos', 'atan', 'asinh', 'acosh', 'atanh')
} | {
    f'aten.{op}.{overload}' for op in ('add', 'sub', 'mul', 'div', 'eq', 'ne')
    for overload in ('Tensor', 'Scalar')
} | {'aten.pow.Scalar', 'aten.pow.Tensor_Scalar', 'aten.pow.Tensor_Tensor'}

# These operations are linear in the two components or only move elements.
# Calling the existing real lowering on logical planes preserves their axes,
# indices, padding rules and output shapes without modifying shared emitters.
_PLANAR = {
    'aten.' + target for target in (
        'alias.default', 'clone.default', 'copy.default', 'cat.default',
        'constant_pad_nd.default', 'cumsum.default', 'embedding.default',
        'expand.default', 'flip.default', 'full_like.default', 'gather.default',
        'index.Tensor', 'index_put.default', 'masked_scatter.default',
        'native_dropout.default', 'permute.default', 'repeat.default', 'scatter.src',
        'scatter.value', 'scatter_add.default', 'select.int', 'slice.Tensor',
        'slice_scatter.default', 'split_with_sizes.default', 'squeeze.dim',
        'squeeze.dims', 'sum.default', 'sum.dim_IntList', 'unsqueeze.default',
        'view.default', 'where.self')
}
_CONTRACTIONS = {'aten.mm.default', 'aten.bmm.default', 'aten.addmm.default'}
_QUERIES = {'aten.' + target for target in (
    'sym_is_contiguous.default', 'sym_numel.default', 'sym_size.int',
    'sym_storage_offset.default', 'sym_stride.int')}
COMPLEX64_TARGETS = _POINTWISE | _PLANAR | _CONTRACTIONS | _QUERIES | {
    'aten.var.dim', 'aten.var.correction', 'aten._to_copy.default',
    'aten.std.default', 'aten.std.dim', 'aten.std.correction',
    'aten.prod.default', 'aten.prod.dim_int', 'aten.mean.default', 'aten.mean.dim',
    'aten.scatter_reduce.two', 'aten.col2im.default', 'aten.embedding_dense_backward.default',
}


def _input_pairs(operands, meta):
    """Use FX dtype metadata, never the physical last extent, to identify pairs."""
    complex_values = set()
    def visit(arg):
        if hasattr(arg, 'meta') and _complex(arg.meta.get('val')):
            complex_values.add(arg.name)
        elif isinstance(arg, (list, tuple)):
            for child in arg: visit(child)
    for arg in meta['_fx_args']: visit(arg)
    for arg in meta['_fx_kwargs'].values(): visit(arg)
    # Operand order is the importer schema order, including tensor lists.
    flags = []
    def flag(arg):
        if hasattr(arg, 'meta') and hasattr(arg.meta.get('val'), 'dtype'):
            flags.append(arg.name in complex_values)
        elif isinstance(arg, (list, tuple)):
            for child in arg: flag(child)
    for arg in meta['_fx_args']: flag(arg)
    for arg in meta['_fx_kwargs'].values(): flag(arg)
    if len(flags) != len(operands):
        raise ValueError('complex64 operand metadata does not match the tensor ABI')
    ops, pairs = [], []
    for source, is_complex in zip(operands, flags):
        if is_complex:
            emitted, planes = _split(source)
            ops += emitted
            pairs.append(planes)
        else:
            pairs.append([source, None])
    return ops, pairs


def _finish(ops, values, meta, hint):
    rid = _next_region_id('complex_' + hint)
    for op in ops:
        op.attributes['prov.region_id'] = StringAttr(rid)
    return DecompResult(ops=ops, result=values[0], results=values if isinstance(meta['val'], (tuple, list)) else [],
                        region_ids=[rid], pattern_hint=hint)


def pointwise(operands, meta, node_name):
    target = meta['_aten_target']
    op = target.split('.')[1]
    shape = list(meta['val'].shape)
    out_complex = _complex(meta['val'])
    ops, pairs = _input_pairs(operands, meta)
    inputs, positions = [], []
    for pair in pairs:
        position = []
        for plane in pair:
            position.append(len(inputs) if plane is not None else None)
            if plane is not None: inputs.append(plane)
        positions.append(position)
    maps = [_broadcast_map(source.type.get_shape(), shape) for source in inputs]
    if any(mapping is None for mapping in maps):
        raise ValueError('complex64 broadcasting requires static compatible shapes')
    outputs = []
    # Transcendental functions use f64 range-sensitive intermediates;
    # arithmetic and c10-specific algorithms keep ATen's f32 rounding.
    wide = op not in ('add', 'sub', 'mul', 'neg', 'eq', 'ne', 'isnan', 'isinf', 'log1p', 'expm1', 'sigmoid')
    for component in range(2 if out_complex else 1):
        elem = Float32Type() if out_complex or op == 'abs' else i1
        def body(args, out_elem, component=component):
            b = _Scalar(Float64Type() if wide else Float32Type())
            if op in ('add', 'sub'):
                from .cpu_arithmetic import cast as cpu_cast
                convert = lambda value: cpu_cast(b.put, value, b.elem)
            else:
                convert = b.cast
            zs = [(convert(args[r]), convert(args[i]) if i is not None else b.c(0)) for r, i in positions]
            if op in ('add', 'sub', 'mul', 'div', 'eq', 'ne', 'pow'):
                if len(zs) == 1:
                    literal = _fx_arg(meta, 0 if target == 'aten.pow.Scalar' else 1)
                    components = (literal.real, literal.imag) if isinstance(literal, complex) else (literal, 0)
                    if op in ('add', 'sub'):
                        from .quant_attrs import fp32
                        # TensorIterator copies a wrapped Scalar without the
                        # checked conversion used for alpha itself.
                        components = tuple(fp32(v) if isinstance(v, float) else v for v in components)
                    scalar = b.c(components[0]), b.c(components[1])
                    zs = [scalar, zs[0]] if target == 'aten.pow.Scalar' else [zs[0], scalar]
                a, c = zs
                if op in ('add', 'sub'):
                    raw_alpha = _fx_arg(meta, 2, 1)
                    from .cpu_arithmetic import negate_alpha
                    alpha = negate_alpha(raw_alpha) if op == 'sub' else raw_alpha
                    alpha_re, alpha_im = ((alpha.real, alpha.imag) if isinstance(alpha, complex) else (alpha, 0))
                    from .cpu_arithmetic import check_coefficient
                    check_coefficient(alpha_re, Float32Type())
                    check_coefficient(alpha_im, Float32Type())
                    # CPU add's complex fmadd is a full complex product plus
                    # self, including both zero cross terms for real alpha.
                    ar, ai = b.c(alpha_re), b.c(alpha_im)
                    from .cpu_arithmetic import binary, nan_result, complex_profiles, rhs_scalar_run
                    def mul(x, y): return binary(b.put, arith.MulfOp, x, y)
                    def add(x, y): return binary(b.put, arith.AddfOp, x, y)
                    def sub(x, y): return binary(b.put, arith.SubfOp, x, y)
                    def fma(x, y, z): return nan_result(b.put, b.put(math.FmaOp(x, y, z)), [x, y, z])
                    def fms(x, y, z): return nan_result(b.put, b.put(math.FmaOp(x, y, b.neg(z))), [x, y, z])
                    # AVX2 complex multiplication uses horizontal subtraction;
                    # the scalar tail uses c10::complex's component equations.
                    vmul = (lambda coefficient, value: mul(value, coefficient)) if rhs_scalar_run(meta) else mul
                    vector = (sub(vmul(ar, c[0]), vmul(ai, c[1])),
                              sub(vmul(ai, c[0]), vmul(b.neg(ar), c[1])))
                    scalar = (sub(mul(ar, c[0]), mul(ai, c[1])),
                              add(mul(ar, c[1]), mul(ai, c[0])))
                    simd, auto, real_first, imag_first, re_other, im_other = complex_profiles(b.put, meta)
                    rz, iz = mul(ai, c[1]), mul(ai, c[0])
                    fr = b.put(math.FmaOp(ar, c[0], b.neg(rz)))
                    fi = b.put(math.FmaOp(ar, c[1], iz))
                    fused = (b.select(re_other,
                                      nan_result(b.put, fr, [c[0], ar, rz]),
                                      nan_result(b.put, fr, [ar, c[0], rz])),
                             b.select(im_other,
                                      nan_result(b.put, fi, [c[1], ar, iz]),
                                      nan_result(b.put, fi, [ar, c[1], iz])))
                    c = tuple(b.select(simd, v, b.select(auto, f, s))
                              for v, f, s in zip(vector, fused, scalar))
                    result = (b.select(real_first, add(c[0], a[0]), add(a[0], c[0])),
                              b.select(imag_first, add(c[1], a[1]), add(a[1], c[1])))
                elif op == 'mul': result = b.cmul(a, c)
                elif op == 'div': result = b.cdiv(a, c)
                elif op in ('eq', 'ne'):
                    pred = 'oeq' if op == 'eq' else 'une'
                    fn = b.both if op == 'eq' else b.either
                    result = (fn(b.cmp(a[0], c[0], pred), b.cmp(a[1], c[1], pred)),)
                else:
                    if target == 'aten.pow.Tensor_Scalar' and isinstance(_fx_arg(meta, 1), (float, int)) and _fx_arg(meta, 1) in (0, 1, 2, 3, -1, -2, 0.5, -0.5):
                        exponent = _fx_arg(meta, 1)
                        if exponent == 0: result = b.c(1), b.c(0)
                        elif exponent == 1: result = a
                        elif exponent in (0.5, -0.5):
                            result = b.csqrt(a)
                            if exponent < 0: result = b.cdiv((b.c(1), b.c(0)), result)
                        elif exponent == -1: result = b.cdiv((b.c(1), b.c(0)), a)
                        else:
                            result = b.cmul_f32(a, a)
                            if exponent == 3: result = b.cmul_f32(result, a)
                            if exponent < 0: result = b.cdiv((b.c(1), b.c(0)), result)
                    else:
                        result = b.cpow(a, c)
            else: result = b.unary(op, zs[0])
            value = result[component]
            if value.type != out_elem: value = b.cast(value, out_elem)
            return b.ops, value
        emitted = _elementwise(inputs, TensorType(elem, shape), body, input_maps=maps)
        ops += emitted[0]; outputs.append(emitted[1])
    if out_complex:
        emitted = _pack(*outputs)
        ops += emitted[0]; outputs = [emitted[1]]
    return _finish(ops, outputs, meta, op)


def _real_meta(value):
    if isinstance(value, (tuple, list)): return type(value)(_real_meta(v) for v in value)
    if _complex(value): return SimpleNamespace(shape=value.shape, dtype=torch.float32, device=value.device)
    return value


def planar(operands, meta, node_name):
    target = meta['_aten_target']
    if target == 'aten.native_dropout.default' and _fx_arg(meta, 2, True):
        raise ValueError('complex dropout requires a shared RNG mask')
    ops, pairs = _input_pairs(operands, meta)
    plane_results = []
    for component in range(2):
        args = [pair[component] if pair[component] is not None else pair[0] for pair in pairs]
        adapted = dict(meta, val=_real_meta(meta['val']))
        # The real decomposition may consult the raw nodes' dtype or shape.
        def adapt(arg):
            if hasattr(arg, 'meta'):
                return SimpleNamespace(name=arg.name, meta={**arg.meta, 'val': _real_meta(arg.meta.get('val'))})
            if isinstance(arg, (tuple, list)): return type(arg)(adapt(child) for child in arg)
            if isinstance(arg, complex): return arg.real if component == 0 else arg.imag
            if arg == torch.complex64: return torch.float32
            return arg
        adapted['_fx_args'] = tuple(adapt(arg) for arg in meta['_fx_args'])
        adapted['_fx_kwargs'] = {key: adapt(arg) for key, arg in meta['_fx_kwargs'].items()}
        adapted['_fx_ssa_args'] = {
            index: args[operands.index(source)] for index, source in meta.get('_fx_ssa_args', {}).items()
        }
        # A real scalar fill or padding has zero imaginary component.
        scalar_position = {'aten.constant_pad_nd.default': 2, 'aten.full_like.default': 1,
                           'aten.scatter.value': 3}.get(target)
        if scalar_position is not None and component == 1:
            raw = _fx_arg(meta, scalar_position, 0)
            scalar = raw.imag if isinstance(raw, complex) else 0
            seq = list(adapted['_fx_args'])
            if scalar_position < len(seq): seq[scalar_position] = scalar
            else:
                schema = getattr(torch.ops.aten, target.split('.')[1])
                argument = getattr(schema, target.split('.')[2])._schema.arguments[scalar_position]
                adapted['_fx_kwargs'][argument.name] = scalar
            adapted['_fx_args'] = tuple(seq)
        result = DECOMPOSITION_TABLE[target](args, adapted, node_name + f'_{component}')
        ops += result.ops
        plane_results.append(result.results or [result.result])
    values = meta['val'] if isinstance(meta['val'], (tuple, list)) else [meta['val']]
    outputs = []
    for ordinal, value in enumerate(values):
        if _complex(value):
            emitted = _pack(plane_results[0][ordinal], plane_results[1][ordinal])
            ops += emitted[0]; outputs.append(emitted[1])
        else: outputs.append(plane_results[0][ordinal])
    return _finish(ops, outputs, meta, target.split('.')[1])


def dispatch(operands, meta, node_name):
    target = meta['_aten_target']
    allowed = (torch.complex64, torch.float32, torch.int64, torch.int32, torch.bool)
    if target in ('aten.add.Tensor', 'aten.sub.Tensor'):
        allowed += (torch.float16, torch.bfloat16)
    if any(getattr(arg.meta.get('val'), 'dtype', None) not in allowed
           for arg in meta['_fx_args'] if hasattr(arg, 'meta') and hasattr(arg.meta.get('val'), 'dtype')):
        raise ValueError('complex lowering currently requires complex64 components')
    if target in _POINTWISE: return pointwise(operands, meta, node_name)
    if target in _STRUCTURAL: return structural(operands, meta, node_name)
    if target.startswith(('aten.sum.', 'aten.prod.')) and meta['val'].dtype == torch.bool:
        # The requested dtype converts complex truthiness before reduction;
        # a nonzero imaginary component must not be discarded.
        converted = cast(operands, meta, node_name + '_truth')
        reduced = DECOMPOSITION_TABLE[target]([converted.result], meta, node_name)
        return _finish([*converted.ops, *reduced.ops], [reduced.result], meta, target.split('.')[1])
    if target.startswith('aten.mean.'): return mean(operands, meta, node_name)
    if target == 'aten.scatter_reduce.two': return scatter_reduce(operands, meta, node_name)
    if target == 'aten.col2im.default': return col2im(operands, meta, node_name)
    if target == 'aten.embedding_dense_backward.default': return embedding_backward(operands, meta, node_name)
    if target in _PLANAR: return planar(operands, meta, node_name)
    if target in _CONTRACTIONS: return contraction(operands, meta, node_name)
    if target.startswith('aten.var.'): return variance(operands, meta, node_name)
    if target.startswith('aten.std.'):
        adapted = dict(meta, _aten_target=target.replace('aten.std.', 'aten.var.'))
        if target.endswith('.default'):
            adapted['_aten_target'] = 'aten.var.dim'
            adapted['_fx_args'] = (_fx_arg(meta, 0), None, _fx_arg(meta, 1, True))
        result = variance(operands, adapted, node_name)
        emitted = _elementwise([result.result], result.result.type,
                              lambda args, elem: ([op := math.SqrtOp(args[0])], op.result))
        return _finish([*result.ops, *emitted[0]], [emitted[1]], meta, 'variance')
    if target.startswith('aten.prod.'): return product(operands, meta, node_name)
    if target == 'aten._to_copy.default': return cast(operands, meta, node_name)
    if target in _QUERIES:
        return query(operands, meta, node_name)
    raise ValueError(f'{target} complex64 lowering is not implemented')


def complex64_decomposition(node, target):
    """Return a pair-aware emitter only when every complex dtype is complex64."""
    if target not in COMPLEX64_TARGETS: return None
    dtypes = []
    def visit(value):
        if isinstance(value, (list, tuple)):
            for child in value: visit(child)
        elif hasattr(value, 'dtype') and value.dtype.is_complex: dtypes.append(value.dtype)
    visit(node.meta.get('val'))
    for argument in node.all_input_nodes: visit(argument.meta.get('val'))
    return dispatch if dtypes and all(dtype == torch.complex64 for dtype in dtypes) else None


def contraction(operands, meta, node_name):
    ops, pairs = _input_pairs(operands, meta)
    target = meta['_aten_target']
    addmm = target == 'aten.addmm.default'
    a, b = pairs[-2:]
    shape = list(meta['val'].shape)
    adapted = dict(meta, val=_real_meta(meta['val']))
    alpha = complex(_fx_arg(meta, 4, 1)) if addmm else 1
    beta = complex(_fx_arg(meta, 3, 1)) if addmm else 0
    if addmm and alpha == 0:
        outputs = []
        for component in range(2):
            if beta == 0:
                emitted = _splat_scalar(0, TensorType(Float32Type(), shape))
            else:
                def body(args, elem, component=component):
                    s = _Scalar()
                    z = args if beta == 1 else s.cmul(args, (s.c(beta.real), s.c(beta.imag)))
                    return s.ops, z[component]
                maps = [_broadcast_map(value.type.get_shape(), shape) for value in pairs[0]]
                emitted = _elementwise(pairs[0], TensorType(Float32Type(), shape), body, input_maps=maps)
            ops += emitted[0]; outputs.append(emitted[1])
        emitted = _pack(*outputs); ops += emitted[0]
        return _finish(ops, [emitted[1]], meta, 'matmul')
    if addmm and alpha != 1:
        # CPU BLAS scales B as part of the product. Scaling only the final
        # real contractions loses imaginary NaNs when alpha*B overflows.
        scaled = []
        for component in range(2):
            def body(args, elem, component=component):
                s = _Scalar()
                value = s.cmul(args, (s.c(alpha.real), s.c(alpha.imag)))[component]
                return s.ops, value
            emitted = _elementwise(b, b[0].type, body)
            ops += emitted[0]; scaled.append(emitted[1])
        b = scaled
    products = []
    for left, right in ((a[0], b[0]), (a[1], b[1]), (a[0], b[1]), (a[1], b[0])):
        fn = DECOMPOSITION_TABLE['aten.mm.default' if addmm else target]
        result = fn([left, right], adapted, node_name)
        ops += result.ops; products.append(result.result)
    inputs = list(products)
    if addmm and beta != 0: inputs += pairs[0]
    maps = [_broadcast_map(source.type.get_shape(), shape) for source in inputs]
    outputs = []
    for component in range(2):
        def body(args, elem, component=component):
            s = _Scalar()
            z = s.sub(args[0], args[1]), s.add(args[2], args[3])
            if addmm and beta != 0:
                bias = args[4], args[5]
                if beta != 1: bias = s.cmul(bias, (s.c(beta.real), s.c(beta.imag)))
                z = s.add(z[0], bias[0]), s.add(z[1], bias[1])
            return s.ops, z[component]
        emitted = _elementwise(inputs, TensorType(Float32Type(), shape), body, input_maps=maps)
        ops += emitted[0]; outputs.append(emitted[1])
    emitted = _pack(*outputs)
    ops += emitted[0]
    return _finish(ops, [emitted[1]], meta, 'matmul')


def variance(operands, meta, node_name):
    from m2m.ir.decompositions import _reduce, _emit_reshape, _cast_tensor, _reduction_dims
    ops, pairs = _input_pairs(operands, meta)
    re, im = pairs[0]
    shape = list(re.type.get_shape()); rank = len(shape)
    # CPU variance uses a double accumulation type for complex64. This is
    # essential even for a constant maximum-finite input, whose f32 sum can
    # overflow before computing its zero deviations.
    widened = []
    for plane in (re, im):
        emitted = _cast_tensor(plane, shape, Float64Type())
        ops += emitted[0]; widened.append(emitted[1])
    re, im = widened
    raw_dims = _fx_arg(meta, 1)
    dims = _reduction_dims(raw_dims, rank)
    count = pymath.prod(shape[d] for d in dims)
    correction = (1 if _fx_arg(meta, 2, True) else 0) if meta['_aten_target'].endswith('.dim') else _fx_arg(meta, 2, None)
    correction = 1 if correction is None else float(correction)
    out_shape = list(meta['val'].shape)
    if count == 0:
        emitted = _splat_scalar(float('nan'), TensorType(Float32Type(), out_shape))
        return _finish([*ops, *emitted[0]], [emitted[1]], meta, 'variance')
    reduced_shape = [extent for d, extent in enumerate(shape) if d not in dims]
    means = []
    for source in (re, im):
        emitted, summ, _ = _reduce(source, shape, dims, 0, arith.AddfOp, Float64Type())
        ops += emitted
        def divide(args, elem):
            b = _Scalar(Float64Type()); value = b.div(args[0], b.c(count)); return b.ops, value
        emitted = _elementwise([summ], TensorType(Float64Type(), reduced_shape), divide)
        ops += emitted[0]; means.append(emitted[1])
    mean_map = AffineMap(rank, 0, tuple(AffineExpr.dimension(d) for d in range(rank) if d not in dims))
    def deviation(args, elem):
        b = _Scalar(Float64Type())
        x, y = b.sub(args[0], args[2]), b.sub(args[1], args[3])
        value = b.add(b.mul(x, x), b.mul(y, y))
        return b.ops, value
    emitted = _elementwise([re, im, *means], TensorType(Float64Type(), shape), deviation,
                           input_maps=[AffineMap.identity(rank)] * 2 + [mean_map] * 2)
    ops += emitted[0]
    emitted, summ, _ = _reduce(emitted[1], shape, dims, 0, arith.AddfOp, Float64Type())
    ops += emitted
    def divide(args, elem):
        b = _Scalar(Float64Type()); value = b.div(args[0], b.c(max(0, count - correction))); return b.ops, value
    emitted = _elementwise([summ], summ.type, divide)
    ops += emitted[0]
    narrowed = _cast_tensor(emitted[1], reduced_shape, Float32Type())
    ops += narrowed[0]
    reshaped = _emit_reshape(narrowed[1], out_shape, Float32Type())
    ops += reshaped[0]
    return _finish(ops, [reshaped[1]], meta, 'variance')


def product(operands, meta, node_name):
    from m2m.ir.decompositions import _emit_reshape, _reduction_dims
    ops, pairs = _input_pairs(operands, meta)
    re, im = pairs[0]
    shape = list(re.type.get_shape()); rank = len(shape)
    dims = _reduction_dims(None if meta['_aten_target'] == 'aten.prod.default' else _fx_arg(meta, 1), rank)
    reduced_shape = [extent for d, extent in enumerate(shape) if d not in dims]
    rt = TensorType(Float32Type(), reduced_shape)
    initial = []
    for value in (1, 0):
        emitted = _splat_scalar(value, rt)
        ops += emitted[0]; initial.append(emitted[1])
    block = Block(arg_types=[Float32Type()] * 4)
    b = _Scalar()
    # Native scalar complex product prioritizes the real product in re and
    # the imaginary product in im. Make NaN quieting/invalid bits explicit;
    # LLVM may otherwise commute the two products (or fold the seed * 1).
    from .cpu_arithmetic import binary
    ar, ai, br, bi = block.args[2], block.args[3], block.args[0], block.args[1]
    rr = binary(b.put, arith.MulfOp, ar, br)
    ii = binary(b.put, arith.MulfOp, ai, bi)
    ri = binary(b.put, arith.MulfOp, ar, bi)
    ir = binary(b.put, arith.MulfOp, ai, br)
    result = binary(b.put, arith.SubfOp, rr, ii), binary(b.put, arith.AddfOp, ri, ir)
    block.add_ops([*b.ops, YieldOp(*result)])
    projection = AffineMap(rank, 0, tuple(AffineExpr.dimension(d) for d in range(rank) if d not in dims))
    generic = GenericOp(inputs=[re, im], outputs=initial, body=Region(block),
                        indexing_maps=[AffineMapAttr(AffineMap.identity(rank))] * 2 + [AffineMapAttr(projection)] * 2,
                        iterator_types=[IteratorTypeAttr(IteratorType.REDUCTION if d in dims else IteratorType.PARALLEL) for d in range(rank)],
                        result_types=[rt, rt])
    ops.append(generic)
    planes = []
    for result in generic.results:
        emitted = _emit_reshape(result, list(meta['val'].shape), Float32Type())
        ops += emitted[0]; planes.append(emitted[1])
    emitted = _pack(*planes); ops += emitted[0]
    return _finish(ops, [emitted[1]], meta, 'prod')


def cast(operands, meta, node_name):
    from m2m.ir.decompositions import _cast_tensor, _element_type_from_meta
    ops, pairs = _input_pairs(operands, meta)
    source = pairs[0][0]
    if _complex(meta['val']):
        if pairs[0][1] is not None: return _finish([], [operands[0]], meta, 'cast')
        emitted = _cast_tensor(source, list(source.type.get_shape()), Float32Type())
        ops += emitted[0]; source = emitted[1]
        emitted = _splat_scalar(0, source.type); ops += emitted[0]
        emitted = _pack(source, emitted[1]); ops += emitted[0]
    elif meta['val'].dtype == torch.bool:
        shape = list(source.type.get_shape())
        def body(args, elem):
            b = _Scalar(); value = b.either(b.cmp(args[0], b.c(0), 'une'), b.cmp(args[1], b.c(0), 'une'))
            return b.ops, value
        emitted = _elementwise(pairs[0], TensorType(i1, shape), body); ops += emitted[0]
    else:
        emitted = _cast_tensor(source, list(source.type.get_shape()), _element_type_from_meta(meta))
        ops += emitted[0]
    return _finish(ops, [emitted[1]], meta, 'cast')


def mean(operands, meta, node_name):
    from m2m.ir.decompositions import _reduce, _emit_reshape, _reduction_dims
    ops, pairs = _input_pairs(operands, meta)
    shape = list(pairs[0][0].type.get_shape()); rank = len(shape)
    raw = None if meta['_aten_target'] == 'aten.mean.default' else _fx_arg(meta, 1)
    dims = _reduction_dims(raw, rank)
    count = pymath.prod(shape[d] for d in dims)
    sums = []
    for plane in pairs[0]:
        emitted, value, reduced_shape = _reduce(plane, shape, dims, 0, arith.AddfOp, Float32Type())
        ops += emitted
        sums.append(value)
    outputs = []
    # sum allocates a contiguous result, then div_ runs TensorIterator over
    # that result (not over the original reduction input's strides).
    division_output = torch.empty(reduced_shape, dtype=torch.complex64, device='meta')
    for component in range(2):
        def body(args, elem, component=component):
            b = _Scalar()
            factor = b.div(b.c(1), b.c(count))
            from .cpu_arithmetic import binary, nan_result
            # CPU mean is sum.div_(count). Complex division's scaled
            # numerators use fused x + y*(+0), y + x*(-0), followed by
            # multiplication by the rounded reciprocal. A NaN addend wins
            # over an invalid 0*inf. The scalar tail keeps the primary NaN;
            # the compiled two-vector loop puts the cross term first.
            cross, primary = args[1 - component], args[component]
            zero = b.c(0. if component == 0 else -0.)
            numerator = b.put(math.FmaOp(cross, zero, primary))
            from .cpu_pointwise import flat_conversion_prefix
            width = 512 if torch.backends.cpu.get_cpu_capability() == 'AVX512' else 256
            vector = flat_conversion_prefix(b.put, division_output, width // 32)
            numerator = b.select(vector,
                nan_result(b.put, numerator, [cross, zero, primary]),
                nan_result(b.put, numerator, [primary, cross, zero]))
            value = binary(b.put, arith.MulfOp, numerator, factor)
            return b.ops, value
        emitted = _elementwise(sums, sums[0].type, body); ops += emitted[0]
        emitted = _emit_reshape(emitted[1], list(meta['val'].shape), Float32Type())
        ops += emitted[0]; outputs.append(emitted[1])
    emitted = _pack(*outputs); ops += emitted[0]
    return _finish(ops, [emitted[1]], meta, 'mean')


class _Index(Builder):
    def c(self, value): return self.put(arith.ConstantOp(IntegerAttr(value, IndexType())))
    def add(self, a, b): return self.put(arith.AddiOp(a, b))
    def sub(self, a, b): return self.put(arith.SubiOp(a, b))
    def mul(self, a, b): return self.put(arith.MuliOp(a, b))
    def div(self, a, b): return self.put(arith.DivSIOp(a, b))
    def rem(self, a, b): return self.put(arith.RemSIOp(a, b))
    def cmp(self, a, b, pred): return self.put(arith.CmpiOp(a, b, pred))
    def select(self, cond, a, b): return self.put(arith.SelectOp(cond, a, b))


_STRUCTURAL = {'aten.' + target for target in (
    'as_strided.default', 'diagonal.default', 'index_select.default', 'fill.Scalar',
    'reflection_pad1d.default', 'reflection_pad2d.default', 'reflection_pad3d.default',
    'replication_pad2d.default', 'replication_pad3d.default', 'select_scatter.default',
    'resize_.default')}
COMPLEX64_TARGETS |= _STRUCTURAL


def structural(operands, meta, node_name):
    from m2m.ir.decompositions import _emit_reshape
    target = meta['_aten_target']; op = target.split('.')[1]
    ops, pairs = _input_pairs(operands, meta)
    shape = list(meta['val'].shape)
    source_shape = list(pairs[0][0].type.get_shape()); rank = len(source_shape)
    if op == 'as_strided':
        value = _fx_arg(meta, 0).meta['val']
        if not value.is_contiguous():
            raise ValueError('complex as_strided requires a contiguous source storage envelope')
        requested_offset = _fx_arg(meta, 3, None)
        base_offset = int(value.storage_offset())
        relative_offset = (base_offset if requested_offset is None else int(requested_offset)) - base_offset
        strides = [int(v) for v in _fx_arg(meta, 2)]
        max_offset = relative_offset + sum((n - 1) * s for n, s in zip(shape, strides))
        if any(s < 0 for s in strides) or relative_offset < 0 or (pymath.prod(shape) and max_offset >= pymath.prod(source_shape)):
            raise ValueError('complex as_strided reads outside the source tensor storage envelope')
    outputs = []
    for component in range(2):
        source = pairs[0][component]
        if op == 'fill':
            scalar = complex(_fx_arg(meta, 1))
            emitted = _splat_scalar(scalar.real if component == 0 else scalar.imag, TensorType(Float32Type(), shape))
        elif op == 'resize_':
            if pymath.prod(source_shape) != pymath.prod(shape):
                raise ValueError('complex resize with changed element count has uninitialized storage semantics')
            emitted = _emit_reshape(source, shape, Float32Type())
        else:
            def build(block, indices):
                b = _Index(block)
                coordinates = list(indices)
                if op == 'as_strided':
                    linear = b.c(relative_offset)
                    for index, stride in zip(indices, strides):
                        linear = b.add(linear, b.mul(index, b.c(stride)))
                    coordinates = []
                    for axis, extent in enumerate(source_shape):
                        divisor = pymath.prod(source_shape[axis + 1:])
                        coordinates.append(b.rem(b.div(linear, b.c(divisor)), b.c(extent)))
                elif op == 'diagonal':
                    offset = int(_fx_arg(meta, 1, 0))
                    d1, d2 = int(_fx_arg(meta, 2, 0)) % rank, int(_fx_arg(meta, 3, 1)) % rank
                    kept = [d for d in range(rank) if d not in (d1, d2)]
                    coordinates = [None] * rank
                    for axis, value in zip(kept, indices[:-1]): coordinates[axis] = value
                    coordinates[d1] = b.add(indices[-1], b.c(max(0, -offset)))
                    coordinates[d2] = b.add(indices[-1], b.c(max(0, offset)))
                elif op == 'index_select':
                    dim = int(_fx_arg(meta, 1)) % rank
                    index_source = pairs[1][0]
                    picked = b.put(tensor.ExtractOp(index_source, [indices[dim]], index_source.type.element_type))
                    coordinates[dim] = b.put(arith.IndexCastOp(picked, IndexType()))
                elif op.startswith(('reflection_pad', 'replication_pad')):
                    pads = list(_fx_arg(meta, 1))
                    for j in range(len(pads) // 2):
                        dim = rank - 1 - j
                        pos = b.sub(indices[dim], b.c(int(pads[2 * j])))
                        if op.startswith('reflection'):
                            pos = b.select(b.cmp(pos, b.c(0), 'slt'), b.sub(b.c(0), pos), pos)
                            pos = b.select(b.cmp(pos, b.c(source_shape[dim]), 'sge'), b.sub(b.c(2 * source_shape[dim] - 2), pos), pos)
                        else:
                            pos = b.select(b.cmp(pos, b.c(0), 'slt'), b.c(0), pos)
                            pos = b.select(b.cmp(pos, b.c(source_shape[dim]), 'sge'), b.c(source_shape[dim] - 1), pos)
                        coordinates[dim] = pos
                elif op == 'select_scatter':
                    from xdsl.dialects import scf
                    dim = int(_fx_arg(meta, 2)) % rank
                    index = int(_fx_arg(meta, 3)) % source_shape[dim]
                    cond = b.cmp(indices[dim], b.c(index), 'eq')
                    yes, no = Block(), Block()
                    src = pairs[1][component]
                    value = tensor.ExtractOp(src, [v for d, v in enumerate(indices) if d != dim], Float32Type())
                    yes.add_ops([value, scf.YieldOp(value.result)])
                    old = tensor.ExtractOp(source, indices, Float32Type())
                    no.add_ops([old, scf.YieldOp(old.result)])
                    return b.put(scf.IfOp(cond, [Float32Type()], Region(yes), Region(no)))
                return b.put(tensor.ExtractOp(source, coordinates, Float32Type()))
            emitted = generate(shape, Float32Type(), build)
        ops += emitted[0]; outputs.append(emitted[1])
    emitted = _pack(*outputs); ops += emitted[0]
    return _finish(ops, [emitted[1]], meta, op)


def query(operands, meta, node_name):
    source = _fx_arg(meta, 0).meta['val']
    target = meta['_aten_target']
    if target == 'aten.sym_is_contiguous.default':
        value = source.is_contiguous(memory_format=_fx_arg(meta, 1, torch.contiguous_format))
    elif target == 'aten.sym_stride.int': value = source.stride(int(_fx_arg(meta, 1)))
    elif target == 'aten.sym_storage_offset.default': value = source.storage_offset()
    elif target == 'aten.sym_size.int': value = source.shape[int(_fx_arg(meta, 1))]
    else: value = source.numel()
    if type(value) not in (int, bool):
        raise ValueError('complex metadata query requires a static layout')
    dtype = i1 if isinstance(value, bool) else i64
    constant = arith.ConstantOp(IntegerAttr(int(value), dtype))
    return _finish([constant], [constant.result], meta, 'local_scalar')


def scatter_reduce(operands, meta, node_name):
    from xdsl.dialects import scf
    ops, pairs = _input_pairs(operands, meta)
    own, index_pair, src = pairs
    index = index_pair[0]
    shape = list(own[0].type.get_shape()); rank = len(shape)
    index_shape = list(index.type.get_shape())
    axis = int(_fx_arg(meta, 1)) % rank
    reduction = _fx_arg(meta, 4)
    include_self = bool(_fx_arg(meta, 5, True))
    if reduction not in ('sum', 'mean', 'prod'):
        raise ValueError('complex scatter_reduce supports sum, mean and prod')
    outputs = []
    for component in range(2):
        def build(block, coordinates, component=component):
            b = _Index(block)
            old = [b.put(tensor.ExtractOp(plane, coordinates, Float32Type())) for plane in own]
            s = _Scalar()
            initial = old if include_self else [s.c(1 if reduction == 'prod' else 0), s.c(0)]
            block.add_ops(s.ops)
            z, step, limit = b.c(0), b.c(1), b.c(index_shape[axis])
            loop_block = Block(arg_types=[IndexType(), Float32Type(), Float32Type(), IndexType()])
            j, ar, ai, count = loop_block.args
            ib = _Index(loop_block)
            position = list(coordinates); position[axis] = j
            # Non-scatter dimensions of the index can be smaller than self.
            within = None
            for d in range(rank):
                if d == axis: continue
                valid = ib.cmp(coordinates[d], ib.c(index_shape[d]), 'slt')
                within = valid if within is None else ib.put(arith.AndIOp(within, valid))
            yes = Block(); yb = _Index(yes)
            picked = yb.put(tensor.ExtractOp(index, position, index.type.element_type))
            picked = yb.put(arith.IndexCastOp(picked, IndexType()))
            match = yb.cmp(picked, coordinates[axis], 'eq')
            update = Block()
            values = []
            for plane in src:
                value = tensor.ExtractOp(plane, position, Float32Type()); update.add_op(value); values.append(value.result)
            scalar = _Scalar()
            if reduction == 'prod': new = scalar.cmul((ar, ai), values)
            else: new = scalar.add(ar, values[0]), scalar.add(ai, values[1])
            update.add_ops(scalar.ops)
            increment = arith.AddiOp(count, step); update.add_op(increment)
            update.add_op(scf.YieldOp(*new, increment.result))
            unchanged = Block(); unchanged.add_op(scf.YieldOp(ar, ai, count))
            branch = scf.IfOp(match, [Float32Type(), Float32Type(), IndexType()], Region(update), Region(unchanged))
            yes.add_ops([branch, scf.YieldOp(*branch.results)])
            if within is None:
                for operation in list(yes.ops): yes.detach_op(operation); loop_block.add_op(operation)
                # Replace the transferred branch terminator by the loop terminator.
            else:
                no = Block(); no.add_op(scf.YieldOp(ar, ai, count))
                valid_branch = scf.IfOp(within, [Float32Type(), Float32Type(), IndexType()], Region(yes), Region(no))
                loop_block.add_ops([valid_branch, scf.YieldOp(*valid_branch.results)])
            initial_count = b.c(1 if include_self else 0)
            loop = scf.ForOp(z, limit, step, [*initial, initial_count], Region(loop_block))
            block.add_op(loop)
            scalar = _Scalar()
            value = loop.results[component]
            if reduction == 'mean':
                float_count = b.put(arith.IndexCastOp(loop.results[2], i64))
                float_count = scalar.cast(float_count)
                value = scalar.div(value, float_count)
            block.add_ops(scalar.ops)
            touched = b.cmp(loop.results[2], z, 'sgt')
            return b.select(touched, value, old[component])
        emitted = generate(shape, Float32Type(), build)
        ops += emitted[0]; outputs.append(emitted[1])
    emitted = _pack(*outputs); ops += emitted[0]
    return _finish(ops, [emitted[1]], meta, 'scatter')


def col2im(operands, meta, node_name):
    from xdsl.dialects import scf
    ops, pairs = _input_pairs(operands, meta)
    shape = list(meta['val'].shape)
    kh, kw = (int(v) for v in _fx_arg(meta, 2))
    dh, dw = (int(v) for v in _fx_arg(meta, 3))
    ph, pw = (int(v) for v in _fx_arg(meta, 4))
    sh, sw = (int(v) for v in _fx_arg(meta, 5))
    height, width = shape[-2:]
    blocks_h = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    blocks_w = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    outputs = []
    for component in range(2):
        source = pairs[0][component]
        def build(block, coordinates):
            b = _Index(block)
            z, step, limit = b.c(0), b.c(1), b.c(kh * kw)
            scalar = _Scalar(); initial = scalar.c(0); block.add_ops(scalar.ops)
            body = Block(arg_types=[IndexType(), Float32Type()]); j, acc = body.args
            ib = _Index(body)
            ky, kx = ib.div(j, ib.c(kw)), ib.rem(j, ib.c(kw))
            y = ib.sub(ib.add(coordinates[-2], ib.c(ph)), ib.mul(ky, ib.c(dh)))
            x = ib.sub(ib.add(coordinates[-1], ib.c(pw)), ib.mul(kx, ib.c(dw)))
            conditions = [ib.cmp(y, z, 'sge'), ib.cmp(x, z, 'sge'),
                          ib.cmp(ib.rem(y, ib.c(sh)), z, 'eq'), ib.cmp(ib.rem(x, ib.c(sw)), z, 'eq')]
            by, bx = ib.div(y, ib.c(sh)), ib.div(x, ib.c(sw))
            conditions += [ib.cmp(by, ib.c(blocks_h), 'slt'), ib.cmp(bx, ib.c(blocks_w), 'slt')]
            cond = conditions[0]
            for other in conditions[1:]: cond = ib.put(arith.AndIOp(cond, other))
            yes, no = Block(), Block(); yb = _Index(yes)
            channel = yb.add(yb.mul(coordinates[-3], yb.c(kh * kw)), j)
            column = yb.add(yb.mul(by, yb.c(blocks_w)), bx)
            position = [*coordinates[:-3], channel, column]
            value = yb.put(tensor.ExtractOp(source, position, Float32Type()))
            added = arith.AddfOp(acc, value); yes.add_ops([added, scf.YieldOp(added.result)])
            no.add_op(scf.YieldOp(acc))
            branch = scf.IfOp(cond, [Float32Type()], Region(yes), Region(no))
            body.add_ops([branch, scf.YieldOp(branch.results[0])])
            loop = scf.ForOp(z, limit, step, [initial], Region(body)); block.add_op(loop)
            return loop.results[0]
        emitted = generate(shape, Float32Type(), build)
        ops += emitted[0]; outputs.append(emitted[1])
    emitted = _pack(*outputs); ops += emitted[0]
    return _finish(ops, [emitted[1]], meta, 'col2im')


def embedding_backward(operands, meta, node_name):
    """Scale complex contributions before the ordered row accumulation."""
    from xdsl.dialects import scf
    from m2m.ir.core_aten import embedding_backward as real_backward
    ops, pairs = _input_pairs(operands, meta)
    grad, index_pair = pairs
    indices = index_pair[0]
    shape = list(grad[0].type.get_shape()); ishape = list(indices.type.get_shape())
    frequency = bool(_fx_arg(meta, 4, False))
    scaled = []
    for component in range(2):
        def build(block, position, component=component):
            b = _Index(block); s = _Scalar()
            values = [b.put(tensor.ExtractOp(plane, position, Float32Type())) for plane in grad]
            if frequency:
                picked = b.put(tensor.ExtractOp(indices, position[:-1], indices.type.element_type))
                z, step, limit = b.c(0), b.c(1), b.c(pymath.prod(ishape))
                body = Block(arg_types=[IndexType(), IndexType()]); iv, count = body.args
                ib = _Index(body); coordinates = []
                for axis, extent in enumerate(ishape):
                    divisor = pymath.prod(ishape[axis + 1:])
                    coordinates.append(ib.rem(ib.div(iv, ib.c(divisor)), ib.c(extent)))
                other = ib.put(tensor.ExtractOp(indices, coordinates, indices.type.element_type))
                equal = ib.cmp(picked, other, 'eq')
                increment = ib.select(equal, step, z)
                added = ib.add(count, increment); body.add_op(scf.YieldOp(added))
                loop = scf.ForOp(z, limit, step, [z], Region(body)); block.add_op(loop)
                integer = b.put(arith.IndexCastOp(loop.results[0], i64))
                factor = s.div(s.c(1), s.cast(integer))
            else: factor = s.c(1)
            result = s.cmul(values, (factor, s.c(0)))[component]
            block.add_ops(s.ops)
            return result
        emitted = generate(shape, Float32Type(), build)
        ops += emitted[0]; scaled.append(emitted[1])
    adapted = dict(meta, val=_real_meta(meta['val']))
    args = list(meta['_fx_args'])
    if len(args) > 4: args[4] = False
    else: adapted['_fx_kwargs'] = {**meta['_fx_kwargs'], 'scale_grad_by_freq': False}
    adapted['_fx_args'] = tuple(args)
    outputs = []
    for plane in scaled:
        result = real_backward([plane, indices], adapted, node_name)
        ops += result.ops; outputs.append(result.result)
    emitted = _pack(*outputs); ops += emitted[0]
    return _finish(ops, [emitted[1]], meta, 'embedding_backward')
