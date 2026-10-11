"""Integer/Boolean ATen arithmetic, isolated from floating and complex emitters.

FX result metadata is PyTorch's promotion oracle (including wrapped scalars,
zero-dimensional tensors and the capture-time default floating dtype). Cast
operands to that dtype before arithmetic; never compute integral results through
floating point. Integer arith ops deliberately have no overflow flags.
"""
from __future__ import annotations

from functools import wraps

from xdsl.dialects import arith, math, scf
from xdsl.dialects.builtin import AnyFloat, IntegerAttr, IntegerType, TensorType
from xdsl.ir import Block, Region


_FLOATING_UNARY = {
    'acos', 'asin', 'atan', 'acosh', 'asinh', 'atanh', 'sin', 'cos', 'tan',
    'sinh', 'cosh', 'tanh', 'exp', 'exp2', 'expm1', 'log', 'log1p', 'log2',
    'log10', 'sqrt', 'rsqrt', 'sigmoid', 'erf', 'reciprocal',
}
_INTEGER_IDENTITY = {'floor', 'ceil', 'round', 'trunc'}


def _constant(value, elem):
    from m2m.ir.decompositions import _wrap_int
    return arith.ConstantOp(IntegerAttr(_wrap_int(value, elem.width.data), elem))


def _integer_pow(base, exponent, elem):
    """ATen's integral powi: squaring modulo 2**width; tensor negatives.

    A positive exponent takes at most width-1 iterations. Negative exponents
    produce zero except for bases +/-1, retaining parity even at INT_MIN.
    """
    zero, one, minus_one = (_constant(v, elem) for v in (0, 1, -1))
    negative = arith.CmpiOp(exponent, zero.result, 'slt')
    start_exp = arith.SelectOp(negative.result, zero.result, exponent)
    before = Block(arg_types=[elem, elem, elem])
    active = arith.CmpiOp(before.args[0], zero.result, 'sgt')
    before.add_ops([active, scf.ConditionOp(active.result, *before.args)])
    after = Block(arg_types=[elem, elem, elem])
    e, b, acc = after.args
    bit = arith.AndIOp(e, one.result)
    odd = arith.CmpiOp(bit.result, zero.result, 'ne')
    product = arith.MuliOp(acc, b)
    chosen = arith.SelectOp(odd.result, product.result, acc)
    squared = arith.MuliOp(b, b)
    shifted = arith.ShRUIOp(e, one.result)
    after.add_ops([bit, odd, product, chosen, squared, shifted,
                   scf.YieldOp(shifted.result, squared.result, chosen.result)])
    loop = scf.WhileOp([start_exp.result, base, one.result], [elem]*3,
                       Region(before), Region(after))
    is_one = arith.CmpiOp(base, one.result, 'eq')
    is_minus_one = arith.CmpiOp(base, minus_one.result, 'eq')
    parity = arith.AndIOp(exponent, one.result)
    is_odd = arith.CmpiOp(parity.result, zero.result, 'ne')
    sign = arith.SelectOp(is_odd.result, minus_one.result, one.result)
    unit = arith.SelectOp(is_one.result, one.result, zero.result)
    negative_result = arith.SelectOp(is_minus_one.result, sign.result, unit.result)
    result = arith.SelectOp(negative.result, negative_result.result, loop.results[2])
    return [zero, one, minus_one, negative, start_exp, loop, is_one,
            is_minus_one, parity, is_odd, sign, unit, negative_result, result], result.result


def _pow(operands, meta, target):
    from m2m.ir import decompositions as d
    elem = d._element_type_from_meta(meta)
    scalar_left = target == 'aten.pow.Scalar'
    scalar_right = target == 'aten.pow.Tensor_Scalar'
    scalar = d._fx_arg(meta, 0 if scalar_left else 1) if scalar_left or scalar_right else None
    if isinstance(elem, IntegerType):
        if elem.width.data == 1:
            raise ValueError('pow is not implemented for Boolean results')
        if scalar_right and isinstance(scalar, int) and scalar < 0:
            raise ValueError('Integers to negative integer powers are not allowed.')
        if scalar_right and not 0 <= int(scalar) < (1 << (elem.width.data - 1)):
            raise ValueError('integer exponent cannot be converted without overflow')

    def build(args, oe):
        pre = []
        if scalar_left or scalar_right:
            if isinstance(oe, IntegerType):
                c = _constant(int(scalar), oe)
            else:
                from xdsl.dialects.builtin import FloatAttr
                c = arith.ConstantOp(FloatAttr(float(scalar), oe))
            pre.append(c)
            base, exp = (c.result, args[0]) if scalar_left else (args[0], c.result)
        else:
            base, exp = args
        if isinstance(oe, IntegerType):
            ops, value = _integer_pow(base, exp, oe)
        elif scalar_right and scalar == 2:
            square = arith.MulfOp(base, base)
            ops, value = [square], square.result
        else:
            power = math.PowFOp(base, exp)
            ops, value = [power], power.result
        return [*pre, *ops], value

    return d._pointwise(operands[:1 if scalar_left or scalar_right else 2],
                        meta, build, family='pow', promote=True)


def _integral_unary(operands, meta, name):
    from m2m.ir import decompositions as d
    elem = d._element_type_from_meta(meta)
    if not isinstance(elem, IntegerType) or elem.width.data == 1:
        raise ValueError(f'{name} is not implemented for Boolean inputs')

    def build(args, oe):
        if name in _INTEGER_IDENTITY:
            return [], args[0]
        zero = _constant(0, oe)
        neg = arith.SubiOp(zero.result, args[0])
        return [zero, neg], neg.result

    return d._pointwise(operands[:1], meta, build, family=name, promote=True)


def _bool_binary(operands, meta, name):
    from m2m.ir import decompositions as d
    alpha = meta.get('_fx_kwargs', {}).get('alpha', d._fx_arg(meta, 2, 1))

    def build(args, oe):
        if name == 'add' and not bool(alpha):
            return [], args[0]
        op = (arith.OrIOp if name == 'add' else arith.AndIOp)(*args)
        return [op], op.result

    # _binary_elementwise's literal scalar path excludes bool. A scalar bool
    # is a wrapped number, so materialize its truth value in the result dtype.
    if len(operands) == 1:
        scalar = d._fx_arg(meta, 1)
        rt = TensorType(d._element_type_from_meta(meta), d._shape_of(operands[0]))
        pre, rhs = d._splat_scalar(bool(scalar), rt)
        result = d._pointwise([operands[0], rhs], meta, build, family=name, promote=True)
        if result is not None:
            result.ops = [*pre, *result.ops]
        return result
    return d._pointwise(operands[:2], meta, build, family=name, promote=True)


def _division(operands, meta, target):
    from m2m.ir import decompositions as d
    from xdsl.dialects.builtin import FloatAttr
    from xdsl.dialects.cf import AssertOp

    mode = ('floor' if 'floor_divide' in target else
            meta.get('_fx_kwargs', {}).get('rounding_mode', d._fx_arg(meta, 2)))
    if mode not in (None, 'floor', 'trunc'):
        raise ValueError(f'unsupported rounding mode: {mode}')

    def build(args, oe):
        a, b = args
        if isinstance(oe, IntegerType):
            if oe.width.data == 1:
                raise ValueError('integer division is not implemented for bool')
            zero, one = _constant(0, oe), _constant(1, oe)
            nonzero = arith.CmpiOp(b, zero.result, 'ne')
            check = AssertOp(nonzero.result, 'ZeroDivisionError')
            minimum = _constant(-(1 << (oe.width.data - 1)), oe)
            minus_one = _constant(-1, oe)
            low = arith.CmpiOp(a, minimum.result, 'eq')
            negative_one = arith.CmpiOp(b, minus_one.result, 'eq')
            overflow = arith.AndIOp(low.result, negative_one.result)
            # ATen floor division wraps INT_MIN/-1. Avoid even evaluating an
            # overflowing MLIR division: a later select cannot remove poison.
            safe = arith.SelectOp(overflow.result, one.result, b)
            quotient = arith.DivSIOp(a, safe.result)
            ops = [zero, one, nonzero, check, minimum, minus_one, low,
                   negative_one, overflow, safe, quotient]
            if mode == 'trunc' and oe.width.data >= 32:
                # CPU truncating division traps for INT_MIN/-1 at these widths.
                true = arith.ConstantOp.from_int_and_width(1, 1)
                valid = arith.XOrIOp(overflow.result, true.result)
                ops += [true, valid, AssertOp(valid.result, 'integer division overflow')]
            value = quotient.result
            if mode == 'floor':
                remainder = arith.RemSIOp(a, safe.result)
                nonexact = arith.CmpiOp(remainder.result, zero.result, 'ne')
                a_negative = arith.CmpiOp(a, zero.result, 'slt')
                b_negative = arith.CmpiOp(b, zero.result, 'slt')
                different = arith.XOrIOp(a_negative.result, b_negative.result)
                adjust = arith.AndIOp(nonexact.result, different.result)
                lower = arith.SubiOp(value, one.result)
                rounded = arith.SelectOp(adjust.result, lower.result, value)
                ops += [remainder, nonexact, a_negative, b_negative, different,
                        adjust, lower, rounded]
                value = rounded.result
            result = arith.SelectOp(overflow.result, a, value)
            return [*ops, result], result.result

        basic = arith.DivfOp(a, b)
        if mode is None:
            return [basic], basic.result
        if mode == 'trunc':
            rounded = math.TruncOp(basic.result)
            return [basic, rounded], rounded.result
        # c10::div_floor_floating: remainder correction and rounded quotient.
        # floor(a/b) alone disagrees near multiples (e.g. 1 // 0.1).
        zero, one, half = (arith.ConstantOp(FloatAttr(v, oe)) for v in (0., 1., .5))
        # xDSL has no wrapper for this upstream arith operation yet.
        from xdsl.dialects.builtin import UnregisteredOp
        rem = UnregisteredOp.with_name('arith.remf').create(operands=[a, b], result_types=[oe])
        rem_value = rem.results[0]
        numerator = arith.SubfOp(a, rem_value)
        quotient = arith.DivfOp(numerator.result, b)
        nonexact = arith.CmpfOp(rem_value, zero.result, 'une')
        b_negative = arith.CmpfOp(b, zero.result, 'olt')
        rem_negative = arith.CmpfOp(rem_value, zero.result, 'olt')
        different = arith.XOrIOp(b_negative.result, rem_negative.result)
        adjust = arith.AndIOp(nonexact.result, different.result)
        lower = arith.SubfOp(quotient.result, one.result)
        div = arith.SelectOp(adjust.result, lower.result, quotient.result)
        floor = math.FloorOp(div.result)
        residual = arith.SubfOp(div.result, floor.result)
        round_up = arith.CmpfOp(residual.result, half.result, 'ogt')
        upper = arith.AddfOp(floor.result, one.result)
        rounded = arith.SelectOp(round_up.result, upper.result, floor.result)
        div_zero = arith.CmpfOp(div.result, zero.result, 'oeq')
        bits = IntegerType(oe.bitwidth)
        sign_mask = _constant(-(1 << (oe.bitwidth - 1)), bits)
        basic_bits = arith.BitcastOp(basic.result, bits)
        sign_bits = arith.AndIOp(basic_bits.result, sign_mask.result)
        signed_zero = arith.BitcastOp(sign_bits.result, oe)
        signed = arith.SelectOp(div_zero.result, signed_zero.result, rounded.result)
        b_zero = arith.CmpfOp(b, zero.result, 'oeq')
        result = arith.SelectOp(b_zero.result, basic.result, signed.result)
        return [basic, zero, one, half, rem, numerator, quotient, nonexact,
                b_negative, rem_negative, different, adjust, lower, div, floor,
                residual, round_up, upper, rounded, div_zero, sign_mask,
                basic_bits, sign_bits, signed_zero, signed,
                b_zero, result], result.result

    if len(operands) == 1:
        elem = d._element_type_from_meta(meta)
        scalar = d._fx_arg(meta, 1)
        if isinstance(elem, IntegerType):
            scalar = d._wrap_int(int(scalar), elem.width.data)
        pre, rhs = d._splat_scalar(scalar, TensorType(elem, d._shape_of(operands[0])))
        result = d._pointwise([operands[0], rhs], meta, build, family='div', promote=True)
        if result is not None:
            result.ops = [*pre, *result.ops]
        return result
    return d._pointwise(operands[:2], meta, build, family='div', promote=True)


def _reduction(operands, meta, name, target):
    from m2m.ir import decompositions as d
    x = operands[0]
    shape = d._shape_of(x)
    if shape is None or any(n < 0 for n in shape):
        return None
    elem = d._element_type_from_meta(meta)
    cast = d._cast_tensor(x, shape, elem)
    if cast is None:
        return None
    ops, x = cast
    if not shape:
        result = d._pointwise([x], meta, lambda args, oe: ([], args[0]), family='reduce')
        if result is not None:
            result.ops = [*ops, *result.ops]
        return result
    dims = None if target in {'aten.sum.default', 'aten.prod.default'} else d._fx_arg(meta, 1)
    if dims is None or dims == [] or dims == ():
        dims = list(range(len(shape)))
    elif isinstance(dims, int):
        dims = [dims]
    dims = sorted({int(dim) % len(shape) for dim in dims})
    if name in {'sum', 'prod'}:
        integral = isinstance(elem, IntegerType)
        combine = ((arith.AddiOp if integral else arith.AddfOp) if name == 'sum' else
                   (arith.MuliOp if integral else arith.MulfOp))
        identity = 0 if name == 'sum' else 1
    elif elem.width.data == 1:
        combine = arith.AndIOp if name == 'amin' else arith.OrIOp
        identity = 1 if name == 'amin' else 0
    else:
        combine = arith.MinSIOp if name == 'amin' else arith.MaxSIOp
        identity = ((1 << (elem.width.data - 1)) - 1 if name == 'amin' else
                    -(1 << (elem.width.data - 1)))
    reduced_ops, value, reduced_shape = d._reduce(x, shape, dims, identity, combine, elem)
    ops += reduced_ops
    value = d._keepdim_reshape(ops, value, reduced_shape, d._static_shape(meta['val'].shape), elem)
    if value is None:
        return None
    rid = d._next_region_id('reduce')
    for op in ops:
        d._attach_region_id(op, rid)
    return d.DecompResult(ops=ops, result=value, region_ids=[rid], pattern_hint='reduce')


def _wrap(target, original):
    name = target.split('.')[1]

    @wraps(original)
    def lower(operands, meta, node_name):
        from m2m.ir import decompositions as d
        integer_input = any(isinstance(o.type, TensorType)
                            and isinstance(o.type.element_type, IntegerType) for o in operands)
        if not integer_input:
            return original(operands, meta, node_name)
        # Signless i8 also represents uint8, which is outside this signed/bool
        # path. Do not guess signedness from an MLIR element type.
        import torch
        for raw in meta.get('_fx_args', ()):
            value = raw.meta.get('val') if hasattr(raw, 'meta') else raw
            if getattr(value, 'dtype', None) in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
                return original(operands, meta, node_name)
        elem = d._element_type_from_meta(meta)
        if name == 'pow':
            result = _pow(operands, meta, target)
        elif name in {'div', 'floor_divide'}:
            result = _division(operands, meta, target)
        elif name in {'sum', 'prod', 'amin', 'amax'}:
            result = _reduction(operands, meta, name, target)
        elif name in _INTEGER_IDENTITY or name == 'neg':
            # These are dtype preserving; never round an integer through a float.
            if isinstance(d._t_elem(operands[0]), IntegerType):
                result = _integral_unary(operands, meta, name)
            else:
                return original(operands, meta, node_name)
        elif name in {'add', 'mul'} and isinstance(elem, IntegerType) and elem.width.data == 1:
            result = _bool_binary(operands, meta, name)
        elif name in _FLOATING_UNARY or name == 'mean':
            if not isinstance(elem, AnyFloat):
                raise ValueError(f'{name} requires a floating result dtype')
            # mean is only valid on integers with an explicit floating dtype;
            # the exported result captures that choice. Unary math instead uses
            # the capture-time default float. Cast before invoking float emitters.
            cast = d._cast_tensor(operands[0], d._shape_of(operands[0]), elem)
            if cast is None:
                result = None
            else:
                compute_meta = meta
                if name == 'mean':
                    # mean.default's schema position 1 is dtype, not dim.
                    # Normalize only the private compute call; source provenance
                    # stays attached to the original ATen target by the importer.
                    dims = None if target == 'aten.mean.default' else d._fx_arg(meta, 1)
                    if dims == [] or dims == ():
                        dims = None
                    keepdim = False if target == 'aten.mean.default' else d._fx_arg(meta, 2, False)
                    compute_meta = {**meta, '_fx_args': (d._fx_arg(meta, 0), dims, keepdim),
                                    '_fx_kwargs': {}, '_aten_target': 'aten.mean.dim'}
                result = original([cast[1], *operands[1:]], compute_meta, node_name)
                for op in cast[0]:
                    if result.region_ids:
                        d._attach_region_id(op, result.region_ids[0])
                result.ops = [*cast[0], *result.ops]
        else:
            return original(operands, meta, node_name)
        if result is not None:
            return result
        return d._opaque_decomp(d._opaque_name(meta, node_name), operands, meta,
                                 name, pattern_hint=name)

    return lower


def register_integer_promotion(table):
    """Add integer-aware paths without changing noninteger dispatch."""
    names = _FLOATING_UNARY | _INTEGER_IDENTITY | {
        'neg', 'pow', 'mean', 'add', 'mul', 'div', 'floor_divide', 'sum', 'prod', 'amin', 'amax',
    }
    for target, original in list(table.items()):
        if target.split('.')[1] in names:
            table[target] = _wrap(target, original)
