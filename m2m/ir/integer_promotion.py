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

from m2m.ir.builder import Builder


_FLOATING_UNARY = {
    'acos', 'asin', 'atan', 'acosh', 'asinh', 'atanh', 'sin', 'cos', 'tan',
    'sinh', 'cosh', 'tanh', 'exp', 'exp2', 'expm1', 'log', 'log1p', 'log2',
    'log10', 'sqrt', 'rsqrt', 'sigmoid', 'erf', 'reciprocal',
}
_INTEGER_IDENTITY = {'floor', 'ceil', 'round', 'trunc'}


def _constant(value, elem):
    from m2m.ir.decompositions import _wrap_int
    return arith.ConstantOp(IntegerAttr(_wrap_int(value, elem.width.data), elem))


def _integer_pow(b, base, exponent, elem):
    """ATen's integral powi: squaring modulo 2**width; tensor negatives.

    A positive exponent takes at most width-1 iterations. Negative exponents
    produce zero except for bases +/-1, retaining parity even at INT_MIN.
    """
    zero, one, minus_one = (b(_constant(v, elem)) for v in (0, 1, -1))
    negative = b(arith.CmpiOp(exponent, zero, 'slt'))
    start_exp = b(arith.SelectOp(negative, zero, exponent))
    before = Block(arg_types=[elem, elem, elem])
    test = Builder(before)
    test(scf.ConditionOp(test(arith.CmpiOp(before.args[0], zero, 'sgt')), *before.args))
    after = Block(arg_types=[elem, elem, elem])
    e, factor, acc = after.args
    step = Builder(after)
    odd = step(arith.CmpiOp(step(arith.AndIOp(e, one)), zero, 'ne'))
    chosen = step(arith.SelectOp(odd, step(arith.MuliOp(acc, factor)), acc))
    squared = step(arith.MuliOp(factor, factor))
    step(scf.YieldOp(step(arith.ShRUIOp(e, one)), squared, chosen))
    loop = scf.WhileOp([start_exp, base, one], [elem]*3, Region(before), Region(after))
    b(loop)
    is_one = b(arith.CmpiOp(base, one, 'eq'))
    is_minus_one = b(arith.CmpiOp(base, minus_one, 'eq'))
    is_odd = b(arith.CmpiOp(b(arith.AndIOp(exponent, one)), zero, 'ne'))
    sign = b(arith.SelectOp(is_odd, minus_one, one))
    unit = b(arith.SelectOp(is_one, one, zero))
    negative_result = b(arith.SelectOp(is_minus_one, sign, unit))
    return b(arith.SelectOp(negative, negative_result, loop.results[2]))


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
        b = Builder()
        if scalar_left or scalar_right:
            if isinstance(oe, IntegerType):
                c = b(_constant(int(scalar), oe))
            else:
                from xdsl.dialects.builtin import FloatAttr
                c = b(arith.ConstantOp(FloatAttr(float(scalar), oe)))
            base, exp = (c, args[0]) if scalar_left else (args[0], c)
        else:
            base, exp = args
        if isinstance(oe, IntegerType):
            value = _integer_pow(b, base, exp, oe)
        elif scalar_right and scalar == 2:
            value = b(arith.MulfOp(base, base))
        else:
            value = b(math.PowFOp(base, exp))
        return b.ops, value

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
        b = Builder()
        return b.ops, b(arith.SubiOp(b(_constant(0, oe)), args[0]))

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
        put = Builder()
        if isinstance(oe, IntegerType):
            if oe.width.data == 1:
                raise ValueError('integer division is not implemented for bool')
            zero, one = put(_constant(0, oe)), put(_constant(1, oe))
            put(AssertOp(put(arith.CmpiOp(b, zero, 'ne')), 'ZeroDivisionError'))
            minimum, minus_one = put(_constant(-(1 << (oe.width.data - 1)), oe)), put(_constant(-1, oe))
            overflow = put(arith.AndIOp(put(arith.CmpiOp(a, minimum, 'eq')), put(arith.CmpiOp(b, minus_one, 'eq'))))
            # ATen floor division wraps INT_MIN/-1. Avoid even evaluating an
            # overflowing MLIR division: a later select cannot remove poison.
            safe = put(arith.SelectOp(overflow, one, b))
            quotient = put((arith.FloorDivSIOp if mode == 'floor' else arith.DivSIOp)(a, safe))
            if mode == 'trunc' and oe.width.data >= 32:
                # CPU truncating division traps for INT_MIN/-1 at these widths.
                valid = put(arith.XOrIOp(overflow, put(arith.ConstantOp.from_int_and_width(1, 1))))
                put(AssertOp(valid, 'integer division overflow'))
            return put.ops, put(arith.SelectOp(overflow, a, quotient))

        basic = put(arith.DivfOp(a, b))
        if mode is None:
            return put.ops, basic
        if mode == 'trunc':
            return put.ops, put(math.TruncOp(basic))
        # c10::div_floor_floating: remainder correction and rounded quotient.
        # floor(a/b) alone disagrees near multiples (e.g. 1 // 0.1).
        zero, one, half = (put(arith.ConstantOp(FloatAttr(v, oe))) for v in (0., 1., .5))
        # xDSL has no wrapper for this upstream arith operation yet.
        from xdsl.dialects.builtin import UnregisteredOp
        rem = put(UnregisteredOp.with_name('arith.remf').create(operands=[a, b], result_types=[oe]))
        quotient = put(arith.DivfOp(put(arith.SubfOp(a, rem)), b))
        nonexact = put(arith.CmpfOp(rem, zero, 'une'))
        different = put(arith.XOrIOp(put(arith.CmpfOp(b, zero, 'olt')), put(arith.CmpfOp(rem, zero, 'olt'))))
        adjust = put(arith.AndIOp(nonexact, different))
        div = put(arith.SelectOp(adjust, put(arith.SubfOp(quotient, one)), quotient))
        floor = put(math.FloorOp(div))
        round_up = put(arith.CmpfOp(put(arith.SubfOp(div, floor)), half, 'ogt'))
        rounded = put(arith.SelectOp(round_up, put(arith.AddfOp(floor, one)), floor))
        div_zero = put(arith.CmpfOp(div, zero, 'oeq'))
        bits = IntegerType(oe.bitwidth)
        sign_mask = put(_constant(-(1 << (oe.bitwidth - 1)), bits))
        signed_zero = put(arith.BitcastOp(put(arith.AndIOp(put(arith.BitcastOp(basic, bits)), sign_mask)), oe))
        signed = put(arith.SelectOp(div_zero, signed_zero, rounded))
        return put.ops, put(arith.SelectOp(put(arith.CmpfOp(b, zero, 'oeq')), basic, signed))

    if isinstance(d._element_type_from_meta(meta), IntegerType):
        # Locate wrapped numbers by schema position, including bool scalars.
        # Keep constants inside the generic so c // tensor preserves operand
        # order without adding a second, splatted tensor input.
        if len(operands) == 1:
            scalar_meta = {**meta, '_fx_args': tuple(
                int(arg) if isinstance(arg, bool) else arg
                for arg in meta.get('_fx_args', ())
            )}
            found = d._scalar_position(scalar_meta)
            if found is None:
                return None
            position, scalar = found

            def scalar_build(args, oe):
                constant = _constant(d._wrap_int(scalar, oe.width.data), oe)
                pair = [constant.result, args[0]] if position == 0 else [args[0], constant.result]
                ops, result = build(pair, oe)
                return [constant, *ops], result

            result = d._pointwise(operands, meta, scalar_build, family='elementwise', promote=True)
        else:
            result = d._pointwise(operands[:2], meta, build, family='elementwise', promote=True)
        if result is not None:
            result.pattern_hint = 'floor_divide' if mode == 'floor' else 'trunc_divide'
        return result

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
        elif name in _INTEGER_IDENTITY or name == 'neg':
            # These are dtype preserving; never round an integer through a float.
            if isinstance(d._t_elem(operands[0]), IntegerType):
                result = _integral_unary(operands, meta, name)
            else:
                return original(operands, meta, node_name)
        elif name in {'add', 'mul'} and isinstance(elem, IntegerType) and elem.width.data == 1:
            result = _bool_binary(operands, meta, name)
        elif name in _FLOATING_UNARY:
            if not isinstance(elem, AnyFloat):
                raise ValueError(f'{name} requires a floating result dtype')
            # Unary math uses the capture-time default floating dtype.
            # Cast before invoking float emitters.
            cast = d._cast_tensor(operands[0], d._shape_of(operands[0]), elem)
            if cast is None:
                result = None
            else:
                result = original([cast[1], *operands[1:]], meta, node_name)
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
        'neg', 'pow', 'add', 'mul', 'div', 'floor_divide',
    }
    for target, original in list(table.items()):
        if target.split('.')[1] in names:
            table[target] = _wrap(target, original)
