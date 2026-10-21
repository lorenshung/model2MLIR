"""ATen floating division modes, following c10/util/generic_math.h."""
from xdsl.dialects import arith, math
from xdsl.dialects.builtin import BFloat16Type, Float16Type, FloatAttr, IntegerAttr, IntegerType, UnregisteredOp, f32
from .builder import Builder
from .decompositions import _cast_scalar_arg


def floating_division_body(mode):
    def build(args, elem):
        # CPU kernels use float opmath for both half formats.
        wide = f32 if isinstance(elem, (Float16Type, BFloat16Type)) else elem
        emit = Builder()
        def constant(value):
            return emit(arith.ConstantOp(FloatAttr(value, wide)))
        values = []
        for arg in args:
            cast, value = _cast_scalar_arg(arg, wide)
            emit.ops.extend(cast); values.append(value)
        a, b = values
        quotient = emit(arith.DivfOp(a, b))
        if mode == "trunc":
            negative = emit(arith.CmpfOp(quotient, constant(0.), "olt"))
            floor = emit(math.FloorOp(quotient))
            ceil = emit(math.CeilOp(quotient))
            result = emit(arith.SelectOp(negative, ceil, floor))
        else:
            zero, one, half = constant(0.), constant(1.), constant(.5)
            mod = emit(UnregisteredOp.with_name("arith.remf").create(
                operands=[a, b], result_types=[wide]))
            div = emit(arith.DivfOp(emit(arith.SubfOp(a, mod)), b))
            nonzero_mod = emit(arith.CmpfOp(mod, zero, "une"))
            b_negative = emit(arith.CmpfOp(b, zero, "olt"))
            mod_negative = emit(arith.CmpfOp(mod, zero, "olt"))
            different = emit(arith.XOrIOp(b_negative, mod_negative))
            adjust = emit(arith.AndIOp(nonzero_mod, different))
            div = emit(arith.SelectOp(adjust, emit(arith.SubfOp(div, one)), div))
            floor = emit(math.FloorOp(div))
            snap = emit(arith.CmpfOp(emit(arith.SubfOp(div, floor)), half, "ogt"))
            rounded = emit(arith.SelectOp(snap, emit(arith.AddfOp(floor, one)), floor))
            integer = IntegerType(wide.bitwidth)
            bits = emit(arith.BitcastOp(quotient, integer))
            sign_mask = emit(arith.ConstantOp(IntegerAttr(-(1 << (wide.bitwidth - 1)), integer)))
            sign = emit(arith.AndIOp(bits, sign_mask))
            signed_zero = emit(arith.BitcastOp(sign, wide))
            nonzero_div = emit(arith.CmpfOp(div, zero, "une"))
            result = emit(arith.SelectOp(nonzero_div, rounded, signed_zero))
            zero_divisor = emit(arith.CmpfOp(b, zero, "oeq"))
            result = emit(arith.SelectOp(zero_divisor, quotient, result))
        cast, result = _cast_scalar_arg(result, elem)
        return [*emit.ops, *cast], result
    return build
