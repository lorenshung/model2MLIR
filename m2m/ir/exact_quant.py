"""Integer-only, correctly rounded scalar quantization builders.

No floating intermediate is used to decide a quantized code. IEEE inputs are
decoded into signed dyadic mantissas/exponents. Widths and shift bounds below
are mathematical bounds, independent of capture samples.
"""
from fractions import Fraction

from xdsl.dialects import arith
from xdsl.dialects.builtin import BFloat16Type, Float32Type, Float64Type, IntegerAttr, IntegerType


class Scalar:
    def __init__(self):
        self.ops = []

    def emit(self, op):
        self.ops.append(op)
        return op.results[0]

    def c(self, value, typ):
        return self.emit(arith.ConstantOp(IntegerAttr(int(value), typ)))

    def cast(self, value, typ, unsigned=False):
        if value.type == typ:
            return value
        if value.type.width.data > typ.width.data:
            return self.emit(arith.TruncIOp(value, typ))
        return self.emit((arith.ExtUIOp if unsigned else arith.ExtSIOp)(value, typ))

    def add(self, a, b): return self.emit(arith.AddiOp(a, b))
    def sub(self, a, b): return self.emit(arith.SubiOp(a, b))
    def mul(self, a, b): return self.emit(arith.MuliOp(a, b))
    def band(self, a, b): return self.emit(arith.AndIOp(a, b))
    def bor(self, a, b): return self.emit(arith.OrIOp(a, b))
    def cmp(self, a, b, pred): return self.emit(arith.CmpiOp(a, b, pred))
    def select(self, cond, yes, no): return self.emit(arith.SelectOp(cond, yes, no))
    def shl(self, a, b): return self.emit(arith.ShLIOp(a, b))
    def shr(self, a, b): return self.emit(arith.ShRUIOp(a, b))
    def neg(self, a): return self.sub(self.c(0, a.type), a)

    def assert_(self, cond, message):
        from xdsl.dialects.cf import AssertOp
        self.ops.append(AssertOp(cond, message))

    def decode(self, value):
        """Return exact signed mantissa, base-2 exponent, and finite flag."""
        if value.type not in (Float32Type(), Float64Type()):
            value = self.emit(arith.ExtFOp(value, Float32Type()))
        width = value.type.bitwidth
        mant_bits, exp_mask, offset = (23, 255, 150) if width == 32 else (52, 2047, 1075)
        i64 = IntegerType(64)
        raw = self.emit(arith.BitcastOp(value, IntegerType(width)))
        negative = self.cmp(raw, self.c(0, raw.type), "slt")
        bits = self.cast(raw, i64, unsigned=True)
        exponent = self.band(self.shr(bits, self.c(mant_bits, i64)), self.c(exp_mask, i64))
        fraction = self.band(bits, self.c((1 << mant_bits) - 1, i64))
        normal = self.cmp(exponent, self.c(0, i64), "ne")
        mantissa = self.select(normal, self.bor(fraction, self.c(1 << mant_bits, i64)), fraction)
        mantissa = self.select(negative, self.neg(mantissa), mantissa)
        power = self.sub(self.select(normal, exponent, self.c(1, i64)), self.c(offset, i64))
        finite = self.cmp(exponent, self.c(exp_mask, i64), "ne")
        return mantissa, power, finite

    def compare_dyadics(self, a, ae, b, be):
        """Compare a*2**ae with b*2**be; |a|, |b| < 2**63.

        Capping alignment at 64 preserves order even at extreme exponents:
        any nonzero shifted integer then dominates the unshifted integer.
        """
        i128, i64 = IntegerType(128), IntegerType(64)
        difference = self.sub(ae, be)
        shift_a = self.emit(arith.MinSIOp(self.emit(arith.MaxSIOp(difference, self.c(0, i64))), self.c(64, i64)))
        shift_b = self.emit(arith.MinSIOp(self.emit(arith.MaxSIOp(self.neg(difference), self.c(0, i64))), self.c(64, i64)))
        left = self.shl(self.cast(a, i128), self.cast(shift_a, i128))
        right = self.shl(self.cast(b, i128), self.cast(shift_b, i128))
        return self.cmp(left, right, "sgt"), self.cmp(left, right, "eq")

    def search(self, zp, low, high, compare):
        """Eight bisections implement saturated RNE over any byte range.

        Threshold parity is BEFORE the zero-point shift. RNE(x+zp) would be
        wrong at half ties for an odd asymmetric zero point.
        """
        i64 = IntegerType(64)
        zp = self.cast(zp, i64)
        lo, hi = self.sub(self.c(low, i64), zp), self.sub(self.c(high, i64), zp)
        for _ in range(8):
            mid = self.emit(arith.ShRSIOp(self.add(lo, hi), self.c(1, i64)))
            greater, equal = compare(mid)
            odd = self.cmp(self.band(mid, self.c(1, i64)), self.c(0, i64), "ne")
            above = self.band(self.bor(greater, self.band(equal, odd)), self.cmp(lo, hi, "slt"))
            lo = self.select(above, self.add(mid, self.c(1, i64)), lo)
            hi = self.select(above, hi, mid)
        return self.add(lo, zp)

    def to_float(self, integer, exponent, out_type):
        """Single RNE from a signed dyadic to IEEE, including subnormals."""
        target_width = out_type.bitwidth
        precision, bias, emin, emax = (
            (8, 127, -126, 127) if isinstance(out_type, BFloat16Type) else
            (11, 15, -14, 15) if target_width == 16 else
            (24, 127, -126, 127) if target_width == 32 else (53, 1023, -1022, 1023))
        # Supply normalization bits even when the input is a small integer.
        typ = IntegerType(integer.type.width.data + precision + 2)
        integer = self.shl(self.cast(integer, typ), self.c(precision, typ))
        exponent = (self.c(exponent - precision, typ) if isinstance(exponent, int)
                    else self.sub(self.cast(exponent, typ), self.c(precision, typ)))
        width = typ.width.data
        zero, one = self.c(0, typ), self.c(1, typ)
        negative = self.cmp(integer, zero, "slt")
        magnitude = self.select(negative, self.neg(integer), integer)
        # Binary bit-length search uses only arith, for arbitrary integer widths.
        # math.ctlz's upstream legalization accepts only machine integer widths.
        probe, length = magnitude, zero
        for step in (1 << k for k in reversed(range((width - 1).bit_length()))):
            large = self.cmp(probe, self.c(1 << step, typ), "uge")
            probe = self.select(large, self.shr(probe, self.c(step, typ)), probe)
            length = self.add(length, self.select(large, self.c(step, typ), zero))
        length = self.add(length, self.cast(self.cmp(magnitude, zero, "ne"), typ, unsigned=True))
        # Drop enough bits for normal precision OR the smallest subnormal unit.
        drop = self.emit(arith.MaxSIOp(self.sub(length, self.c(precision, typ)),
                                       self.sub(self.c(emin - precision + 1, typ), exponent)))
        drop = self.emit(arith.MaxSIOp(drop, zero))
        bounded = self.emit(arith.MinSIOp(drop, self.c(width - 1, typ)))
        truncated = self.shr(magnitude, bounded)
        mask = self.sub(self.shl(one, bounded), one)
        remainder = self.band(magnitude, mask)
        half_shift = self.emit(arith.MaxSIOp(self.sub(bounded, one), zero))
        half = self.shl(one, half_shift)
        odd = self.cmp(self.band(truncated, one), zero, "ne")
        up = self.bor(self.cmp(remainder, half, "ugt"),
                      self.band(self.cmp(remainder, half, "eq"), odd))
        up = self.band(up, self.cmp(drop, zero, "sgt"))
        rounded = self.add(truncated, self.cast(up, typ, unsigned=True))
        rounded = self.select(self.cmp(drop, self.c(width, typ), "sge"), zero, rounded)
        # Carry can add one mantissa bit.
        carry = self.cmp(rounded, self.c(1 << precision, typ), "uge")
        rounded = self.select(carry, self.shr(rounded, one), rounded)
        unit = self.add(exponent, self.add(drop, self.cast(carry, typ, unsigned=True)))
        normal = self.cmp(rounded, self.c(1 << (precision - 1), typ), "uge")
        exp_field = self.add(unit, self.c(precision - 1 + bias, typ))
        bits = self.select(normal, self.add(self.shl(exp_field, self.c(precision - 1, typ)),
                                             self.sub(rounded, self.c(1 << (precision - 1), typ))), rounded)
        overflow = self.band(normal, self.cmp(exp_field, self.c(emax + bias, typ), "sgt"))
        infinity = ((1 << (target_width - precision)) - 1) << (precision - 1)
        bits = self.select(overflow, self.c(infinity, typ), bits)
        bits = self.bor(bits, self.select(negative, self.c(1 << (target_width - 1), typ), zero))
        bits = self.cast(bits, IntegerType(target_width), unsigned=True)
        return self.emit(arith.BitcastOp(bits, out_type))


def quantize_scalar(builder, x, scale, zp, low, high, out_type, unsigned=False):
    i64 = IntegerType(64)
    xm, xe, finite = builder.decode(x)
    sm, se, sf = builder.decode(scale)
    builder.assert_(builder.band(sf, builder.cmp(sm, builder.c(0, i64), "sgt")),
                    "exact quantize requires a finite positive scale")
    # Infinity saturates by sign through the same comparisons. NaN has no
    # exact-rational meaning and must never reach an FP-to-int conversion.
    builder.assert_(builder.emit(arith.CmpfOp(x, x, "ord")),
                    "exact quantize requires a non-NaN input")
    zp_low, zp_high = (0, 255) if unsigned else (-128, 127)
    builder.assert_(builder.band(builder.cmp(zp, builder.c(zp_low, zp.type), "sge"),
                                 builder.cmp(zp, builder.c(zp_high, zp.type), "sle")),
                    "quantize zero point outside byte storage range")
    def compare(mid):
        threshold = builder.mul(builder.add(builder.mul(mid, builder.c(2, i64)), builder.c(1, i64)), sm)
        return builder.compare_dyadics(xm, xe, threshold, builder.sub(se, builder.c(1, i64)))
    code = builder.search(zp, low, high, compare)
    endpoint = builder.select(builder.cmp(xm, builder.c(0, i64), "slt"),
                              builder.c(low, i64), builder.c(high, i64))
    code = builder.select(finite, code, endpoint)
    return builder.cast(code, out_type)


def readout_plan(scales, beta, bias_type, output_scale):
    """Prove a finite integer width for every possible runtime operand."""
    from m2m.capture.exact_quant import rational
    fractions = [Fraction(s) for s in scales]
    if not fractions or any(s <= 0 for s in fractions):
        raise ValueError("exact readout requires positive finite scales")
    base = min(-s.denominator.bit_length() + 1 for s in fractions)
    bias_unit = None
    beta_coeff = 0
    if bias_type is not None:
        if bias_type != Float32Type():
            raise ValueError("exact readout requires float32 bias")
        bias_unit = -149
        bf = rational(beta)
        base = min(base, bias_unit - bf.denominator.bit_length() + 1)
        beta_coeff = int(bf * Fraction(2) ** (bias_unit - base))
    coeffs = [int(s / Fraction(2) ** base) for s in fractions]
    # int32 input; a finite f32 bias is < 2**128 or <2**277 subnormal units.
    bound = max(abs(v) for v in coeffs) * (1 << 31) + abs(beta_coeff) * (1 << 277)
    ratio = Fraction(2) ** base / (Fraction(output_scale) if output_scale is not None else 1)
    width = max(64, (2 * bound * abs(ratio.numerator)).bit_length() + 2,
                (1024 * ratio.denominator).bit_length() + 2, bound.bit_length() + 2)
    # Avoid unbounded compiler artifacts for pathological exponents. This is a
    # declaration-wide refusal, never a test of the calibration sample.
    if width > 4096:
        raise ValueError("exact readout requires more than 4096 integer bits")
    return base, coeffs, beta_coeff, bias_unit, ratio, width


def build_readout(acc, scales, *, bias=None, beta=1.0, output_scale=None,
                  zero_point=0, quant_min=-128, quant_max=127, relu=False, unsigned=False):
    """Lower exact affine readout using integer arithmetic and one final RNE."""
    from xdsl.dialects.builtin import IndexType, TensorType
    from xdsl.dialects.linalg import IndexOp
    from m2m.ir.decompositions import _elementwise, _broadcast_map
    from xdsl.ir.affine import AffineMap
    if beta == 0:
        bias = None  # addmm beta=0 ignores self, including NaN/inf.
    if output_scale is not None:
        bounds = (0, 255) if unsigned else (-128, 127)
        if not bounds[0] <= quant_min < quant_max <= bounds[1] or not bounds[0] <= zero_point <= bounds[1]:
            raise ValueError("invalid readout range or zero point for byte storage")
        if not Fraction(output_scale) > 0:
            raise ValueError("readout output scale must be finite and positive")
    base, coeffs, beta_coeff, bias_unit, ratio, width = readout_plan(
        scales, beta, None if bias is None else bias.type.element_type, output_scale)
    typ, i64 = IntegerType(width), IntegerType(64)
    shape = list(acc.type.get_shape())
    if len(scales) > 1 and (not shape or shape[-1] != len(scales)):
        raise ValueError("exact readout scales must match the last axis")
    out_type = Float32Type() if output_scale is None else IntegerType(8)

    def build(args, oe):
        b = Scalar()
        coeff = b.c(coeffs[0], typ)
        if len(set(coeffs)) > 1:
            index = b.emit(IndexOp(len(shape) - 1))
            for ordinal, value in enumerate(coeffs[1:], 1):
                match = b.cmp(index, b.c(ordinal, IndexType()), "eq")
                coeff = b.select(match, b.c(value, typ), coeff)
        value = b.mul(b.cast(args[0], typ), coeff)
        if bias is not None:
            bm, be, finite = b.decode(args[1])
            b.assert_(finite, "exact readout requires finite bias")
            shift = b.cast(b.sub(be, b.c(bias_unit, i64)), typ)
            fixed = b.shl(b.cast(bm, typ), shift)
            value = b.add(value, b.mul(fixed, b.c(beta_coeff, typ)))
        if output_scale is None:
            result = b.to_float(value, base, oe)
        else:
            lhs = b.mul(value, b.c(2 * ratio.numerator, typ))
            def compare(mid):
                threshold = b.mul(b.cast(b.add(b.mul(mid, b.c(2, i64)), b.c(1, i64)), typ),
                                  b.c(ratio.denominator, typ))
                return b.cmp(lhs, threshold, "sgt"), b.cmp(lhs, threshold, "eq")
            low = min(quant_max, max(quant_min, zero_point)) if relu else quant_min
            code = b.search(b.c(zero_point, i64), low, quant_max, compare)
            result = b.cast(code, oe)
        return b.ops, result

    maps = [AffineMap.identity(len(shape))]
    if bias is not None:
        mapping = _broadcast_map(bias.type.get_shape(), shape)
        if mapping is None:
            raise ValueError("exact readout bias does not broadcast to accumulator")
        maps.append(mapping)
    return _elementwise([acc, *([bias] if bias is not None else [])], TensorType(out_type, shape), build,
                        input_maps=maps)
