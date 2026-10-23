"""Emit exact scale provenance according to :mod:`m2m.ir.quant_schema`."""

from fractions import Fraction
import math
import struct


def fp32(value):
    """Round exact rationals/integers directly; narrow stored binary64 values."""
    if isinstance(value, (Fraction, int)):
        from m2m.capture.exact_quant import rounded_float

        return rounded_float(Fraction(value))
    try:
        return struct.unpack("f", struct.pack("f", float(value)))[0]
    except OverflowError:
        return -math.inf if value < 0 else math.inf


def canonical_scales(values):
    return all(math.isfinite(rounded := fp32(v)) and rounded == v for v in values)


def scale_attributes(values, *, source_scales=None):
    """Rational coefficients are authoritative; fp32 ``scale`` is optional info.

    Host utilities may explicitly preserve noncanonical binary64 scales. Mark
    these host-only; accelerator capture canonicalizes the graph before import.
    """
    from xdsl.dialects.builtin import ArrayAttr, Float32Type, FloatAttr, StringAttr

    values = [Fraction(v) for v in values]
    attrs = {
        "prov.quant.scale_numerator": ArrayAttr([StringAttr(str(v.numerator)) for v in values]),
        "prov.quant.scale_denominator": ArrayAttr([StringAttr(str(v.denominator)) for v in values]),
    }
    approximations = [fp32(v) for v in values]
    if all(math.isfinite(v) for v in approximations):
        scales = [FloatAttr(v, Float32Type()) for v in approximations]
        attrs["prov.quant.scale"] = scales[0] if len(scales) == 1 else ArrayAttr(scales)
    if not canonical_scales(values if source_scales is None else source_scales):
        attrs["prov.quant.host_only"] = StringAttr("true")
    return attrs
