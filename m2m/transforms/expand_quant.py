"""Expand quant_ext.* ops to pure standard dialects (linalg.generic + arith + math).

The default output keeps QDQ as `quant_ext.quantize_*` / `quant_ext.dequantize_*` (the one
non-core op family). For a "purely upstream-MLIR" artifact, this pass lowers those to
`linalg.generic` + `arith`/`math` -- no quant dialect, no custom dialect, just core compute
dialects:

* dequantize: ``(sitofp(q) - zp) * scale`` (``build_dequantize_body``);
* quantize: PT2E's ``clamp(roundeven(x * (1 / scale)) + zp, quant_min, quant_max)`` with the
  reciprocal taken first, then ``fptosi`` (``build_quantize_body``). A literal per-tensor scale
  (``prov.quant.scale``) gives the exact f32 reciprocal of the double scale, as PT2E computes.

Each expansion keeps the source op's ``prov.quant.*`` attributes.
"""

from __future__ import annotations

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.rewriter import InsertPoint, Rewriter

from m2m.ir.decompositions import _attach_region_id, _next_region_id, build_dequantize_body
from m2m.ir.quant.ops import DequantizePerChannelOp, DequantizePerTensorOp, QuantizePerChannelOp, QuantizePerTensorOp


def _expand_dequant(op):
    out_t = op.results[0].type
    out_shape = list(out_t.get_shape())
    axis = op.axis.value.data if getattr(op, "axis", None) is not None else 0
    return build_dequantize_body(op.input, op.scales if hasattr(op, "scales") else op.scale,
                                 op.zero_points if hasattr(op, "zero_points") else op.zero_point,
                                 axis=axis, out_shape=out_shape, out_elem=out_t.element_type)


def build_quantize_body(inp, scales, zps, *, axis, out_shape, out_elem, quant_min, quant_max,
                        literal_scale=None, literal_zero_point=None):
    """Standard-dialect PT2E quantize as one linalg.generic, or None for dynamic shapes."""
    from xdsl.dialects import arith
    from xdsl.dialects import math as mlir_math
    from xdsl.dialects.builtin import Float32Type, FloatAttr, IntegerType, TensorType
    from xdsl.ir.affine import AffineExpr, AffineMap

    from m2m.ir.decompositions import _elementwise

    if any(d < 0 for d in out_shape):
        return None
    rank = len(out_shape)
    id_map = AffineMap.identity(rank)
    chan_map = AffineMap(rank, 0, (AffineExpr.dimension(axis % rank),)) if rank else AffineMap(0, 0, ())
    f32 = Float32Type()

    def build(args, oe):
        x, scale, zero_point = args
        ops = []
        if literal_scale is not None:
            inverse = arith.ConstantOp(FloatAttr(1.0 / literal_scale, f32), f32)
            ops.append(inverse)
        else:
            if scale.type != f32:
                widened = arith.ExtFOp(scale, f32) if scale.type.bitwidth < 32 else arith.TruncFOp(scale, f32)
                ops.append(widened)
                scale = widened.result
            one = arith.ConstantOp(FloatAttr(1.0, f32), f32)
            inverse = arith.DivfOp(one.result, scale)
            ops += [one, inverse]
        if x.type != f32:
            widened = arith.ExtFOp(x, f32)
            ops.append(widened)
            x = widened.result
        scaled = arith.MulfOp(x, inverse.results[0])
        rounded = mlir_math.RoundEvenOp(scaled.result)
        ops += [scaled, rounded]
        if literal_zero_point is not None:
            zp_op = arith.ConstantOp(FloatAttr(float(literal_zero_point), f32), f32)
        elif isinstance(zero_point.type, IntegerType):
            zp_op = arith.SIToFPOp(zero_point, f32)
        elif zero_point.type != f32:
            zp_op = arith.TruncFOp(zero_point, f32) if zero_point.type.bitwidth > 32 else arith.ExtFOp(zero_point, f32)
        else:
            zp_op = None
        if zp_op is not None:
            ops.append(zp_op)
        shifted = arith.AddfOp(rounded.result, zp_op.results[0] if zp_op is not None else zero_point)
        lo = arith.ConstantOp(FloatAttr(float(quant_min), f32), f32)
        hi = arith.ConstantOp(FloatAttr(float(quant_max), f32), f32)
        clamp_lo = arith.MaximumfOp(shifted.result, lo.result)
        clamp_hi = arith.MinimumfOp(clamp_lo.result, hi.result)
        narrow = arith.FPToSIOp(clamp_hi.result, oe)
        ops += [shifted, lo, hi, clamp_lo, clamp_hi, narrow]
        return ops, narrow.results[0]

    return _elementwise([inp, scales, zps], TensorType(out_elem, out_shape), build,
                        input_maps=[id_map, _qparam_map(scales, rank, chan_map), _qparam_map(zps, rank, chan_map)])


def _qparam_map(value, rank, chan_map):
    """A rank-0 (per-tensor) qparam is read with the empty map; a vector along the axis."""
    from xdsl.ir.affine import AffineMap

    return AffineMap(rank, 0, ()) if not list(value.type.get_shape()) else chan_map


def _expand_quantize(op):
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr

    out_t = op.results[0].type
    out_shape = list(out_t.get_shape())
    per_channel = isinstance(op, QuantizePerChannelOp)
    axis = op.axis.value.data if per_channel else 0
    if op.quant_min is None or op.quant_max is None:
        return None
    literal_scale = literal_zero_point = None
    if not per_channel:
        scale_attr = op.attributes.get("prov.quant.scale")
        zp_attr = op.attributes.get("prov.quant.zero_point")
        if isinstance(scale_attr, FloatAttr) and isinstance(zp_attr, IntegerAttr):
            literal_scale = float(scale_attr.value.data)
            literal_zero_point = int(zp_attr.value.data)
    if not per_channel and (list(op.scale.type.get_shape()) or list(op.zero_point.type.get_shape())):
        return None  # A per-tensor qparam must be rank 0.
    return build_quantize_body(op.input, op.scales if per_channel else op.scale,
                               op.zero_points if per_channel else op.zero_point,
                               axis=axis, out_shape=out_shape, out_elem=out_t.element_type,
                               quant_min=op.quant_min.value.data, quant_max=op.quant_max.value.data,
                               literal_scale=literal_scale, literal_zero_point=literal_zero_point)


QUANT_EXPANDERS = {
    DequantizePerChannelOp: _expand_dequant,
    DequantizePerTensorOp: _expand_dequant,
    QuantizePerChannelOp: _expand_quantize,
    QuantizePerTensorOp: _expand_quantize,
}


def expand_quant_to_standard(module: ModuleOp) -> ModuleOp:
    """Lower every quant_ext quantize/dequantize op to standard linalg+arith+math, in place.
    Returns the module. Ops it can't lower (dynamic shapes) are left untouched."""
    for op in list(module.walk()):
        handler = QUANT_EXPANDERS.get(type(op))
        if handler is None:
            continue
        built = handler(op)
        if built is None:
            continue
        ops, result = built
        kind = "quantize" if handler is _expand_quantize else "dequantize"
        rid = _next_region_id(kind)
        quant_attrs = {key: value for key, value in op.attributes.items() if key.startswith("prov.quant.")}
        for o in ops:
            from m2m.capture.trace import copy_mlir_sources

            copy_mlir_sources(o, op, role="lowering")
            _attach_region_id(o, rid)
            o.attributes["prov.op"] = StringAttr(kind)
            o.attributes["prov.family"] = StringAttr("quantize")
            for key in ("prov.aten",):
                if key in op.attributes:
                    o.attributes[key] = op.attributes[key]
            if o.name == "linalg.generic":
                o.attributes.update(quant_attrs)
        Rewriter.insert_op(ops, InsertPoint.before(op))
        op.results[0].replace_all_uses_with(result)
        Rewriter.erase_op(op)
    return module
