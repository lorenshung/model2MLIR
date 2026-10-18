"""Expand byte quant_ext Q/DQ to standard linalg, tensor and arith operations.

Quantize computes clamp(RNE(x / scale) + zero_point) in exact arithmetic.
Dequantize computes (code - zero_point) * scale with one final IEEE rounding.
Scale literals keep their original f64 value. Tensor scales keep their stored
f32/f64 value, including per-channel parameters. Signedness travels separately
from MLIR's signless byte storage type. No float-to-byte conversion is emitted.

Expansions preserve source provenance and mark prov.quant.arithmetic = "exact".
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
                                 axis=axis, out_shape=out_shape, out_elem=out_t.element_type,
                                 unsigned=getattr(op.attributes.get("prov.quant.storage_dtype"), "data", None) == "uint8"
                                          or getattr(op.input_dtype, "data", None) == "uint8",
                                 zero_point_unsigned=getattr(op.attributes.get("prov.quant.zero_point_dtype"), "data", None) == "uint8")


def build_quantize_body(inp, scales, zps, *, axis, out_shape, out_elem, quant_min, quant_max,
                        literal_scale=None, literal_zero_point=None, unsigned=False, zero_point_unsigned=False):
    """Standard-dialect PT2E quantize as one linalg.generic, or None for dynamic shapes."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import Float64Type, FloatAttr, IntegerType, TensorType
    from xdsl.ir.affine import AffineExpr, AffineMap

    from m2m.ir.decompositions import _elementwise
    from m2m.ir.exact_quant import Scalar, quantize_scalar

    if any(d < 0 for d in out_shape):
        return None
    if out_elem != IntegerType(8):
        raise ValueError("exact quantize expansion currently requires byte storage")
    bounds = (0, 255) if unsigned else (-128, 127)
    if not bounds[0] <= quant_min < quant_max <= bounds[1]:
        raise ValueError("quantization range disagrees with storage signedness")
    rank = len(out_shape)
    id_map = AffineMap.identity(rank)
    chan_map = AffineMap(rank, 0, (AffineExpr.dimension(axis % rank),)) if rank else AffineMap(0, 0, ())

    def build(args, oe):
        x, scale, zero_point = args
        builder = Scalar()
        if literal_scale is not None:
            scale = builder.emit(arith.ConstantOp(FloatAttr(literal_scale, Float64Type())))
        if literal_zero_point is not None:
            zero_point = builder.c(literal_zero_point, IntegerType(64))
        zero_point = builder.cast(zero_point, IntegerType(64), zero_point_unsigned)
        result = quantize_scalar(builder, x, scale, zero_point, quant_min, quant_max, oe, unsigned)
        return builder.ops, result

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
                               literal_scale=literal_scale, literal_zero_point=literal_zero_point,
                               unsigned=getattr(op.output_dtype, "data", None) == "uint8",
                               zero_point_unsigned=getattr(op.attributes.get("prov.quant.zero_point_dtype"), "data", None) == "uint8")


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
                o.attributes["prov.quant.arithmetic"] = StringAttr("exact")
        Rewriter.insert_op(ops, InsertPoint.before(op))
        op.results[0].replace_all_uses_with(result)
        Rewriter.erase_op(op)
    return module
