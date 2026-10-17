"""Import the ``m2m_q`` integer-readout operators (see ``m2m.capture.integer_readout``).

Every emitted operation is standard ``linalg``/``tensor``/``arith``/``math``:

* ``int_mm_seeded`` -> the i8 x i8 -> i32 contraction ``linalg.generic`` (tagged
  ``prov.op = "int_matmul"`` like ``aten._int_mm``) whose ``outs`` operand is the
  int32 bias seed, broadcast with ``linalg.broadcast`` (``prov.role = "acc_seed"``)
  instead of a zero splat.
* ``requantize`` / ``acc_readout`` -> one parallel ``linalg.generic`` (``prov.role =
  "readout"``) carrying the scale, zero point, range and rounding as ``prov.quant.*``
  attributes. Per-channel scales are an inlined dense constant.
"""

from __future__ import annotations

from typing import Any

from xdsl.dialects import arith
from xdsl.dialects import math as mlir_math
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    Float32Type,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    TensorType,
    i64,
)
from xdsl.dialects.linalg import BroadcastOp, GenericOp, IteratorType, IteratorTypeAttr, YieldOp
from xdsl.dialects.tensor import SplatOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineExpr, AffineMap


def _helpers():
    from m2m.ir import decompositions as d

    return d


def _scale_list(meta: dict[str, Any]) -> list[float]:
    d = _helpers()
    raw = d._fx_arg(meta, 1)
    if isinstance(raw, (int, float)):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw or not all(isinstance(v, (int, float)) for v in raw):
        raise ValueError(f"m2m_q readout scale must be a literal float list, got {raw!r}")
    return [float(v) for v in raw]


def _quant_attrs(op, scales: list[float], **extra: Any) -> None:
    if len(scales) == 1:
        op.attributes["prov.quant.scale"] = FloatAttr(scales[0], Float32Type())
    else:
        op.attributes["prov.quant.scale"] = ArrayAttr([FloatAttr(v, Float32Type()) for v in scales])
        op.attributes["prov.quant.granularity"] = StringAttr("channel")
    for key, value in extra.items():
        if isinstance(value, bool):
            op.attributes[f"prov.quant.{key}"] = StringAttr("true" if value else "false")
        elif isinstance(value, int):
            op.attributes[f"prov.quant.{key}"] = IntegerAttr(value, i64)
        else:
            op.attributes[f"prov.quant.{key}"] = StringAttr(str(value))


def _readout(acc, scales: list[float], out_elem, body, meta, *, hint: str):
    """A rank-preserving parallel generic ``out = body(f32(acc) * scale)``."""
    d = _helpers()
    shape = list(acc.type.get_shape())
    rank = len(shape)
    f32 = Float32Type()
    ops = []
    inputs = [acc]
    maps = [AffineMap.identity(rank)]
    per_channel = len(scales) > 1
    if per_channel:
        if not shape or shape[-1] != len(scales):
            raise ValueError(f"{len(scales)} channel scales do not match the last axis of {shape}")
        scale_type = TensorType(f32, [len(scales)])
        const = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(scale_type, scales), scale_type)
        const.attributes["prov.role"] = StringAttr("readout_scale")
        ops.append(const)
        inputs.append(const.result)
        maps.append(AffineMap(rank, 0, (AffineExpr.dimension(rank - 1),)))
    result_type = TensorType(out_elem, shape)
    empty = d._make_empty(result_type)
    ops.append(empty)
    maps.append(AffineMap.identity(rank))
    block = Block(arg_types=[acc.type.element_type, *([f32] if per_channel else []), out_elem])
    as_float = arith.SIToFPOp(block.args[0], f32)
    block.add_op(as_float)
    if per_channel:
        scale = block.args[1]
    else:
        constant = arith.ConstantOp(FloatAttr(scales[0], f32), f32)
        block.add_op(constant)
        scale = constant.result
    scaled = arith.MulfOp(as_float.result, scale)
    block.add_op(scaled)
    block.add_op(YieldOp(body(block, scaled.result)))
    generic = GenericOp(
        inputs=inputs,
        outputs=[empty.results[0]],
        body=Region(block),
        indexing_maps=[AffineMapAttr(m) for m in maps],
        iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)] * rank,
        result_types=[result_type],
    )
    generic.attributes["prov.role"] = StringAttr("readout")
    ops.append(generic)
    rid = d._next_region_id(hint)
    for op in ops:
        d._attach_region_id(op, rid)
    return ops, generic, rid


def decompose_requantize(operands, meta, node_name):
    """``m2m_q.requantize(acc:i32, scale, zp, qmin, qmax, relu) -> i8``."""
    d = _helpers()
    if not operands:
        raise IndexError("m2m_q.requantize needs an accumulator operand")
    acc = operands[0]
    scales = _scale_list(meta)
    zero_point, qmin, qmax = (int(d._fx_arg(meta, i)) for i in (2, 3, 4))
    relu = bool(d._fx_arg(meta, 5, False) or (meta.get("_fx_kwargs") or {}).get("relu", False))
    low = max(qmin, zero_point) if relu else qmin
    f32 = Float32Type()
    i8 = IntegerType(8)

    def body(block, value):
        rounded = mlir_math.RoundEvenOp(value)
        block.add_op(rounded)
        value = rounded.result
        if zero_point:
            zp = arith.ConstantOp(FloatAttr(float(zero_point), f32), f32)
            shifted = arith.AddfOp(value, zp.result)
            block.add_ops([zp, shifted])
            value = shifted.result
        lo = arith.ConstantOp(FloatAttr(float(low), f32), f32)
        hi = arith.ConstantOp(FloatAttr(float(qmax), f32), f32)
        # The scaled accumulator is finite, so max/min saturate exactly.
        clamp_lo = arith.MaximumfOp(value, lo.result)
        clamp_hi = arith.MinimumfOp(clamp_lo.result, hi.result)
        narrow = arith.FPToSIOp(clamp_hi.result, i8)
        block.add_ops([lo, hi, clamp_lo, clamp_hi, narrow])
        return narrow.result

    ops, generic, rid = _readout(acc, scales, i8, body, meta, hint="requantize")
    _quant_attrs(generic, scales, zero_point=zero_point, qmin=qmin, qmax=qmax,
                 rounding="half_to_even", saturate=True, relu=relu, accumulator="i32")
    return d.DecompResult(ops=ops, result=generic.results[0], region_ids=[rid], pattern_hint="requantize")


def decompose_acc_readout(operands, meta, node_name):
    """``m2m_q.acc_readout(acc:i32, scale) -> f32`` with one scale multiply."""
    d = _helpers()
    if not operands:
        raise IndexError("m2m_q.acc_readout needs an accumulator operand")
    scales = _scale_list(meta)
    ops, generic, rid = _readout(operands[0], scales, Float32Type(), lambda block, value: value, meta,
                                 hint="acc_readout")
    _quant_attrs(generic, scales, accumulator="i32")
    return d.DecompResult(ops=ops, result=generic.results[0], region_ids=[rid], pattern_hint="acc_readout")


def decompose_int_mm_seeded(operands, meta, node_name):
    """``m2m_q.int_mm_seeded(a:i8[M,K], b:i8[K,N], seed:i32[N] | i32[M,N]) -> i32[M,N]``."""
    d = _helpers()
    if len(operands) < 3:
        raise IndexError("m2m_q.int_mm_seeded needs lhs, rhs and seed operands")
    a, b, seed = operands[:3]
    sa, sb, ss = d._shape_of(a), d._shape_of(b), d._shape_of(seed)
    if (sa is None or sb is None or ss is None or len(sa) != 2 or len(sb) != 2 or sa[1] != sb[0]
            or any(dim < 0 for dim in (*sa, *sb, *ss))):
        raise ValueError("m2m_q.int_mm_seeded requires static rank-2 operands")
    m, n = sa[0], sb[1]
    i32 = IntegerType(32)
    result_type = TensorType(i32, [m, n])
    ops = []
    if ss == [m, n]:
        init = seed
    elif ss == [n]:
        empty = d._make_empty(result_type)
        broadcast = BroadcastOp(seed, empty.results[0], DenseArrayBase.from_list(i64, [0]), result_type)
        broadcast.attributes["prov.role"] = StringAttr("acc_seed")
        broadcast.attributes["prov.op"] = StringAttr("broadcast")
        ops += [empty, broadcast]
        init = broadcast.results[0]
    elif ss == []:
        from xdsl.dialects.tensor import ExtractOp

        scalar = ExtractOp(seed, [], i32)
        splat = SplatOp(scalar.result, [], result_type)
        splat.attributes["prov.role"] = StringAttr("acc_seed")
        ops += [scalar, splat]
        init = splat.results[0]
    else:
        raise ValueError(f"m2m_q.int_mm_seeded seed shape {ss} is not [N] or [M, N]")
    D = AffineExpr.dimension
    block = Block(arg_types=[a.type.element_type, b.type.element_type, i32])
    lhs = arith.ExtSIOp(block.args[0], i32)
    rhs = arith.ExtSIOp(block.args[1], i32)
    prod = arith.MuliOp(lhs.result, rhs.result)
    acc = arith.AddiOp(block.args[2], prod.result)
    block.add_ops([lhs, rhs, prod, acc, YieldOp(acc.result)])
    par, red = IteratorTypeAttr(IteratorType.PARALLEL), IteratorTypeAttr(IteratorType.REDUCTION)
    generic = GenericOp(
        inputs=[a, b],
        outputs=[init],
        body=Region(block),
        indexing_maps=[AffineMapAttr(AffineMap(3, 0, (D(0), D(2)))), AffineMapAttr(AffineMap(3, 0, (D(2), D(1)))),
                       AffineMapAttr(AffineMap(3, 0, (D(0), D(1))))],
        iterator_types=[par, par, red],
        result_types=[result_type],
    )
    generic.attributes["prov.accumulator_init"] = StringAttr("int32_seed")
    ops.append(generic)
    rid = d._next_region_id("matmul")
    for op in ops:
        d._attach_region_id(op, rid)
    return d.DecompResult(ops=ops, result=generic.results[0], region_ids=[rid], pattern_hint="int_matmul")


INTEGER_READOUT_DECOMPOSITIONS = {
    "m2m_q.int_mm_seeded.default": decompose_int_mm_seeded,
    "m2m_q.requantize.default": decompose_requantize,
    "m2m_q.acc_readout.default": decompose_acc_readout,
}
