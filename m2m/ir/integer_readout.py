"""Import m2m_q readout as standard linalg/tensor/arith with exact arithmetic.

Original binary scale factors and finite f32 bias stay exact until final RNE.
Rational numerator/denominator strings carry the exact combined scale; no
rounded product or ratio is substituted in provenance. Byte storage stays
signless i8, with signedness carried by the source/public dtype.
"""

from __future__ import annotations

from typing import Any

from xdsl.dialects import arith
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    DenseArrayBase,
    Float64Type,
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


def _quant_attrs(op, scales: list[float] | None, **extra: Any) -> None:
    if scales is None:
        pass
    elif len(scales) == 1:
        op.attributes["prov.quant.scale"] = FloatAttr(scales[0], Float64Type())
    else:
        op.attributes["prov.quant.scale"] = ArrayAttr([FloatAttr(v, Float64Type()) for v in scales])
        op.attributes["prov.quant.granularity"] = StringAttr("channel")
    for key, value in extra.items():
        if isinstance(value, bool):
            op.attributes[f"prov.quant.{key}"] = StringAttr("true" if value else "false")
        elif isinstance(value, int):
            op.attributes[f"prov.quant.{key}"] = IntegerAttr(value, i64)
        else:
            op.attributes[f"prov.quant.{key}"] = StringAttr(str(value))


def decompose_requantize(operands, meta, node_name):
    """``m2m_q.requantize(acc:i32, scale, zp, qmin, qmax, relu) -> i8``."""
    d = _helpers()
    if not operands:
        raise IndexError("m2m_q.requantize needs an accumulator operand")
    from m2m.ir.exact_quant import build_readout
    acc = operands[0]
    scales = _scale_list(meta)
    zero_point, qmin, qmax = (int(d._fx_arg(meta, i)) for i in (2, 3, 4))
    relu = bool(d._fx_arg(meta, 5, False))
    ops, result = build_readout(acc, scales, output_scale=1.0, zero_point=zero_point,
                                quant_min=qmin, quant_max=qmax, relu=relu)
    _quant_attrs(result.owner, scales, zero_point=zero_point, quant_min=qmin, quant_max=qmax,
                 rounding="half_to_even", saturate=True, relu=relu, accumulator="i32", arithmetic="exact")
    result.owner.attributes["prov.role"] = StringAttr("readout")
    rid = d._next_region_id("requantize")
    for op in ops:
        d._attach_region_id(op, rid)
    return d.DecompResult(ops=ops, result=result, region_ids=[rid], pattern_hint="requantize")


def decompose_acc_readout(operands, meta, node_name):
    """``m2m_q.acc_readout(acc:i32, scale) -> f32`` with one scale multiply."""
    d = _helpers()
    if not operands:
        raise IndexError("m2m_q.acc_readout needs an accumulator operand")
    from m2m.ir.exact_quant import build_readout
    scales = _scale_list(meta)
    ops, result = build_readout(operands[0], scales)
    _quant_attrs(result.owner, scales, accumulator="i32", arithmetic="exact")
    result.owner.attributes["prov.role"] = StringAttr("readout")
    rid = d._next_region_id("acc_readout")
    for op in ops:
        d._attach_region_id(op, rid)
    return d.DecompResult(ops=ops, result=result, region_ids=[rid], pattern_hint="acc_readout")


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
    bound = sa[1] * 128 * 128
    if bound > (1 << 31) - 1:
        raise ValueError("seeded contraction may overflow its int32 accumulator")
    i32 = IntegerType(32)
    result_type = TensorType(i32, [m, n])
    from m2m.ir.exact_quant import Scalar
    def check_seed(args, oe):
        scalar = Scalar()
        value = scalar.cast(args[0], i64)
        valid = scalar.band(scalar.cmp(value, scalar.c(-(1 << 31) + bound, i64), "sge"),
                            scalar.cmp(value, scalar.c((1 << 31) - 1 - bound, i64), "sle"))
        scalar.assert_(valid, "seeded contraction may overflow its int32 accumulator")
        return scalar.ops, args[0]
    ops, seed = d._elementwise([seed], seed.type, check_seed)
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


def decompose_exact_readout(operands, meta, node_name):
    from fractions import Fraction
    from m2m.ir.exact_quant import build_readout
    d = _helpers()
    activation = float(d._fx_arg(meta, 1))
    weights = list(d._fx_arg(meta, 2))
    alpha = float(d._fx_arg(meta, 3, 1.0))
    bias_arg = d._fx_arg(meta, 4)
    bias = None
    if bias_arg is not None:
        bias = (meta.get("_fx_ssa_args") or {}).get(4)
        if bias is None:
            bias = operands[1]
    beta = float(d._fx_arg(meta, 5, 1.0))
    output_scale = d._fx_arg(meta, 6)
    zp, low, high = (int(d._fx_arg(meta, i, default)) for i, default in ((7, 0), (8, -128), (9, 127)))
    relu, unsigned = bool(d._fx_arg(meta, 10, False)), bool(d._fx_arg(meta, 11, False))
    scales = [Fraction(activation) * Fraction(alpha) * Fraction(float(s)) for s in weights]
    ops, result = build_readout(operands[0], scales, bias=bias, beta=beta, output_scale=output_scale,
                               zero_point=zp, quant_min=low, quant_max=high, relu=relu, unsigned=unsigned)
    generic = result.owner
    _quant_attrs(generic, None, accumulator="i32")
    ratio = [s / Fraction(output_scale) if output_scale is not None else s for s in scales]
    generic.attributes["prov.quant.scale_numerator"] = ArrayAttr([StringAttr(str(s.numerator)) for s in ratio])
    generic.attributes["prov.quant.scale_denominator"] = ArrayAttr([StringAttr(str(s.denominator)) for s in ratio])
    # Factors, rather than their rounded product, define the exact scale.
    generic.attributes["prov.quant.lhs_scale"] = FloatAttr(activation, Float64Type())
    generic.attributes["prov.quant.rhs_scale"] = ArrayAttr([FloatAttr(float(s), Float64Type()) for s in weights])
    generic.attributes["prov.quant.alpha"] = FloatAttr(alpha, Float64Type())
    generic.attributes["prov.quant.beta"] = FloatAttr(beta, Float64Type())
    generic.attributes["prov.quant.arithmetic"] = StringAttr("exact")
    generic.attributes["prov.role"] = StringAttr("readout")
    if output_scale is not None:
        generic.attributes["prov.quant.output_scale"] = FloatAttr(output_scale, Float64Type())
        generic.attributes["prov.quant.storage_dtype"] = StringAttr("uint8" if unsigned else "int8")
        _quant_attrs(generic, None, zero_point=zp, quant_min=low, quant_max=high,
                     rounding="half_to_even", saturate=True, relu=relu, accumulator="i32")
    rid = d._next_region_id("requantize" if output_scale is not None else "acc_readout")
    for op in ops:
        d._attach_region_id(op, rid)
    return d.DecompResult(ops=ops, result=result, region_ids=[rid],
                          pattern_hint="requantize" if output_scale is not None else "acc_readout")


INTEGER_READOUT_DECOMPOSITIONS["m2m_q.exact_readout.default"] = decompose_exact_readout


def decompose_exact_quantize(operands, meta, node_name):
    import torch
    d = _helpers()
    cloned = dict(meta)
    args = list(meta["_fx_args"])
    args[5:6] = [torch.uint8 if d._fx_arg(meta, 5, False) else torch.int8]
    cloned["_fx_args"] = args
    return d.decompose_quantize_per_tensor(operands, cloned, node_name)


INTEGER_READOUT_DECOMPOSITIONS["m2m_q.quantize.default"] = decompose_exact_quantize


def decompose_exact_quantize_channel(operands, meta, node_name):
    import torch
    d = _helpers()
    cloned = dict(meta)
    args = list(meta["_fx_args"])
    args[6:7] = [torch.uint8 if d._fx_arg(meta, 6, False) else torch.int8]
    cloned["_fx_args"] = args
    return d.decompose_quantize_per_channel(operands, cloned, node_name)


INTEGER_READOUT_DECOMPOSITIONS["m2m_q.quantize_channel.default"] = decompose_exact_quantize_channel
