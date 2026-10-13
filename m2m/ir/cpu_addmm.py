"""Stable CPU float32 addmm with compensated double-precision accumulation.

Float32 inputs have exact products in f64. Neumaier compensation retains small
terms across large cancellations without overflowing the float32 accumulator.
The public matrix result is rounded to f32 before its separate f32 bias add;
alpha and beta are converted to the source scalar type first, as CPU BLAS does.
"""

import ctypes
from numbers import Real

from xdsl.dialects import arith, math
from xdsl.dialects.builtin import AffineMapAttr, StringAttr, TensorType, f32, f64
from xdsl.dialects.linalg import GenericOp, IteratorType, IteratorTypeAttr, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineExpr, AffineMap


def stable_addmm(operands, meta, node_name):
    from .decompositions import (
        DecompResult,
        _bin_build,
        _broadcast_map,
        _cast_tensor,
        _elementwise,
        _fx_arg,
        _shape_of,
        _splat_scalar,
        _t_elem,
    )

    if len(operands) != 3 or any(_t_elem(value) != f32 for value in operands):
        return None
    source = getattr(_fx_arg(meta, 1), "meta", {}).get("val")
    if source is None or source.device.type != "cpu":
        return None
    bias, a, b = operands
    ashape, bshape, bias_shape = _shape_of(a), _shape_of(b), _shape_of(bias)
    shape = list(meta["val"].shape)
    if (
        ashape is None
        or bshape is None
        or len(ashape) != 2
        or len(bshape) != 2
        or any(n < 0 for n in (*ashape, *bshape, *shape))
    ):
        return None
    alpha = meta.get("_fx_kwargs", {}).get("alpha", _fx_arg(meta, 4, 1))
    beta = meta.get("_fx_kwargs", {}).get("beta", _fx_arg(meta, 3, 1))
    if not isinstance(alpha, Real) or not isinstance(beta, Real):
        return None
    alpha, beta = ctypes.c_float(alpha).value, ctypes.c_float(beta).value
    result_type = TensorType(f32, shape)
    bias_map = _broadcast_map(bias_shape, shape)
    identity = _broadcast_map(shape, shape)
    if bias_map is None or identity is None:
        return None
    ops = []

    def scale(value, value_shape, factor, elem):
        if factor == 1:
            return value
        constant = _splat_scalar(factor, TensorType(elem, value_shape))
        emitted = _elementwise([value, constant[1]], TensorType(elem, value_shape), _bin_build(arith.MulfOp))
        ops.extend([*constant[0], *emitted[0]])
        return emitted[1]

    def bias_only():
        if beta == 0:
            zeros = _splat_scalar(0.0, result_type)
            ops.extend(zeros[0])
            return zeros[1]
        scaled = scale(bias, bias_shape, beta, f32)
        copied = _elementwise([scaled], result_type, lambda args, elem: ([], args[0]), input_maps=[bias_map])
        ops.extend(copied[0])
        return copied[1]

    # BLAS ignores A/B for alpha zero, and an empty contraction contributes
    # nothing regardless of alpha. Neither case may evaluate 0 * NaN/Inf.
    if alpha == 0 or ashape[1] == 0:
        return DecompResult(ops=ops, result=bias_only(), pattern_hint="addmm")

    acc_type = TensorType(f64, shape)
    initial = [_splat_scalar(0.0, acc_type) for _ in range(2)]
    for value in initial:
        ops.extend(value[0])
    block = Block(arg_types=[f32, f32, f64, f64])
    left, right, total, error = block.args

    def put(op):
        block.add_op(op)
        return op.results[0]

    left = put(arith.ExtFOp(left, f64))
    right = put(arith.ExtFOp(right, f64))
    product = put(arith.MulfOp(left, right))
    updated = put(arith.AddfOp(total, product))
    total_abs = put(math.AbsFOp(total))
    product_abs = put(math.AbsFOp(product))
    larger = put(arith.CmpfOp(total_abs, product_abs, "oge"))
    total_error = put(arith.AddfOp(put(arith.SubfOp(total, updated)), product))
    product_error = put(arith.AddfOp(put(arith.SubfOp(product, updated)), total))
    correction = put(arith.SelectOp(larger, total_error, product_error))
    from xdsl.dialects.builtin import FloatAttr

    infinity = put(arith.ConstantOp(FloatAttr(float("inf"), f64), f64))
    zero = put(arith.ConstantOp(FloatAttr(0.0, f64), f64))
    finite_total = put(arith.CmpfOp(total_abs, infinity, "olt"))
    finite_product = put(arith.CmpfOp(product_abs, infinity, "olt"))
    finite = put(arith.AndIOp(finite_total, finite_product))
    correction = put(arith.SelectOp(finite, correction, zero))
    error = put(arith.AddfOp(error, correction))
    block.add_op(YieldOp(updated, error))
    d = AffineExpr.dimension
    output_map = AffineMap(3, 0, (d(0), d(1)))
    reduction = GenericOp(
        inputs=[a, b],
        outputs=[value[1] for value in initial],
        body=Region(block),
        indexing_maps=[
            AffineMapAttr(AffineMap(3, 0, (d(0), d(2)))),
            AffineMapAttr(AffineMap(3, 0, (d(2), d(1)))),
            AffineMapAttr(output_map),
            AffineMapAttr(output_map),
        ],
        iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)] * 2 + [IteratorTypeAttr(IteratorType.REDUCTION)],
        result_types=[acc_type, acc_type],
    )
    reduction.attributes["prov.accumulation"] = StringAttr("compensated_f64")
    ops.append(reduction)
    combined = _elementwise(list(reduction.results), acc_type, _bin_build(arith.AddfOp))
    ops.extend(combined[0])
    scaled = scale(combined[1], shape, alpha, f64)
    narrowed = _cast_tensor(scaled, shape, f32)
    ops.extend(narrowed[0])
    result = narrowed[1]
    if beta != 0:
        scaled_bias = scale(bias, bias_shape, beta, f32)
        added = _elementwise(
            [result, scaled_bias], result_type, _bin_build(arith.AddfOp), input_maps=[identity, bias_map]
        )
    else:
        # BLAS starts a beta-zero output from positive zero. Preserve that
        # signed-zero observation without reading the ignored bias operand.
        zero_output = _splat_scalar(0.0, result_type)
        ops.extend(zero_output[0])
        added = _elementwise([result, zero_output[1]], result_type, _bin_build(arith.AddfOp))
    ops.extend(added[0])
    result = added[1]
    return DecompResult(ops=ops, result=result, pattern_hint="addmm")
