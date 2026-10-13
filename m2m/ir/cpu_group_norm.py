"""Native CPU group normalization using rowwise Welford moments."""

from math import prod

from xdsl.dialects import arith, math
from xdsl.dialects.builtin import BFloat16Type, Float16Type, TensorType, f32


def group_norm(operands, meta, node_name):
    from .decompositions import (
        DecompResult,
        _cast_tensor,
        _elementwise,
        _emit_reshape,
        _fx_arg,
        _opaque_decomp,
        _shape_of,
        _t_elem,
    )
    from .half_layer_norm import build_half_native

    x = operands[0]
    shape, elem = _shape_of(x), _t_elem(x)
    if shape is None or any(n <= 0 for n in shape) or elem not in (f32, Float16Type(), BFloat16Type()):
        return _opaque_decomp("aten_native_group_norm", operands, meta, "normalization")
    groups = int(_fx_arg(meta, 6, 1))
    count = prod(shape[1:]) // groups
    grouped = [shape[0], groups, count]
    slots = meta.get("_fx_ssa_args", {})
    weight, bias = slots.get(1), slots.get(2)
    ops, wide = _cast_tensor(x, shape, f32)
    reshaped = _emit_reshape(wide, grouped, f32)
    ops += reshaped[0]
    source_value = getattr(_fx_arg(meta, 0), "meta", {}).get("val")
    import torch

    channels_last = source_value is not None and (
        len(shape) == 4
        and source_value.is_contiguous(memory_format=torch.channels_last)
        or len(shape) == 5
        and source_value.is_contiguous(memory_format=torch.channels_last_3d)
    )
    if channels_last and not source_value.is_contiguous():
        from .decompositions import _bin_build, _reduce, _splat_scalar

        moment_shape = [shape[0], groups, shape[1] // groups, prod(shape[2:])]
        moment_input = _emit_reshape(wide, moment_shape, f32)
        ops += moment_input[0]
        square = _elementwise(
            [moment_input[1], moment_input[1]], TensorType(f32, moment_shape), _bin_build(arith.MulfOp)
        )
        ops += square[0]
        reduced = []
        for value in (moment_input[1], square[1]):
            first, value, reduced_shape = _reduce(value, moment_shape, [3], 0.0, arith.AddfOp, f32)
            second, value, reduced_shape = _reduce(value, reduced_shape, [2], 0.0, arith.AddfOp, f32)
            divisor = _splat_scalar(float(count), TensorType(f32, reduced_shape))
            divided = _elementwise([value, divisor[1]], TensorType(f32, reduced_shape), _bin_build(arith.DivfOp))
            reshaped_stat = _emit_reshape(divided[1], [shape[0], groups, 1], f32)
            ops += [*first, *second, *divisor[0], *divided[0], *reshaped_stat[0]]
            reduced.append(reshaped_stat[1])
        mean = reduced[0]

        def inverse(args, elem):
            from .torch_cpu_rng import Emitter

            b = Emitter()
            variance = b.binary(arith.SubfOp, args[1], b.binary(arith.MulfOp, args[0], args[0]))
            variance = b.binary(arith.MaximumfOp, variance, b.floating(0.0, f32))
            denominator = b.emit(
                math.SqrtOp(b.binary(arith.AddfOp, variance, b.floating(float(_fx_arg(meta, 7, 1e-5)), f32)))
            )
            return b.ops, b.binary(arith.DivfOp, b.floating(1.0, f32), denominator)

        inverse_ops, rstd = _elementwise(reduced, mean.type, inverse)
        ops += inverse_ops
    else:
        built = build_half_native(
            reshaped[1], None, None, eps=float(_fx_arg(meta, 7, 1e-5)), k=1, storage_half=elem != f32
        )
        ops += built[0]
        mean, rstd = built[2:]
    # Moment outputs are saved in the input storage dtype.
    stats = []
    for value in (mean, rstd):
        reshaped = _emit_reshape(value, [shape[0], groups], f32)
        cast = _cast_tensor(reshaped[1], [shape[0], groups], elem)
        ops += [*reshaped[0], *cast[0]]
        stats.append(cast[1])
    # Broadcast group statistics over each group's channels and spatial axes.
    expanded = [shape[0], groups, shape[1] // groups, *shape[2:]]
    reshape = _emit_reshape(wide, expanded, f32)
    ops += reshape[0]
    values = [reshape[1], mean, rstd]
    from xdsl.ir.affine import AffineExpr, AffineMap

    maps = [
        AffineMap.identity(len(expanded)),
        AffineMap(len(expanded), 0, (AffineExpr.dimension(0), AffineExpr.dimension(1), AffineExpr.constant(0))),
    ]
    maps.append(maps[1])
    for value in (weight, bias):
        if value is not None:
            cast = _cast_tensor(value, _shape_of(value), f32)
            grouped_affine = _emit_reshape(cast[1], [groups, shape[1] // groups], f32)
            ops += [*cast[0], *grouped_affine[0]]
            values.append(grouped_affine[1])
            maps.append(AffineMap(len(expanded), 0, (AffineExpr.dimension(1), AffineExpr.dimension(2))))

    def apply(args, _elem):
        emitted = []

        def put(op):
            emitted.append(op)
            return op.results[0]

        scale = args[2]
        pos = 3
        if weight is not None:
            scale = put(arith.MulfOp(scale, args[pos]))
            pos += 1
        offset = put(arith.NegfOp(put(arith.MulfOp(scale, args[1]))))
        if bias is not None:
            offset = put(arith.AddfOp(offset, args[pos]))
        result = put(arith.AddfOp(put(arith.MulfOp(args[0], scale)), offset))
        return emitted, result

    applied = _elementwise(values, TensorType(f32, expanded), apply, input_maps=maps)
    ops += applied[0]
    reshape = _emit_reshape(applied[1], shape, f32)
    cast = _cast_tensor(reshape[1], shape, elem)
    ops += [*reshape[0], *cast[0]]
    return DecompResult(ops=ops, result=cast[1], results=[cast[1], *stats], pattern_hint="group_norm")
