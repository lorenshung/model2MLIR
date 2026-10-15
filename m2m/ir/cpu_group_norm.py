"""Group normalization with portable structured moments and float opmath."""

from math import prod

from xdsl.dialects import arith
from xdsl.dialects.builtin import BFloat16Type, Float16Type, TensorType, f32


def group_norm(operands, meta, node_name):
    from .decompositions import (
        DecompResult,
        _cast_tensor,
        _element_type_from_meta,
        _elementwise,
        _emit_reshape,
        _fx_arg,
        _shape_of,
        _t_elem,
        build_layer_norm_body,
    )

    x = operands[0]
    shape, elem = _shape_of(x), _t_elem(x)
    if (
        shape is None
        or any(n <= 0 for n in shape)
        or elem not in (f32, Float16Type(), BFloat16Type())
    ):
        # Preserve the framework decomposition for other numeric types.
        import torch
        from torch._decomp import decomposition_table

        from .core_aten import upstream_lowering

        return upstream_lowering(decomposition_table[torch.ops.aten.native_group_norm.default])(
            operands, meta, node_name
        )
    groups = int(_fx_arg(meta, 6, 1))
    count = prod(shape[1:]) // groups
    grouped = [shape[0], groups, count]
    slots = meta.get("_fx_ssa_args", {})
    weight, bias = slots.get(1), slots.get(2)
    ops, wide = _cast_tensor(x, shape, f32)
    reshaped = _emit_reshape(wide, grouped, f32)
    ops += reshaped[0]
    built = build_layer_norm_body(
        reshaped[1], None, None, eps=float(_fx_arg(meta, 7, 1e-5)), k=1, return_stats=True
    )
    ops += built[0]
    mean, rstd = built[2:]
    # Moment outputs are saved in the input storage dtype.
    stats = []
    for ordinal, value in enumerate((mean, rstd), 1):
        reshaped = _emit_reshape(value, [shape[0], groups], f32)
        statistic_elem = _element_type_from_meta({"val": meta["val"][ordinal]})
        cast = _cast_tensor(reshaped[1], [shape[0], groups], statistic_elem)
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

        centered = put(arith.SubfOp(args[0], args[1]))
        result = put(arith.MulfOp(centered, args[2]))
        pos = 3
        if weight is not None:
            result = put(arith.MulfOp(result, args[pos]))
            pos += 1
        if bias is not None:
            result = put(arith.AddfOp(result, args[pos]))
        return emitted, result

    applied = _elementwise(values, TensorType(f32, expanded), apply, input_maps=maps)
    ops += applied[0]
    reshape = _emit_reshape(applied[1], shape, f32)
    cast = _cast_tensor(reshape[1], shape, elem)
    ops += [*reshape[0], *cast[0]]
    return DecompResult(ops=ops, result=cast[1], results=[cast[1], *stats], pattern_hint="group_norm")
