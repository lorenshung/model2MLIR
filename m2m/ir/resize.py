"""Static resize by source gathers, preserving IEEE values and opmath storage."""

import math
import struct

from xdsl.dialects import arith, builtin, linalg, tensor
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap


def axis_samples(n_in, n_out, *, nearest, align_corners, scale=None, wide=False):
    """ATen source indices and linear lambdas, including supplied scale factors."""

    def rounded(value):
        return value if wide else struct.unpack("f", struct.pack("f", value))[0]

    ratio = rounded(
        (n_in - 1) / (n_out - 1)
        if align_corners and n_out > 1
        else 0.0
        if align_corners
        else 1.0 / scale
        if scale is not None and scale > 0
        else n_in / n_out
    )
    rows = []
    for i in range(n_out):
        if nearest:
            index = (
                i if n_in == n_out else i // 2 if n_out == 2 * n_in else min(math.floor(rounded(i * ratio)), n_in - 1)
            )
            rows.append((index, index, 1.0, 0.0))
        elif n_in == n_out:
            rows.append((i, i, 1.0, 0.0))
        else:
            position = rounded(i * ratio) if align_corners else max(rounded(rounded((i + 0.5) * ratio) - 0.5), 0.0)
            low = min(math.floor(position), n_in - 1)
            high = low + (low < n_in - 1)
            fraction = rounded(min(max(rounded(position - low), 0.0), 1.0))
            rows.append((low, high, rounded(1.0 - fraction), fraction))
    return rows


def resize2d(source, out_shape, *, nearest, align_corners, scales=None):
    """Gather one nearest sample or four bilinear samples and round once."""
    from m2m.ir.decompositions import _cast_scalar_arg

    shape = source.type.get_shape()
    elem = source.type.element_type
    opmath = builtin.f32 if isinstance(elem, (builtin.Float16Type, builtin.BFloat16Type)) else elem
    out_type = builtin.TensorType(elem, out_shape)
    empty = tensor.EmptyOp([], out_type)
    ops = [empty]
    tables = []
    for axis in (2, 3):
        rows = axis_samples(
            shape[axis],
            out_shape[axis],
            nearest=nearest,
            align_corners=align_corners,
            scale=scales[axis - 2] if scales else None,
            wide=isinstance(elem, builtin.Float64Type) and not nearest,
        )
        indices = arith.ConstantOp(
            builtin.DenseIntOrFPElementsAttr.from_list(
                builtin.TensorType(builtin.i64, [len(rows), 2]), [v for row in rows for v in row[:2]]
            )
        )
        weights = arith.ConstantOp(
            builtin.DenseIntOrFPElementsAttr.from_list(
                builtin.TensorType(opmath, [len(rows), 2]), [v for row in rows for v in row[2:]]
            )
        )
        ops.extend([indices, weights])
        tables.append((indices.result, weights.result))
    block = Block(arg_types=[elem])
    coords = [linalg.IndexOp(axis) for axis in range(4)]
    block.add_ops(coords)
    selected, lambdas = [], []
    for axis, (indices, weights) in enumerate(tables):
        positions, factors = [], []
        for side in range(1 if nearest else 2):
            ordinal = arith.ConstantOp.from_int_and_width(side, builtin.IndexType())
            index = tensor.ExtractOp(indices, [coords[axis + 2].result, ordinal.result], builtin.i64)
            cast = arith.IndexCastOp(index.result, builtin.IndexType())
            block.add_ops([ordinal, index, cast])
            positions.append(cast.result)
            if not nearest:
                weight = tensor.ExtractOp(weights, [coords[axis + 2].result, ordinal.result], opmath)
                block.add_op(weight)
                factors.append(weight.result)
        selected.append(positions)
        lambdas.append(factors)
    values = []
    for h in range(1 if nearest else 2):
        for w in range(1 if nearest else 2):
            sample = tensor.ExtractOp(
                source, [coords[0].result, coords[1].result, selected[0][h], selected[1][w]], elem
            )
            block.add_op(sample)
            if nearest:
                values.append(sample.result)
            else:
                cast_ops, value = _cast_scalar_arg(sample.result, opmath)
                weight = arith.MulfOp(lambdas[0][h], lambdas[1][w])
                product = arith.MulfOp(value, weight.result)
                block.add_ops([*cast_ops, weight, product])
                values.append(product.result)
    result = values[0]
    for value in values[1:]:
        add = arith.AddfOp(result, value)
        block.add_op(add)
        result = add.result
    cast_ops, result = _cast_scalar_arg(result, elem)
    block.add_ops([*cast_ops, linalg.YieldOp(result)])
    generic = linalg.GenericOp(
        inputs=[],
        outputs=[empty.tensor],
        body=Region(block),
        indexing_maps=[builtin.AffineMapAttr(AffineMap.identity(4))],
        iterator_types=[linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * 4,
        result_types=[out_type],
    )
    ops.append(generic)
    return ops, generic.results[0]
