"""Convolution input gradients without multiplying structural padding zeros."""

from xdsl.dialects import arith, builtin, linalg, scf, tensor
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap


def input_gradient(grad, weight, shape, *, stride, padding, dilation, groups, storage_elem=None):
    """Reduce each valid kernel patch; absent forward edges contribute no product."""
    from m2m.ir.decompositions import _cast_scalar_arg

    elem = grad.type.element_type
    storage_elem = elem if storage_elem is None else storage_elem
    wshape = weight.type.get_shape()
    gshape = grad.type.get_shape()
    kh, kw = wshape[-2:]
    channels = wshape[0] // groups
    result_type = builtin.TensorType(storage_elem, shape)
    empty = tensor.EmptyOp([], result_type)
    ops = [empty]
    body = Block(arg_types=[storage_elem])

    def emit(block, op):
        block.add_op(op)
        return op.results[0]

    def index(block, value):
        return emit(block, arith.ConstantOp.from_int_and_width(value, builtin.IndexType()))

    zero, one, patches, filters = [index(body, v) for v in (0, 1, kh * kw, channels)]
    fzero = emit(body, arith.ConstantOp(builtin.FloatAttr(0.0, elem)))
    coordinates = [emit(body, linalg.IndexOp(i)) for i in range(4)]
    storage_zero = emit(body, arith.ConstantOp(builtin.FloatAttr(0.0, storage_elem)))
    kernel = Block(arg_types=[builtin.IndexType(), storage_elem])
    patch, total = kernel.args
    width = index(kernel, kw)
    ky = emit(kernel, arith.DivUIOp(patch, width))
    kx = emit(kernel, arith.RemUIOp(patch, width))
    selected, checks = [], []
    for axis, k in enumerate((ky, kx)):
        pad, dil, step, limit = [
            index(kernel, v) for v in (padding[axis], dilation[axis], stride[axis], gshape[axis + 2])
        ]
        shifted = emit(kernel, arith.AddiOp(coordinates[axis + 2], pad))
        delta = emit(kernel, arith.MuliOp(k, dil))
        numerator = emit(kernel, arith.SubiOp(shifted, delta))
        source = emit(kernel, arith.DivSIOp(numerator, step))
        remainder = emit(kernel, arith.RemSIOp(numerator, step))
        selected.append(source)
        checks.extend(
            [
                emit(kernel, arith.CmpiOp(numerator, zero, "sge")),
                emit(kernel, arith.CmpiOp(source, limit, "slt")),
                emit(kernel, arith.CmpiOp(remainder, zero, "eq")),
            ]
        )
    valid = checks[0]
    for check in checks[1:]:
        valid = emit(kernel, arith.AndIOp(valid, check))
    yes, no = Block(), Block()
    cgroup = index(yes, wshape[1])
    group = emit(yes, arith.DivUIOp(coordinates[1], cgroup))
    channel = emit(yes, arith.RemUIOp(coordinates[1], cgroup))
    base = emit(yes, arith.MuliOp(group, filters))
    reduction = Block(arg_types=[builtin.IndexType(), elem])
    filt, acc = reduction.args
    out_channel = emit(reduction, arith.AddiOp(base, filt))
    dy = emit(reduction, tensor.ExtractOp(grad, [coordinates[0], out_channel, *selected], elem))
    w = emit(reduction, tensor.ExtractOp(weight, [out_channel, channel, ky, kx], elem))
    product = emit(reduction, arith.MulfOp(dy, w))
    summed = emit(reduction, arith.AddfOp(acc, product))
    reduction.add_op(scf.YieldOp(summed))
    partial = emit(yes, scf.ForOp(zero, filters, one, [fzero], Region(reduction)))
    # CPU backward GEMM publishes its column buffer in input storage, then
    # col2im adds each patch into input storage. Preserve both observations.
    rounded_ops, partial = _cast_scalar_arg(partial, storage_elem)
    total_ops, total_wide = _cast_scalar_arg(total, elem)
    partial_ops, partial_wide = _cast_scalar_arg(partial, elem)
    yes.add_ops([*rounded_ops, *total_ops, *partial_ops])
    added = emit(yes, arith.AddfOp(total_wide, partial_wide))
    narrow_ops, added = _cast_scalar_arg(added, storage_elem)
    yes.add_ops(narrow_ops)
    yes.add_op(scf.YieldOp(added))
    no.add_op(scf.YieldOp(total))
    branch = emit(kernel, scf.IfOp(valid, [storage_elem], Region(yes), Region(no)))
    kernel.add_op(scf.YieldOp(branch))
    result = emit(body, scf.ForOp(zero, patches, one, [storage_zero], Region(kernel)))
    body.add_op(linalg.YieldOp(result))
    generic = linalg.GenericOp(
        inputs=[],
        outputs=[empty.tensor],
        body=Region(body),
        indexing_maps=[builtin.AffineMapAttr(AffineMap.identity(4))],
        iterator_types=[linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * 4,
        result_types=[result_type],
    )
    return [*ops, generic], generic.results[0]
