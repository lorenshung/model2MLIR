"""CPU backward statistics are contiguous pointer reads, including storage holes."""

from math import prod

from xdsl.dialects import memref, tensor
from xdsl.dialects.builtin import IndexType, MemRefType, StridedLayoutAttr, TensorType, UnregisteredOp

from .torch_cpu_rng import Emitter


def contiguous_storage_tensor(source, value):
    """Read the captured in-bounds scalar pointer span into a logical tensor."""
    shape = list(source.type.get_shape())
    count = prod(shape)
    capacity = int(value.untyped_storage().nbytes() // value.element_size())
    offset = int(value.storage_offset())
    if offset < 0 or offset + count > capacity:
        raise ValueError(
            f"native statistics pointer span {offset}+{count} exceeds captured backing storage {capacity}; "
            f"shape={value.shape}, stride={value.stride()}"
        )
    elem = source.type.element_type
    typ = MemRefType(elem, shape, StridedLayoutAttr([None] * len(shape), None))
    borrowed = UnregisteredOp.with_name("bufferization.to_buffer").create(operands=[source], result_types=[typ])
    descriptor = memref.ExtractStridedMetaDataOp(borrowed.results[0])
    linear_type = MemRefType(elem, [count], StridedLayoutAttr([1], None))
    linear = memref.ReinterpretCastOp.from_dynamic(borrowed.results[0], [descriptor.offset], [count], [1], linear_type)
    e = Emitter([borrowed, descriptor, linear])
    empty = e.emit(tensor.EmptyOp([], TensorType(elem, shape)))

    def body(b, j, output):
        from xdsl.dialects import arith

        loaded = b.emit(memref.LoadOp.get(linear.result, [j]))
        coords = []
        for axis, extent in enumerate(shape):
            q = b.binary(arith.DivUIOp, j, b.integer(prod(shape[axis + 1 :]), IndexType()))
            coords.append(b.binary(arith.RemUIOp, q, b.integer(extent, IndexType())))
        return [b.emit(tensor.InsertOp(loaded, output, coords))]

    (output,) = e.loop(count, [empty], body)
    return e.ops, output


def layer_norm_backward(operands, meta, node_name):
    import torch
    from torch._decomp import decomposition_table

    from .core_aten import upstream_lowering
    from .decompositions import _fx_arg

    rewritten = list(operands)
    prefix = []
    # Tensor operands 2/3 are schema arguments 3/4 (normalized_shape is static).
    for operand, argument in ((2, 3), (3, 4)):
        value = _fx_arg(meta, argument).meta["val"]
        if not value.is_contiguous():
            emitted, rewritten[operand] = contiguous_storage_tensor(rewritten[operand], value)
            prefix += emitted
    lower = upstream_lowering(decomposition_table[torch.ops.aten.native_layer_norm_backward.default])
    result = lower(rewritten, meta, node_name)
    result.ops = [*prefix, *result.ops]
    return result
