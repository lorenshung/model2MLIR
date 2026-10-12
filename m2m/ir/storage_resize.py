"""resize_ reads backing storage, not the flattened logical tensor elements."""
from math import prod
import torch
from xdsl.dialects import arith, memref, tensor
from xdsl.dialects.builtin import IndexType, MemRefType, StridedLayoutAttr, TensorType, UnregisteredOp
from .decompositions import DecompResult, _fx_arg, _next_region_id
from .torch_cpu_rng import Emitter


def resize_storage(operands, meta, node_name):
    source = operands[0]
    value = _fx_arg(meta, 0).meta['val']
    result = meta['val']
    pair = value.dtype == torch.complex64
    components = 2 if pair else 1
    shape = list(result.shape) + ([2] if pair else [])
    if list(result.shape) == list(value.shape):
        return DecompResult(ops=[], result=source, pattern_hint='view')
    old_shape = list(source.type.get_shape())
    elem = source.type.element_type
    # A descriptor includes offset/strides even though tensor types do not.
    typ = MemRefType(elem, old_shape, StridedLayoutAttr([None] * len(old_shape), None))
    # xDSL's ToBufferOp inverse tensor constraint incorrectly discards layouts.
    # Emit the standard operation directly until that upstream constraint is fixed.
    borrowed = UnregisteredOp.with_name('bufferization.to_buffer').create(operands=[source], result_types=[typ])
    descriptor = memref.ExtractStridedMetaDataOp(borrowed.results[0])
    capacity = int(value.untyped_storage().nbytes() // value.element_size())
    offset = int(value.storage_offset())
    # Retain the source offset at runtime. Storage size is part of this static
    # capture contract; clients must supply the captured backing allocation.
    linear_type = MemRefType(elem, [capacity * components], StridedLayoutAttr([1], None))
    linear = memref.ReinterpretCastOp.from_dynamic(
        borrowed.results[0], [descriptor.offset], [capacity * components], [1], linear_type)
    e = Emitter([borrowed, descriptor, linear])
    empty = e.emit(tensor.EmptyOp([], TensorType(elem, shape)))
    retained = min(prod(shape), max(0, capacity - offset) * components)
    def body(b, j, output):
        loaded = b.emit(memref.LoadOp.get(linear.result, [j]))
        coords = []
        for axis, extent in enumerate(shape):
            q = b.binary(arith.DivUIOp, j, b.integer(prod(shape[axis + 1:]), IndexType()))
            coords.append(b.binary(arith.RemUIOp, q, b.integer(extent, IndexType())))
        return [b.emit(tensor.InsertOp(loaded, output, coords))]
    output, = e.loop(retained, [empty], body)
    return DecompResult(ops=e.ops, result=output, region_ids=[_next_region_id('resize_storage')], pattern_hint='view')
