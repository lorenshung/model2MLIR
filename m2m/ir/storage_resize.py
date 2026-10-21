"""resize_ reads backing storage, not the flattened logical tensor elements."""
from math import prod
import torch
from xdsl.dialects import memref, tensor
from xdsl.dialects.builtin import MemRefType, StridedLayoutAttr, TensorType, UnregisteredOp
from .decompositions import DecompResult, _fx_arg, _next_region_id
from .torch_cpu_rng import Emitter


def read_storage(source, shape, span, count):
    """Load the first ``count`` of ``span`` backing-storage scalars, from the source's
    runtime offset, into a new row-major tensor of ``shape``."""
    old_shape = list(source.type.get_shape())
    elem = source.type.element_type
    # A descriptor includes offset/strides even though tensor types do not.
    typ = MemRefType(elem, old_shape, StridedLayoutAttr([None] * len(old_shape), None))
    # xDSL's ToBufferOp inverse tensor constraint incorrectly discards layouts.
    # Emit the standard operation directly until that upstream constraint is fixed.
    borrowed = UnregisteredOp.with_name('bufferization.to_buffer').create(operands=[source], result_types=[typ])
    descriptor = memref.ExtractStridedMetaDataOp(borrowed.results[0])
    linear_type = MemRefType(elem, [span], StridedLayoutAttr([1], None))
    linear = memref.ReinterpretCastOp.from_dynamic(borrowed.results[0], [descriptor.offset], [span], [1], linear_type)
    e = Emitter(ops=[borrowed, descriptor, linear])
    empty = e.put(tensor.EmptyOp([], TensorType(elem, shape)))
    def body(b, j, output):
        loaded = b.put(memref.LoadOp.get(linear.result, [j]))
        return [b.put(tensor.InsertOp(loaded, output, b.coordinates(j, shape)))]
    output, = e.loop(0, count, 1, [empty], body)
    return e.ops, output


def resize_storage(operands, meta, node_name):
    source = operands[0]
    value = _fx_arg(meta, 0).meta['val']
    result = meta['val']
    if value.dtype.is_complex and value.dtype != torch.complex64:
        raise ValueError('storage resize supports the complex64 pair ABI only')
    pair = value.dtype == torch.complex64
    components = 2 if pair else 1
    shape = list(result.shape) + ([2] if pair else [])
    if list(result.shape) == list(value.shape):
        return DecompResult(ops=[], result=source, pattern_hint='view')
    capacity = int(value.untyped_storage().nbytes() // value.element_size())
    offset = int(value.storage_offset())
    # Retain the source offset at runtime. Storage size is part of this static
    # capture contract; clients must supply the captured backing allocation.
    retained = min(prod(shape), max(0, capacity - offset) * components)
    ops, output = read_storage(source, shape, capacity * components, retained)
    return DecompResult(ops=ops, result=output, region_ids=[_next_region_id('resize_storage')], pattern_hint='view')
