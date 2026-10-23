"""ATen CPU sum traversal: TensorIterator runs, four-way ILP and cascade sums.

Only static layout and dtype choose an order. All data remains runtime SSA.
The policies follow native/cpu/SumKernel.cpp (multi_row_sum, row_sum and
vectorized_inner_sum); they are deliberately not reassociable reductions.
"""

from math import prod

from xdsl.dialects import arith, tensor

from .builder import Builder


def multi_sum(b, count, loads):
    """Sum each load stream, with the native four cascade levels."""
    nrows = len(loads)
    power = max(4, (max(1, count) - 1).bit_length() // 4)
    step = 1 << power
    levels = 4 if count >= step else 1
    zero = b.fp(0)

    def body(bb, i, *values):
        values = list(values)
        for k, load in enumerate(loads):
            values[k] = bb.add(values[k], load(bb, i))
        if levels > 1:
            next_i = bb.iadd(i, bb.ix(1))
            remainder = bb.put(arith.RemUIOp(next_i, bb.ix(step)))
            carry = bb.put(arith.CmpiOp(remainder, bb.ix(0), "eq"))
            for level in range(1, levels):
                old = list(values)
                for k in range(nrows):
                    total = bb.add(values[level * nrows + k], values[(level - 1) * nrows + k])
                    values[level * nrows + k] = bb.put(arith.SelectOp(carry, total, old[level * nrows + k]))
                    values[(level - 1) * nrows + k] = bb.put(arith.SelectOp(carry, zero, old[(level - 1) * nrows + k]))
                mask = (step - 1) << (level * power)
                bits = bb.put(arith.AndIOp(next_i, bb.ix(mask)))
                even = bb.put(arith.CmpiOp(bits, bb.ix(0), "eq"))
                carry = bb.put(arith.AndIOp(carry, even))
        return values

    result = b.loop(0, count, 1, [zero] * (levels * nrows), body)
    for level in range(1, levels):
        for k in range(nrows):
            result[k] = b.add(result[k], result[level * nrows + k])
    return result[:nrows]


def row_sum(b, count, load):
    """Native four interleaved sums, remainder into lane zero, then lanes 1..3."""
    streams = []
    for lane in range(4):
        def at(bb, i, lane=lane):
            return load(bb, bb.iadd(bb.imul(i, bb.ix(4)), bb.ix(lane)))
        streams.append(at)
    partials = multi_sum(b, count // 4, streams)
    def tail(bb, i, total):
        return [bb.add(total, load(bb, i))]
    partials[0], = b.loop(count // 4 * 4, count, 1, [partials[0]], tail)
    for value in partials[1:]:
        partials[0] = b.add(partials[0], value)
    return partials[0]


def serial_sum_tensor(x, dims, source):
    """cpu_serial_kernel traversal without dtype-changing iterator buffers."""
    from .structural import generate
    shape = list(x.type.get_shape())
    elem = x.type.element_type
    axes = [d for d in reversed(range(len(shape))) if d in dims]
    strides = source.stride()
    for i in range(1, len(axes)):
        j = i
        while j and strides[axes[j]] and strides[axes[j - 1]] and strides[axes[j]] < strides[axes[j - 1]]:
            axes[j], axes[j - 1] = axes[j - 1], axes[j]
            j -= 1
    kept = [d for d in range(len(shape)) if d not in dims]
    def build(block, coords):
        b = Builder(block, elem)
        def body(bb, linear, total):
            indices = [bb.ix(0) for _ in shape]
            for axis, index in zip(kept, coords):
                indices[axis] = index
            for axis in axes:
                indices[axis] = bb.put(arith.RemUIOp(linear, bb.ix(shape[axis])))
                linear = bb.put(arith.DivUIOp(linear, bb.ix(shape[axis])))
            return [bb.add(total, bb.put(tensor.ExtractOp(x, indices, elem)))]
        value, = b.loop(0, prod(shape[d] for d in dims), 1, [b.fp(0)], body)
        return value
    return generate([shape[d] for d in kept], elem, build)


def sum_tensor(x, dims, source):
    """Return a CPU-ordered sum after any input dtype promotion already in SSA."""
    import torch
    from .structural import generate

    shape = list(x.type.get_shape())
    elem = x.type.element_type
    strides = list(source.stride())
    # A dtype-changing to(preserve_format) densifies padded/expanded storage.
    if source.element_size() * 8 != elem.bitwidth:
        from torch._prims_common import is_non_overlapping_and_dense
        if not is_non_overlapping_and_dense(source):
            stride = 1
            for axis in reversed(range(len(shape))):
                strides[axis] = stride
                stride *= shape[axis]
    axes = [d for d in reversed(range(len(shape))) if d in dims]
    # TensorIterator ignores zero strides when choosing between input axes.
    for i in range(1, len(axes)):
        j = i
        while j and strides[axes[j]] and strides[axes[j - 1]] and strides[axes[j]] < strides[axes[j - 1]]:
            axes[j], axes[j - 1] = axes[j - 1], axes[j]
            j -= 1
    groups = []
    for axis in axes:
        if shape[axis] == 1:
            continue
        if groups and strides[axis] == strides[groups[-1][0]] * prod(shape[d] for d in groups[-1]):
            groups[-1].append(axis)
        else:
            groups.append([axis])
    inner = groups[0] if groups else []
    inner_size = prod(shape[d] for d in inner)
    inner_stride = strides[inner[0]] if inner else 1
    kept = [d for d in range(len(shape)) if d not in dims]
    outer_axes = [d for group in reversed(groups[1:]) for d in reversed(group)]
    outer_size = prod(shape[d] for d in outer_axes)
    nonreduce_stride = min((strides[d] for d in kept if shape[d] > 1 and strides[d]), default=float("inf"))
    vector_width = (512 if torch.backends.cpu.get_cpu_capability() == "AVX512" else 256) // elem.bitwidth
    vectorized = inner_stride == 1 and inner_size >= vector_width

    def build(block, output_coords):
        b = Builder(block, elem)
        def outer(bb, linear, total):
            coords = [bb.ix(0) for _ in shape]
            for axis, value in zip(kept, output_coords):
                coords[axis] = value
            for axis in reversed(outer_axes):
                coords[axis] = bb.put(arith.RemUIOp(linear, bb.ix(shape[axis])))
                linear = bb.put(arith.DivUIOp(linear, bb.ix(shape[axis])))
            def load(bbb, index):
                current = list(coords)
                for axis in inner:
                    current[axis] = bbb.put(arith.RemUIOp(index, bbb.ix(shape[axis])))
                    index = bbb.put(arith.DivUIOp(index, bbb.ix(shape[axis])))
                return bbb.put(tensor.ExtractOp(x, current, elem))
            if vectorized:
                nvec = inner_size // vector_width
                partials = []
                for lane in range(vector_width):
                    def vector_load(bbb, i, lane=lane):
                        return load(bbb, bbb.iadd(bbb.imul(i, bbb.ix(vector_width)), bbb.ix(lane)))
                    partials.append(row_sum(bb, nvec, vector_load))
                def tail(bbb, i, acc): return [bbb.add(acc, load(bbb, i))]
                value, = bb.loop(nvec * vector_width, inner_size, 1, [bb.fp(0)], tail)
                for partial in partials:
                    value = bb.add(value, partial)
            elif inner_stride < nonreduce_stride:
                value = row_sum(bb, inner_size, load)
            else:
                value, = multi_sum(bb, inner_size, [load])
            return [bb.add(total, value)]
        value, = b.loop(0, outer_size if prod(shape[d] for d in dims) else 0, 1, [b.fp(0)], outer)
        return value

    return generate([shape[d] for d in kept], elem, build)
