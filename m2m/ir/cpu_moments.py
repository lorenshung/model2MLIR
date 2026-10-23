"""ATen CPU RowwiseMoments, including vector Welford and scalar lane merges.

The native moments_utils.h uses sixteen-vector chunks and a binary cascade.
Its vector updates and final lane merges use FMA; scalar Welford tails retain
their separate arithmetic. Empty lanes must merge: 0*inf propagates NaN.
"""

from xdsl.dialects import arith, tensor
from xdsl.dialects.math import FmaOp
from xdsl.dialects.builtin import IndexType, i64

from .builder import Builder


def rowwise_moments(b, count, load, *, vector_width, accumulator_width):
    """Emit the mean and population variance of one row of runtime scalars."""
    nvectors = count // vector_width
    chunks = (nvectors + 15) // 16
    depth = max(1, (chunks - 1).bit_length())
    zero = b.fp(0)

    def add_scalar(bb, n0, mean, m2, nadd, madd, vadd):
        coefficient = bb.div(bb.fp(nadd), bb.fp(n0 + nadd)) if n0 + nadd else bb.fp(0)
        delta = bb.sub(madd, mean)
        mean = bb.put(FmaOp(coefficient, delta, mean))
        term = bb.mul(bb.mul(delta, delta), coefficient)
        correction = bb.put(FmaOp(term, bb.fp(n0), vadd))
        return mean, bb.add(m2, correction)

    def add_vector(bb, n0, mean, m2, nadd, madd, vadd):
        coefficient = bb.div(bb.fp(nadd), bb.fp(n0 + nadd)) if n0 + nadd else bb.fp(0)
        delta = bb.sub(madd, mean)
        cd = bb.mul(coefficient, delta)
        md = bb.mul(delta, bb.fp(n0))
        return bb.add(mean, cd), bb.put(FmaOp(md, cd, bb.add(m2, vadd)))

    # One accumulator per float SIMD lane. Half vectors update the lower and
    # upper float halves independently and merge them in that order.
    stack_counts = [0] * depth
    stack_means = [[zero] * accumulator_width for _ in range(depth)]
    stack_m2 = [[zero] * accumulator_width for _ in range(depth)]
    for chunk in range(chunks):
        length = min(16, nvectors - chunk * 16)
        for half in range(vector_width // accumulator_width):
            def update(bb, j, *state):
                means, squares = list(state[:accumulator_width]), list(state[accumulator_width:])
                one_based = bb.iadd(j, bb.ix(1))
                integer = bb.put(arith.IndexCastOp(one_based, i64))
                denominator = bb.put(arith.SIToFPOp(integer, bb.elem))
                coefficient = bb.div(bb.fp(1), denominator)
                for lane in range(accumulator_width):
                    base = bb.iadd(bb.imul(j, bb.ix(vector_width)),
                                   bb.ix(chunk * 16 * vector_width + half * accumulator_width + lane))
                    value = load(bb, base)
                    delta = bb.sub(value, means[lane])
                    means[lane] = bb.put(FmaOp(delta, coefficient, means[lane]))
                    squares[lane] = bb.put(FmaOp(delta, bb.sub(value, means[lane]), squares[lane]))
                return [*means, *squares]
            state = b.loop(0, length, 1, [zero] * (2 * accumulator_width), update)
            for lane in range(accumulator_width):
                stack_means[0][lane], stack_m2[0][lane] = add_vector(
                    b, stack_counts[0], stack_means[0][lane], stack_m2[0][lane], length,
                    state[lane], state[accumulator_width + lane])
            stack_counts[0] += length
        mask = chunk + 1
        for level in range(1, depth):
            if mask & 1:
                break
            for lane in range(accumulator_width):
                stack_means[level][lane], stack_m2[level][lane] = add_vector(
                    b, stack_counts[level], stack_means[level][lane], stack_m2[level][lane],
                    stack_counts[level - 1], stack_means[level - 1][lane], stack_m2[level - 1][lane])
            stack_counts[level] += stack_counts[level - 1]
            stack_counts[level - 1] = 0
            stack_means[level - 1] = [zero] * accumulator_width
            stack_m2[level - 1] = [zero] * accumulator_width
            mask >>= 1
    for level in range(1, depth):
        for lane in range(accumulator_width):
            stack_means[0][lane], stack_m2[0][lane] = add_vector(
                b, stack_counts[0], stack_means[0][lane], stack_m2[0][lane],
                stack_counts[level], stack_means[level][lane], stack_m2[level][lane])
        stack_counts[0] += stack_counts[level]

    start = nvectors * vector_width
    def tail(bb, i, mean, m2):
        value = load(bb, i)
        delta = bb.sub(value, mean)
        ordinal = bb.iadd(bb.put(arith.SubiOp(i, bb.ix(start))), bb.ix(1))
        integer = bb.put(arith.IndexCastOp(ordinal, i64))
        denominator = bb.put(arith.SIToFPOp(integer, bb.elem))
        mean = bb.add(mean, bb.div(delta, denominator))
        m2 = bb.add(m2, bb.mul(delta, bb.sub(value, mean)))
        return [mean, m2]
    mean, m2 = b.loop(start, count, 1, [zero, zero], tail)
    m0, nadd = count - start, nvectors * vector_width // accumulator_width
    for lane in range(accumulator_width):
        mean, m2 = add_scalar(b, m0, mean, m2, nadd, stack_means[0][lane], stack_m2[0][lane])
        m0 += nadd
    return mean, b.div(m2, b.fp(count))


def moments_tensor(x, *, storage_bits):
    """Moments of the last dimension, in opmath dtype, with a final size-one axis."""
    import torch
    from .decompositions import _make_empty
    from xdsl.dialects import linalg
    from xdsl.dialects.builtin import AffineMapAttr, TensorType
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap
    shape = list(x.type.get_shape())
    elem = x.type.element_type
    bits = 512 if torch.backends.cpu.get_cpu_capability() == "AVX512" else 256
    result_type = TensorType(elem, [*shape[:-1], 1])
    empty_mean, empty_var = _make_empty(result_type), _make_empty(result_type)
    body = Block(arg_types=[elem, elem])
    b = Builder(body, elem)
    coords = [b.put(linalg.IndexOp(i)) for i in range(len(shape) - 1)]
    def load(bb, index): return bb.put(tensor.ExtractOp(x, [*coords, index], elem))
    mean, variance = rowwise_moments(b, shape[-1], load,
                                   vector_width=bits // storage_bits, accumulator_width=bits // elem.bitwidth)
    body.add_op(linalg.YieldOp(mean, variance))
    result = linalg.GenericOp(inputs=[], outputs=[empty_mean.results[0], empty_var.results[0]],
        body=Region(body), indexing_maps=[AffineMapAttr(AffineMap.identity(len(shape)))] * 2,
        iterator_types=[linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * len(shape),
        result_types=[result_type, result_type])
    return [empty_mean, empty_var, result], *result.results


def columnwise_moments(x, *, storage_bits):
    """Channels-last GroupNorm's per-lane sums of x and x*x.

    x has logical shape [N, groups, channels_per_group, spatial]. Native
    small-feature-map kernels visit spatial positions before channel vectors.
    """
    import torch
    from .decompositions import _make_empty
    from xdsl.dialects import linalg
    from xdsl.dialects.builtin import AffineMapAttr, TensorType
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    n, groups, channels, spatial = x.type.get_shape()
    elem = x.type.element_type
    bits = 512 if torch.backends.cpu.get_cpu_capability() == "AVX512" else 256
    lanes, storage_lanes = bits // elem.bitwidth, bits // storage_bits
    typ = TensorType(elem, [n, groups, 1])
    outputs = [_make_empty(typ), _make_empty(typ)]
    body = Block(arg_types=[elem, elem])
    b = Builder(body, elem)
    row = [b.put(linalg.IndexOp(i)) for i in (0, 1)]
    zero = b.fp(0)

    def load(bb, channel, position):
        if channel >= channels:
            return zero
        return bb.put(tensor.ExtractOp(x, [*row, bb.ix(channel), position], elem))

    if spatial < 1024:
        def accumulate(bb, position, *state):
            sums, squares = list(state[:lanes]), list(state[lanes:])
            for start in range(0, channels, storage_lanes):
                for lane in range(lanes):
                    a = load(bb, start + lane, position)
                    square = bb.mul(a, a)
                    if storage_lanes > lanes and channels - start > lanes:
                        other = load(bb, start + lanes + lane, position)
                        a = bb.add(a, other)
                        square = bb.add(square, bb.mul(other, other))
                    sums[lane] = bb.add(sums[lane], a)
                    squares[lane] = bb.add(squares[lane], square)
            return [*sums, *squares]
        state = b.loop(0, spatial, 1, [zero] * (2 * lanes), accumulate)
        totals = []
        for partials in (state[:lanes], state[lanes:]):
            while len(partials) > 1:
                half = len(partials) // 2
                partials = [b.add(a, v) for a, v in zip(partials[:half], partials[half:])]
            totals.append(partials[0])
    else:
        # The large-feature-map implementation accumulates each channel's
        # spatial sum first, then folds channels (and thread buffers).
        if torch.get_num_threads() != 1:
            raise ValueError("large channels-last GroupNorm requires a fixed single-thread CPU reduction policy")
        def accumulate(bb, position, *state):
            sums, squares = list(state[:channels]), list(state[channels:])
            for channel in range(channels):
                value = load(bb, channel, position)
                sums[channel] = bb.add(sums[channel], value)
                squares[channel] = bb.put(FmaOp(value, value, squares[channel]))
            return [*sums, *squares]
        state = b.loop(0, spatial, 1, [zero] * (2 * channels), accumulate)
        totals = []
        for partials in (state[:channels], state[channels:]):
            total = zero
            for partial in partials:
                total = b.add(total, partial)
            totals.append(total)
    scale = b.div(b.fp(1), b.fp(channels * spatial))
    mean = b.mul(totals[0], scale)
    square_mean = b.mul(mean, mean)
    variance = b.put(FmaOp(totals[1], scale, b.put(arith.NegfOp(square_mean))))
    body.add_op(linalg.YieldOp(mean, variance))
    result = linalg.GenericOp(inputs=[], outputs=[op.results[0] for op in outputs],
        body=Region(body), indexing_maps=[AffineMapAttr(AffineMap.identity(3))] * 2,
        iterator_types=[linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * 3,
        result_types=[typ, typ])
    return [*outputs, result], *result.results
