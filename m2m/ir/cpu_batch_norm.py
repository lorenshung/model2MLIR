"""CPU BatchNorm collect-stats storage and accumulation boundaries."""

from math import prod

from xdsl.dialects import arith, tensor
from xdsl.dialects.builtin import f32, f64
from xdsl.dialects.math import FmaOp

from .cpu_reductions import Builder
from .structural import generate


def contiguous_stats(x, source, storage_elem):
    """Native NCHW scalar/double or reduced-precision SIMD collect-stats."""
    import torch

    shape = list(x.type.get_shape())
    elem = x.type.element_type
    spatial = prod(shape[2:])
    count = shape[0] * spatial
    channels_last = not source.is_contiguous()
    reduced = storage_elem.bitwidth == 16
    bits = 512 if torch.backends.cpu.get_cpu_capability() == "AVX512" else 256
    width = bits // 32
    storage_width = bits // storage_elem.bitwidth
    acc_elem = f64 if not channels_last and not reduced and elem == f32 else elem

    def stats(block, coords, variance=False, means=None):
        b = Builder(block, elem)

        def cast(bb, value, dtype):
            if value.type == dtype:
                return value
            op = arith.ExtFOp if value.type.bitwidth < dtype.bitwidth else arith.TruncFOp
            return bb.put(op(value, dtype))

        def load(bb, linear):
            n = bb.put(arith.DivUIOp(linear, bb.ix(spatial)))
            offset = bb.put(arith.RemUIOp(linear, bb.ix(spatial)))
            indices = [n, coords[0]]
            tail = []
            for size in reversed(shape[2:]):
                tail.append(bb.put(arith.RemUIOp(offset, bb.ix(size))))
                offset = bb.put(arith.DivUIOp(offset, bb.ix(size)))
            return bb.put(tensor.ExtractOp(x, [*indices, *reversed(tail)], elem))

        mean = b.put(tensor.ExtractOp(means, coords, elem)) if variance else None

        def term(bb, linear):
            value = load(bb, linear)
            if variance:
                value = bb.sub(value, mean)
                value = bb.mul(value, value)
            return value

        if reduced and not channels_last:
            # Native halves share one float-vector accumulator: low half,
            # then high half, plus a separate scalar tail for each image.
            limit = spatial // storage_width * storage_width
            def batch(bb, n, scalar, *lanes):
                base = bb.imul(n, bb.ix(spatial))
                def vectors(bbb, d, *state):
                    state = list(state)
                    for half in range(2):
                        for lane in range(width):
                            index = bbb.iadd(base, bbb.iadd(d, bbb.ix(half * width + lane)))
                            state[lane] = bbb.add(state[lane], term(bbb, index))
                    return state
                lanes = bb.loop(0, limit, storage_width, lanes, vectors)
                def tail(bbb, d, value):
                    return [bbb.add(value, term(bbb, bbb.iadd(base, d)))]
                scalar, = bb.loop(limit, spatial, 1, [scalar], tail)
                return [scalar, *lanes]
            scalar, *lanes = b.loop(0, shape[0], 1, [b.fp(0)] * (width + 1), batch)
            # vec_reduce_all's scalar fallback visits lanes in order here.
            lane_sum = lanes[0]
            for lane in lanes[1:]:
                lane_sum = b.add(lane_sum, lane)
            total = b.add(scalar, lane_sum)
        else:
            acc = Builder(block, acc_elem)
            def serial(bb, i, total):
                if channels_last and variance:
                    delta = bb.sub(load(bb, i), mean)
                    return [bb.put(FmaOp(delta, delta, total))]
                return [bb.add(total, cast(bb, term(bb, i), acc_elem))]
            total, = acc.loop(0, count, 1, [acc.fp(0)], serial)
        if not variance:
            acc = Builder(block, acc_elem)
            total = acc.div(total, acc.fp(count))
        return cast(b, total, elem)

    mean_ops, mean = generate([shape[1]], elem, lambda block, coords: stats(block, coords))
    var_ops, variance = generate([shape[1]], elem,
                                 lambda block, coords: stats(block, coords, True, mean))
    return [*mean_ops, *var_ops], mean, variance
