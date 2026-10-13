"""ATen CPU short contiguous sums: eight independent scalar accumulators."""

from math import prod

import torch
from xdsl.dialects import arith, tensor
from xdsl.dialects.builtin import IndexType, f32

from .structural import generate
from .torch_cpu_rng import Emitter


def short_contiguous_sum(source, shape, source_value):
    if source_value is None or not source_value.is_contiguous() or source.type.element_type != f32:
        return None
    count = prod(shape)
    capability = torch.backends.cpu.get_cpu_capability()
    # Sum uses REGISTER_DISPATCH rather than the AVX512 override, like
    # native LayerNorm: AVX512 hosts still select the eight-float AVX2 path.
    lanes = 8 if capability in ("AVX512", "AVX2", "SVE256") else 4
    if count >= 4 * lanes:
        return None

    def build(block, coords):
        e = Emitter()
        zero = e.floating(0.0, f32)

        def read(b, position):
            coordinates = []
            for axis, extent in enumerate(shape):
                q = b.binary(arith.DivUIOp, position, b.integer(prod(shape[axis + 1 :]), IndexType()))
                coordinates.append(b.binary(arith.RemUIOp, q, b.integer(extent, IndexType())))
            return b.emit(tensor.ExtractOp(source, coordinates, f32))

        def body(b, j, *accumulators):
            start = b.binary(arith.MuliOp, j, b.integer(8, IndexType()))
            return [
                b.binary(
                    arith.AddfOp, accumulator, read(b, b.binary(arith.AddiOp, start, b.integer(lane, IndexType())))
                )
                for lane, accumulator in enumerate(accumulators)
            ]

        accumulators = e.loop(count // 8, [zero] * 8, body)

        def tail(b, j, accumulator):
            position = b.binary(arith.AddiOp, j, b.integer(count // 8 * 8, IndexType()))
            return [b.binary(arith.AddfOp, accumulator, read(b, position))]

        (first,) = e.loop(count % 8, [accumulators[0]], tail)
        value = first
        for accumulator in accumulators[1:]:
            value = e.binary(arith.AddfOp, value, accumulator)
        block.add_ops(e.ops)
        return value

    return generate([], f32, build)
