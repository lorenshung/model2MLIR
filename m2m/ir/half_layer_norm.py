"""CPU half LayerNorm moments and affine evaluation in float opmath.

Follow ATen/native/cpu/moments_utils.h, including zero-count lane merges:
those operations matter at IEEE overflow and NaN boundaries.
"""
from math import prod
import platform
import torch
from xdsl.dialects import arith, math, tensor
from xdsl.dialects.builtin import AffineMapAttr, IndexType, TensorType, f32, i64
from xdsl.dialects.linalg import GenericOp, IndexOp, IteratorType, IteratorTypeAttr, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from .decompositions import _broadcast_map, _elementwise
from .torch_cpu_rng import Emitter


def build_half_native(x, weight, bias, *, eps, k):
    shape = list(x.type.get_shape())
    outer, normalized = shape[:-k], shape[-k:]
    count = prod(normalized)
    keep = [*outer, *([1] * k)]
    # LayerNorm uses REGISTER_DISPATCH, which deliberately does not register
    # AVX512 (DispatchStub.h); AVX512 hosts run its AVX2 eight-float-lane kernel.
    capability = torch.backends.cpu.get_cpu_capability()
    lanes = 4 if platform.machine() in ('aarch64', 'arm64') and capability != 'SVE256' else 8
    width = lanes * 2
    vectors, tail = divmod(count, width)
    chunks = (vectors + 15) // 16
    depth = max(1, (chunks - 1).bit_length())

    block = Block(arg_types=[f32, f32])
    row = []
    for axis in range(len(outer)):
        index = IndexOp(axis); block.add_op(index); row.append(index.result)
    e = Emitter()
    def c(value): return e.floating(float(value), f32)
    def add(a, b): return e.binary(arith.AddfOp, a, b)
    def sub(a, b): return e.binary(arith.SubfOp, a, b)
    def mul(a, b): return e.binary(arith.MulfOp, a, b)
    def fma(a, b, d): return e.emit(math.FmaOp(a, b, d))
    def read(b, position):
        coords = list(row)
        for axis, extent in enumerate(normalized):
            q = b.binary(arith.DivUIOp, position, b.integer(prod(normalized[axis + 1:]), IndexType()))
            coords.append(b.binary(arith.RemUIOp, q, b.integer(extent, IndexType())))
        return b.emit(tensor.ExtractOp(x, coords, f32))

    zero = c(0.)
    def tail_body(b, j, mean, m2):
        position = b.binary(arith.AddiOp, j, b.integer(vectors * width, IndexType()))
        value = read(b, position)
        n = b.binary(arith.AddiOp, j, b.integer(1, IndexType()))
        n = b.emit(arith.IndexCastOp(n, i64))
        n = b.emit(arith.SIToFPOp(n, f32))
        delta = b.binary(arith.SubfOp, value, mean)
        mean = b.binary(arith.AddfOp, mean, b.binary(arith.DivfOp, delta, n))
        m2 = b.emit(math.FmaOp(delta, b.binary(arith.SubfOp, value, mean), m2))
        return [mean, m2]
    mean, m2 = e.loop(tail, [zero, zero], tail_body)
    total = tail

    def merge_vector(previous, incoming):
        old_n, old_mean, old_m2 = previous
        new_n, new_mean, new_m2 = incoming
        n = old_n + new_n
        coefficient = c(new_n / n if n else 0.)
        delta = sub(new_mean, old_mean)
        cd = mul(coefficient, delta)
        return n, add(old_mean, cd), fma(mul(delta, c(old_n)), cd, add(old_m2, new_m2))

    for lane in range(lanes):
        stack = [(0, zero, zero) for _ in range(depth)]
        for chunk in range(chunks):
            length = min(16, vectors - chunk * 16)
            def vector_body(b, j, mean0, m20, mean1, m21):
                position = b.binary(arith.AddiOp, j, b.integer(chunk * 16, IndexType()))
                position = b.binary(arith.MuliOp, position, b.integer(width, IndexType()))
                position = b.binary(arith.AddiOp, position, b.integer(lane, IndexType()))
                n = b.binary(arith.AddiOp, j, b.integer(1, IndexType()))
                n = b.emit(arith.IndexCastOp(n, i64))
                n = b.emit(arith.SIToFPOp(n, f32))
                coefficient = b.binary(arith.DivfOp, b.floating(1., f32), n)
                values = [read(b, position), read(b, b.binary(arith.AddiOp, position, b.integer(lanes, IndexType())))]
                outputs = []
                for value, old_mean, old_m2 in zip(values, (mean0, mean1), (m20, m21)):
                    delta = b.binary(arith.SubfOp, value, old_mean)
                    new_mean = b.emit(math.FmaOp(delta, coefficient, old_mean))
                    new_m2 = b.emit(math.FmaOp(delta, b.binary(arith.SubfOp, value, new_mean), old_m2))
                    outputs.extend([new_mean, new_m2])
                return outputs
            a, b, d, f = e.loop(length, [zero] * 4, vector_body)
            stack[0] = merge_vector(stack[0], (length, a, b))
            stack[0] = merge_vector(stack[0], (length, d, f))
            mask = chunk + 1
            for level in range(1, depth):
                if mask & 1: break
                stack[level] = merge_vector(stack[level], stack[level - 1])
                stack[level - 1] = (0, zero, zero)
                mask >>= 1
        for level in range(1, depth):
            stack[0] = merge_vector(stack[0], stack[level])
        n, lane_mean, lane_m2 = stack[0]
        coefficient = c(n / (total + n) if total + n else 0.)
        delta = sub(lane_mean, mean)
        mean = fma(coefficient, delta, mean)
        correction = fma(mul(mul(delta, delta), coefficient), c(total), lane_m2)
        m2 = add(m2, correction)
        total += n
    variance = e.binary(arith.DivfOp, m2, c(count))
    denominator = e.emit(math.SqrtOp(add(variance, c(eps))))
    rstd = e.binary(arith.DivfOp, c(1.), denominator)
    block.add_ops(e.ops)
    block.add_op(YieldOp(mean, rstd))
    typ = TensorType(f32, keep)
    empties = [tensor.EmptyOp([], typ) for _ in range(2)]
    moments = GenericOp(inputs=[], outputs=[v.tensor for v in empties], body=Region(block),
                        indexing_maps=[AffineMapAttr(AffineMap.identity(len(keep)))] * 2,
                        iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)] * len(keep), result_types=[typ, typ])
    ops = [*empties, moments]
    mean, rstd = moments.results
    inputs = [x, mean, rstd] + [v for v in (weight, bias) if v is not None]
    maps = [_broadcast_map(list(v.type.get_shape()), shape) for v in inputs]
    def apply(args, elem):
        b = Emitter()
        offset = b.emit(arith.NegfOp(b.binary(arith.MulfOp, args[2], args[1])))
        output = b.binary(arith.AddfOp, b.binary(arith.MulfOp, args[0], args[2]), offset)
        position = 3
        if weight is not None:
            output = b.binary(arith.MulfOp, output, args[position]); position += 1
        if bias is not None: output = b.binary(arith.AddfOp, output, args[position])
        return b.ops, output
    emitted = _elementwise(inputs, TensorType(f32, shape), apply, input_maps=maps)
    return [*ops, *emitted[0]], emitted[1], mean, rstd
