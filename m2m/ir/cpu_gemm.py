"""Ordered CPU small GEMM policies, expressed with portable scalar MLIR.

MKL's AMD small-SGEMM path uses SSE multiply/add (including scalar peeling),
not a reassociable dot product. ATen's reduced-width NN fallback uses four
independent float partial sums. These are source CPU policies; integerized
contractions use their separate exact integer lowering.
"""

from pathlib import Path

from xdsl.dialects import arith, scf, tensor
from xdsl.dialects.builtin import f32
from xdsl.ir import Block, Region

from .cpu_reductions import Builder
from .structural import generate


def mkl_sse_small_kernel():
    import torch
    import os
    return (torch.backends.mkl.is_available()
            and torch.backends.cpu.get_cpu_capability() == "AVX2"
            and Path("/proc/cpuinfo").exists()
            and "AuthenticAMD" in Path("/proc/cpuinfo").read_text()
            and not any(os.environ.get(k) for k in ("MKL_CBWR", "MKL_ENABLE_INSTRUCTIONS")))


def choose(b, condition, yes, no):
    branches = []
    for fn in (yes, no):
        block = Block()
        value = fn(Builder(block, b.elem))
        block.add_op(scf.YieldOp(value))
        branches.append(Region(block))
    return b.put(scf.IfOp(condition, [b.elem], *branches))


def tn_dot(b, count, load, offset, trailing):
    """Fortran TN: groups of four outputs use one vector; tails use two."""
    def aligned(bb, prefix, width):
        if count < prefix + width:
            prefix = count
        def serial(bbb, i, value):
            a, c = load(bbb, i)
            return [bbb.add(value, bbb.mul(a, c))]
        value, = bb.loop(0, prefix, 1, [bb.fp(0)], serial)
        if prefix == count:
            return value
        end = prefix + (count - prefix) // width * width
        def vector(bbb, i, *state):
            state = list(state)
            for lane in range(width):
                a, c = load(bbb, bbb.iadd(i, bbb.ix(lane)))
                state[lane] = bbb.add(state[lane], bbb.mul(a, c))
            return state
        lanes = bb.loop(prefix, end, width, [value, *[bb.fp(0)] * (width - 1)], vector)
        if width == 8:
            lanes = [bb.add(lanes[i], lanes[i + 4]) for i in range(4)]
        value = bb.add(bb.add(lanes[0], lanes[2]), bb.add(lanes[1], lanes[3]))
        value, = bb.loop(end, count, 1, [value], serial)
        return value

    def width(bb, size):
        value = aligned(bb, 0, size)
        for remainder in range(1, 4):
            test = bb.put(arith.CmpiOp(offset, bb.ix(remainder), "eq"))
            other = aligned(bb, 4 - remainder, size)
            value = bb.put(arith.SelectOp(test, other, value))
        return value
    return choose(b, trailing, lambda bb: width(bb, 8), lambda bb: width(bb, 4))


def nn_dot(b, count, load, seed, output_offset, output_extent, alpha=1):
    """Fortran NN: alpha scales B first; peeled elements split each K=8 tile."""
    def product(bb, index):
        a, c = load(bb, index)
        if alpha != 1:
            a = bb.mul(a, bb.fp(alpha))
        return bb.mul(a, c)

    def policy(bb, peeled):
        def tile(bbb, k, value):
            first = value
            for lane in range(4):
                first = bbb.add(first, product(bbb, bbb.iadd(k, bbb.ix(lane))))
            if peeled:
                second = product(bbb, bbb.iadd(k, bbb.ix(4)))
                for lane in range(5, 8):
                    second = bbb.add(second, product(bbb, bbb.iadd(k, bbb.ix(lane))))
                return [bbb.add(first, second)]
            for lane in range(4, 8):
                first = bbb.add(first, product(bbb, bbb.iadd(k, bbb.ix(lane))))
            return [first]
        value, = bb.loop(0, count // 8 * 8, 8, [seed], tile)
        def tail(bbb, k, value):
            return [bbb.add(value, product(bbb, k))]
        value, = bb.loop(count // 8 * 8, count, 1, [value], tail)
        return value

    # Only the alignment prefix splits K tiles. Vector and scalar epilogues
    # both accumulate all eight terms in order.
    index = b.put(arith.RemUIOp(output_offset, b.ix(output_extent)))
    row = b.put(arith.DivUIOp(output_offset, b.ix(output_extent)))
    alignment = b.put(arith.RemUIOp(b.imul(row, b.ix(output_extent)), b.ix(4)))
    prefix = b.put(arith.RemUIOp(b.put(arith.SubiOp(b.ix(4), alignment)), b.ix(4)))
    enough = b.put(arith.CmpiOp(b.iadd(prefix, b.ix(4)), b.ix(output_extent), "ule"))
    peeled = b.put(arith.AndIOp(enough, b.put(arith.CmpiOp(index, prefix, "ult"))))
    return choose(b, peeled, lambda bb: policy(bb, True), lambda bb: policy(bb, False))


def reduced_nn_dot(b, count, load, seed):
    """BlasKernel.cpp::sum: four ILP streams, tail in stream zero."""
    def body(bb, i, *sums):
        sums = list(sums)
        for lane in range(4):
            a, c = load(bb, bb.iadd(i, bb.ix(lane)))
            sums[lane] = bb.add(sums[lane], bb.mul(a, c))
        return sums
    sums = b.loop(0, count // 4 * 4, 4, [b.fp(0)] * 4, body)
    def tail(bb, i, value):
        a, c = load(bb, i)
        return [bb.add(value, bb.mul(a, c))]
    sums[0], = b.loop(count // 4 * 4, count, 1, [sums[0]], tail)
    for value in sums[1:]:
        sums[0] = b.add(sums[0], value)
    return b.add(seed, sums[0])


def addmm(operands, meta, alpha, beta):
    """Row-major contiguous CPU addmm on MKL's small NN path."""
    args = meta.get("_fx_args", ())
    if len(args) < 3 or not mkl_sse_small_kernel():
        return None
    sources = [getattr(arg, "meta", {}).get("val") for arg in args[1:3]]
    if any(value is None or value.device.type != "cpu" or not value.is_contiguous()
           for value in sources):
        return None
    bias, a, c = operands[:3]
    if a.type.element_type != f32:
        return None
    m, k = a.type.get_shape()
    _, n = c.type.get_shape()
    if min(m, n, k) >= 8:
        return None
    bias_shape = list(bias.type.get_shape())
    def build(block, coords):
        b = Builder(block, f32)
        row, col = coords
        seed = b.fp(0)
        if beta != 0:
            indices = [b.ix(0) if extent == 1 else index
                       for extent, index in zip(bias_shape, coords[2 - len(bias_shape):])]
            seed = b.put(tensor.ExtractOp(bias, indices, f32))
            if beta != 1:
                seed = b.mul(seed, b.fp(beta))
        if alpha == 0:
            return seed
        def load(bb, index):
            return (bb.put(tensor.ExtractOp(a, [row, index], f32)),
                    bb.put(tensor.ExtractOp(c, [index, col], f32)))
        offset = b.iadd(b.imul(row, b.ix(n)), col)
        return nn_dot(b, k, load, seed, offset, n, alpha)
    return generate([m, n], f32, build)


def slow_conv_selected(meta, stride, padding, dilation, groups):
    import torch
    from torch._prims_common import suggest_memory_format
    args = meta.get("_fx_args", ())
    if len(args) < 2:
        return False
    x, w = (getattr(arg, "meta", {}).get("val") for arg in args[:2])
    if x is None or w is None or x.device.type != "cpu" or x.ndim != 4:
        return False
    backend = torch._C._select_conv_backend(x, w, None, stride, padding, dilation,
                                            False, [0, 0], groups)
    return (backend == torch._C._ConvBackend.Slow2d
            and suggest_memory_format(x) == torch.contiguous_format
            and suggest_memory_format(w) == torch.contiguous_format)


def conv_forward(x, weight, bias, out_shape, stride, padding, storage_elem):
    shape, ws = list(x.type.get_shape()), list(weight.type.get_shape())
    kh, kw = ws[2:]
    count, spatial = ws[1] * kh * kw, out_shape[2] * out_shape[3]
    if storage_elem == f32 and not (mkl_sse_small_kernel() and min(count, spatial, ws[0]) < 8):
        return None

    def build(block, coords):
        b = Builder(block, f32)
        n, channel, oh, ow = coords
        def load(bb, k):
            c = bb.put(arith.DivUIOp(k, bb.ix(kh * kw)))
            tap = bb.put(arith.RemUIOp(k, bb.ix(kh * kw)))
            h = bb.iadd(bb.imul(oh, bb.ix(stride[0])), bb.put(arith.DivUIOp(tap, bb.ix(kw))))
            w = bb.iadd(bb.imul(ow, bb.ix(stride[1])), bb.put(arith.RemUIOp(tap, bb.ix(kw))))
            h = bb.put(arith.SubiOp(h, bb.ix(padding[0])))
            w = bb.put(arith.SubiOp(w, bb.ix(padding[1])))
            inside = bb.put(arith.AndIOp(bb.put(arith.CmpiOp(h, bb.ix(shape[2]), "ult")),
                                       bb.put(arith.CmpiOp(w, bb.ix(shape[3]), "ult"))))
            value = choose(bb, inside,
                           lambda bbb: bbb.put(tensor.ExtractOp(x, [n, c, h, w], f32)),
                           lambda bbb: bbb.fp(0))
            wr = bb.put(arith.DivUIOp(tap, bb.ix(kw)))
            wc = bb.put(arith.RemUIOp(tap, bb.ix(kw)))
            coefficient = bb.put(tensor.ExtractOp(weight, [channel, c, wr, wc], f32))
            return coefficient, value
        seed = b.put(tensor.ExtractOp(bias, [channel], f32)) if bias is not None else b.fp(0)
        if storage_elem.bitwidth == 16:
            return reduced_nn_dot(b, count, load, seed)
        offset = b.iadd(b.imul(channel, b.ix(spatial)), b.iadd(b.imul(oh, b.ix(out_shape[3])), ow))
        return nn_dot(b, count, load, seed, offset, spatial)
    return generate(out_shape, f32, build)


def conv_weight_gradient(x, grad, weight_shape, stride, padding, source_grad):
    gs = list(grad.type.get_shape())
    spatial = gs[2] * gs[3]
    columns = weight_shape[1] * weight_shape[2] * weight_shape[3]
    if not (mkl_sse_small_kernel() and min(spatial, columns, gs[1]) < 8):
        return None
    shape = list(x.type.get_shape())
    def build(block, coords):
        b = Builder(block, f32)
        f, c, kh, kw = coords
        column = b.iadd(b.imul(c, b.ix(weight_shape[2] * weight_shape[3])),
                        b.iadd(b.imul(kh, b.ix(weight_shape[3])), kw))
        trailing = b.put(arith.CmpiOp(column, b.ix(columns // 4 * 4), "uge"))
        def batch(bb, n, total):
            # contiguous() preserves an already contiguous source's offset.
            source_offset = source_grad.storage_offset() if source_grad.is_contiguous() else 0
            base = bb.iadd(bb.imul(n, bb.ix(gs[1] * spatial)), bb.imul(f, bb.ix(spatial)))
            offset = bb.put(arith.RemUIOp(bb.iadd(base, bb.ix(source_offset)), bb.ix(4)))
            def load(bbb, i):
                oh = bbb.put(arith.DivUIOp(i, bbb.ix(gs[3])))
                ow = bbb.put(arith.RemUIOp(i, bbb.ix(gs[3])))
                h = bbb.put(arith.SubiOp(bbb.iadd(bbb.imul(oh, bbb.ix(stride[0])), kh), bbb.ix(padding[0])))
                w = bbb.put(arith.SubiOp(bbb.iadd(bbb.imul(ow, bbb.ix(stride[1])), kw), bbb.ix(padding[1])))
                inside = bbb.put(arith.AndIOp(bbb.put(arith.CmpiOp(h, bbb.ix(shape[2]), "ult")),
                                             bbb.put(arith.CmpiOp(w, bbb.ix(shape[3]), "ult"))))
                value = choose(bbb, inside,
                               lambda bbbb: bbbb.put(tensor.ExtractOp(x, [n, c, h, w], f32)),
                               lambda bbbb: bbbb.fp(0))
                dy = bbb.put(tensor.ExtractOp(grad, [n, f, oh, ow], f32))
                return dy, value
            return [bb.add(total, tn_dot(bb, spatial, load, offset, trailing))]
        result, = b.loop(0, gs[0], 1, [b.fp(0)], batch)
        return result
    return generate(weight_shape, f32, build)
