"""Static-shape structural ATen kernels with runtime values and counted loops.

Keep structural indexing and reduction policies separate from arithmetic dtype
promotion. These kernels never consult example tensor contents.
"""

from __future__ import annotations

import math

from xdsl.dialects import arith, scf, tensor
from xdsl.dialects import math as mm
from xdsl.dialects.builtin import (
    AffineMapAttr,
    AnyFloat,
    BFloat16Type,
    Float16Type,
    Float32Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    TensorType,
    i64,
)
from xdsl.dialects.linalg import GenericOp, IndexOp, IteratorType, IteratorTypeAttr, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from m2m.ir.builder import Builder


def scalar(n, elem):
    return arith.ConstantOp(
        IntegerAttr(int(bool(n)) if elem.bitwidth == 1 else int(n), elem)
        if isinstance(elem, IntegerType)
        else FloatAttr(float(n), elem),
        elem,
    )


def delinearize(b, linear, shape):
    """Row-major coordinates of a linear index, peeled from the innermost axis."""
    result = [None] * len(shape)
    for axis in reversed(range(len(shape))):
        size = b.ix(shape[axis])
        result[axis] = b(arith.RemUIOp(linear, size))
        if axis:
            linear = b(arith.DivUIOp(linear, size))
    return result


def coordinates(block, linear, shape):
    return delinearize(Builder(block), linear, [max(extent, 1) for extent in shape])


def generate(shape, elem, build):
    from m2m.ir.decompositions import _make_empty

    rt = TensorType(elem, shape)
    empty = _make_empty(rt)
    block = Block(arg_types=[elem])
    b = Builder(block)
    coords = [b(IndexOp(axis)) for axis in range(len(shape))]
    block.add_op(YieldOp(build(block, coords)))
    op = GenericOp(
        inputs=[],
        outputs=[empty.results[0]],
        body=Region(block),
        indexing_maps=[AffineMapAttr(AffineMap.identity(len(shape)))],
        iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)] * len(shape),
        result_types=[rt],
    )
    return [empty, op], op.results[0]


def diagonal(operands, meta, node_name):
    from m2m.ir.decompositions import DecompResult, _fx_arg, _shape_of, _t_elem

    x = operands[0]
    shape = _shape_of(x)
    offset = int(_fx_arg(meta, 1, 0))
    d1, d2 = (int(_fx_arg(meta, 2, 0)) % len(shape), int(_fx_arg(meta, 3, 1)) % len(shape))

    def build(block, coords):
        b = Builder(block)
        src = []
        remaining = iter(coords[:-1])
        for axis in range(len(shape)):
            if axis in (d1, d2):
                shift = b.ix(max(-offset, 0) if axis == d1 else max(offset, 0))
                src.append(b(arith.AddiOp(coords[-1], shift)))
            else:
                src.append(next(remaining))
        return b(tensor.ExtractOp(x, src, _t_elem(x)))

    ops, result = generate(list(meta["val"].shape), _t_elem(x), build)
    return DecompResult(ops=ops, result=result, pattern_hint="diagonal")


def distance(operands, meta, node_name):
    from m2m.ir.decompositions import DecompResult, _fx_arg, _shape_of, _t_elem

    x = operands[0]
    xs = _shape_of(x)
    elem = _t_elem(x)
    cdist = "cdist" in meta["_aten_target"]
    y = operands[1] if cdist else x
    ys = _shape_of(y)
    p = float(_fx_arg(meta, 2 if cdist else 1, 2.0))
    if not isinstance(elem, AnyFloat) or p < 0 or any(d < 0 for d in [*xs, *ys]):
        raise ValueError("distance requires static real floating tensors and p >= 0")
    out = list(meta["val"].shape)

    def build(block, coords):
        b = Builder(block)
        zero, one, features = b.ix(0), b.ix(1), b.ix(xs[-1])
        init = b(scalar(0, elem))
        if cdist:
            batch = coords[:-2]

            def batch_coords(shape):
                return [zero if size == 1 else c for size, c in zip(shape[:-2], batch[len(batch) - len(shape[:-2]) :])]

            xc, yc = batch_coords(xs) + [coords[-2]], batch_coords(ys) + [coords[-1]]
        else:
            # Locate the triangular row by subtracting successive row lengths.
            search = Block(arg_types=[IndexType(), IndexType(), IndexType()])
            row, remainder, chosen = search.args
            s = Builder(search)
            width = s(arith.SubiOp(s.ix(xs[0] - 1), row))
            fits = s(arith.AndIOp(s(arith.CmpiOp(remainder, width, "sge")), s(arith.CmpiOp(row, chosen, "eq"))))
            reduced = s(arith.SubiOp(remainder, width))
            nextrow = s(arith.AddiOp(row, one))
            s(scf.YieldOp(s(arith.SelectOp(fits, reduced, remainder)), s(arith.SelectOp(fits, nextrow, chosen))))
            loop = scf.ForOp(zero, b.ix(max(xs[0] - 1, 0)), one, [coords[0], zero], Region(search))
            b(loop)
            col = b(arith.AddiOp(b(arith.AddiOp(loop.results[0], loop.results[1])), one))
            xc, yc = [loop.results[1]], [col]
        body = Block(arg_types=[IndexType(), elem])
        k, acc = body.args
        e = Builder(body)
        left, right = e(tensor.ExtractOp(x, [*xc, k], elem)), e(tensor.ExtractOp(y, [*yc, k], elem))
        value = absval = e(mm.AbsFOp(e(arith.SubfOp(left, right))))
        if p == 0:
            nonzero = e(arith.CmpfOp(left, right, "une"))
            contribution = e(arith.SelectOp(nonzero, e(scalar(1, elem)), init))
            nan = e(arith.CmpfOp(absval, absval, "uno"))
            value = e(arith.SelectOp(nan, absval, contribution))
        elif p == 2:
            value = e(arith.MulfOp(value, value))
        elif p not in (1, float("inf")):
            value = e(mm.PowFOp(value, e(scalar(p, elem))))
        # CPU infinity norms ignore unordered differences, as their strict
        # comparison does; other p norms propagate NaNs through addition.
        if p == float("inf"):
            if cdist:
                comb = e(arith.SelectOp(e(arith.CmpfOp(value, acc, "ogt")), value, acc))
            else:
                comb = e(arith.MaximumfOp(value, acc))
        else:
            comb = e(arith.AddfOp(acc, value))
        e(scf.YieldOp(comb))
        value = b(scf.ForOp(zero, features, one, [init], Region(body)))
        if p == 2:
            value = b(mm.SqrtOp(value))
        elif p not in (0, 1, float("inf")):
            value = b(mm.PowFOp(value, b(scalar(1 / p, elem))))
        return value

    ops, result = generate(out, elem, build)
    return DecompResult(ops=ops, result=result, pattern_hint="distance")


def sort(operands, meta, node_name):
    """Stable adjacent compare-exchange with loop-sized IR and original indices."""
    from m2m.ir.decompositions import DecompResult, _fx_arg, _shape_of, _t_elem

    x = operands[0]
    shape = _shape_of(x)
    elem = _t_elem(x)
    rank = len(shape)
    topk = "topk" in meta["_aten_target"]
    stable = meta["_aten_target"] == "aten.sort.stable"
    dim = int(_fx_arg(meta, 2 if topk or stable else 1, -1)) % max(rank, 1)
    descending = bool(_fx_arg(meta, 3 if topk or stable else 2, topk))
    if not isinstance(elem, (AnyFloat, IntegerType)) or any(d < 0 for d in shape):
        raise ValueError("sort requires static real tensors")

    def indices(block, coords):
        b = Builder(block)
        return b(arith.IndexCastOp(coords[dim], i64)) if rank else b(scalar(0, i64))

    out = list(meta["val"][0].shape) if topk else None

    def finish(ops, results, sliced):
        if sliced:
            results = [Builder(ops=ops)(tensor.ExtractSliceOp.from_static_parameters(r, [0] * rank, out, [1] * rank))
                       for r in results]
        return DecompResult(ops=ops, result=results[0], results=results, pattern_hint="sort")

    ops, ix = generate(shape, i64, indices)
    if not rank or shape[dim] < 2 or math.prod(shape) == 0:
        return finish(ops, [x, ix], topk and rank and out != shape)
    top = Builder(ops=ops)
    zero, one, extent, total = top.ix(0), top.ix(1), top.ix(shape[dim]), top.ix(math.prod(shape) // shape[dim])
    vt, it = x.type, ix.type
    outer = Block(arg_types=[IndexType(), vt, it])
    outer_coords = coordinates(outer, outer.args[0], shape[:dim] + shape[dim + 1 :])
    passes = Block(arg_types=[IndexType(), vt, it])
    pb = Builder(passes)
    stop = pb(arith.SubiOp(pb(arith.SubiOp(extent, passes.args[0])), one))
    body = Block(arg_types=[IndexType(), vt, it])
    pos, v, i = body.args
    b = Builder(body)
    nextpos = b(arith.AddiOp(pos, one))
    lc = outer_coords[:dim] + [pos] + outer_coords[dim:]
    rc = outer_coords[:dim] + [nextpos] + outer_coords[dim:]
    left, right = b(tensor.ExtractOp(v, lc, elem)), b(tensor.ExtractOp(v, rc, elem))
    li, ri = b(tensor.ExtractOp(i, lc, i64)), b(tensor.ExtractOp(i, rc, i64))
    if isinstance(elem, AnyFloat):
        cmp = b(arith.CmpfOp(left, right, "olt" if descending else "ogt"))
        nan, number = (right, left) if descending else (left, right)
        unordered = b(arith.AndIOp(b(arith.CmpfOp(nan, nan, "uno")), b(arith.CmpfOp(number, number, "ord"))))
        swap = b(arith.OrIOp(cmp, unordered))
    else:
        unsigned = elem.bitwidth == 1
        swap = b(arith.CmpiOp(left, right, ("u" if unsigned else "s") + ("lt" if descending else "gt")))
    lv, rv = b(arith.SelectOp(swap, right, left)), b(arith.SelectOp(swap, left, right))
    liv, riv = b(arith.SelectOp(swap, ri, li)), b(arith.SelectOp(swap, li, ri))
    values = b(tensor.InsertOp(rv, b(tensor.InsertOp(lv, v, lc)), rc))
    order = b(tensor.InsertOp(riv, b(tensor.InsertOp(liv, i, lc)), rc))
    b(scf.YieldOp(values, order))
    inner = scf.ForOp(zero, stop, one, list(passes.args[1:]), Region(body))
    passes.add_ops([inner, scf.YieldOp(*inner.results)])
    loop = scf.ForOp(zero, extent, one, list(outer.args[1:]), Region(passes))
    outer.add_ops([loop, scf.YieldOp(*loop.results)])
    loop = scf.ForOp(zero, total, one, [x, ix], Region(outer))
    ops.append(loop)
    return finish(ops, list(loop.results), topk)



def avg_pool(operands, meta, node_name):
    """Position-dependent windows for regular and adaptive average pooling."""
    from m2m.ir.decompositions import DecompResult, _cast_tensor, _fx_arg, _shape_of, _t_elem

    x = operands[0]
    shape = _shape_of(x)
    elem = _t_elem(x)
    target = meta["_aten_target"]
    ndim = 3 if "3d" in target else 2 if "2d" in target else 1
    out = list(meta["val"].shape)
    prefix = len(shape) - ndim
    adaptive = "adaptive" in target
    override = None

    def vector(v, default):
        if v is None or v == []:
            v = default
        if isinstance(v, int):
            return [v] * ndim
        return list(v) * ndim if len(v) == 1 else list(v)

    if not adaptive:
        kernel = vector(_fx_arg(meta, 1, None), 1)
        stride = vector(_fx_arg(meta, 2, None), kernel)
        padding = vector(_fx_arg(meta, 3, None), 0)
        include = bool(_fx_arg(meta, 5, True))
        override = _fx_arg(meta, 6, None)
    # Native adaptive 3D pooling keeps its running sum in scalar_t,
    # unlike the 2D opmath kernel. Each addition is a storage observation.
    acc_elem = Float32Type() if isinstance(elem, (Float16Type, BFloat16Type)) and not (adaptive and ndim == 3) else elem
    casts, x = _cast_tensor(x, shape, acc_elem)

    def build(block, coords):
        b = Builder(block)
        z, o = b.ix(0), b.ix(1)
        init = b(scalar(0, acc_elem))
        starts = []
        ends = []
        divisor = o
        for axis in range(ndim):
            size = b.ix(shape[prefix + axis])
            coord = coords[prefix + axis]
            if adaptive:
                ins, outs = b.ix(shape[prefix + axis]), b.ix(out[prefix + axis])
                start = b(arith.DivUIOp(b(arith.MuliOp(coord, ins)), outs))
                scaled_end = b(arith.MuliOp(b(arith.AddiOp(coord, o)), ins))
                end = b(arith.DivUIOp(b(arith.AddiOp(scaled_end, b(arith.SubiOp(outs, o)))), outs))
            else:
                st, pad, k = b.ix(stride[axis]), b.ix(padding[axis]), b.ix(kernel[axis])
                start = b(arith.SubiOp(b(arith.MuliOp(coord, st)), pad))
                end = b(arith.MinSIOp(b(arith.AddiOp(start, k)), b(arith.AddiOp(size, pad))))
            low, high = b(arith.MaxSIOp(start, z)), b(arith.MinSIOp(end, size))
            starts.append(low)
            ends.append(high)
            span = (high, low) if adaptive or not include else (end, start)
            divisor = b(arith.MuliOp(divisor, b(arith.SubiOp(*span))))

        def nested(axis, nb, spatial, acc):
            if axis == ndim:
                value = nb(tensor.ExtractOp(x, [*coords[:prefix], *spatial], acc_elem))
                return nb((arith.AddfOp if isinstance(acc_elem, AnyFloat) else arith.AddiOp)(acc, value))
            inner = Block(arg_types=[IndexType(), acc_elem])
            val = nested(axis + 1, Builder(inner), [*spatial, inner.args[0]], inner.args[1])
            inner.add_op(scf.YieldOp(val))
            return nb(scf.ForOp(starts[axis], ends[axis], o, [acc], Region(inner)))

        summ = nested(0, b, [], init)
        if override is not None:
            denom = b(scalar(override, acc_elem))
        else:
            cast = b(arith.IndexCastOp(divisor, i64))
            denom = b(arith.SIToFPOp(cast, acc_elem) if isinstance(acc_elem, AnyFloat)
                      else arith.IndexCastOp(divisor, acc_elem))
        return b((arith.DivfOp if isinstance(acc_elem, AnyFloat) else arith.DivSIOp)(summ, denom))

    ops, result = generate(out, acc_elem, build)
    final, result = _cast_tensor(result, out, elem)
    return DecompResult(ops=[*casts, *ops, *final], result=result, pattern_hint="avg_pool")


def scatter(operands, meta, node_name):
    """Reduce indexed updates in source order, tracking touched cells and counts."""
    from m2m.ir.decompositions import DecompResult, _fx_arg, _shape_of, _t_elem

    x, indices = operands[:2]
    shape = _shape_of(x)
    ishape = _shape_of(indices)
    elem = _t_elem(x)
    target = meta["_aten_target"]
    rank = len(shape)
    dim = int(_fx_arg(meta, 1, 0)) % max(rank, 1)
    mode = (
        str(_fx_arg(meta, 4, "sum")) if "scatter_reduce" in target else "sum" if "scatter_add" in target else "assign"
    )
    include = bool(meta.get("_fx_kwargs", {}).get("include_self", _fx_arg(meta, 5, True)))
    source = operands[2] if len(operands) > 2 else None
    if mode not in ("sum", "prod", "mean", "amin", "amax", "assign"):
        raise ValueError("unknown scatter reduction")

    def build(block, coords):
        b = Builder(block)
        z, o, count = b.ix(0), b.ix(1), b.ix(math.prod(ishape))
        original = b(tensor.ExtractOp(x, coords, elem))
        body = Block(arg_types=[IndexType(), elem, IndexType()])
        pos, acc, n = body.args
        ic = coordinates(body, pos, ishape)
        e = Builder(body)
        dst = e(arith.IndexCastOp(e(tensor.ExtractOp(indices, ic, indices.type.element_type)), IndexType()))
        # scatter indices are nonnegative; unlike advanced indexing they do not wrap.
        from xdsl.dialects.cf import AssertOp

        size = e.ix(shape[dim] if rank else 1)
        valid = e(arith.AndIOp(e(arith.CmpiOp(dst, z, "sge")), e(arith.CmpiOp(dst, size, "slt"))))
        e(AssertOp(valid, "scatter index out of bounds"))
        matches = None
        for axis in range(rank):
            eq = e(arith.CmpiOp(dst if axis == dim else ic[axis], coords[axis], "eq"))
            matches = eq if matches is None else e(arith.AndIOp(matches, eq))
        if not rank:
            matches = e(scalar(1, IntegerType(1)))
        update = Block()
        u = Builder(update)
        val = u(tensor.ExtractOp(source, ic, elem) if source is not None else scalar(_fx_arg(meta, 3, 0), elem))
        value = val
        if mode != "assign":
            if isinstance(elem, IntegerType):
                cls = {
                    "sum": arith.OrIOp if elem.bitwidth == 1 else arith.AddiOp,
                    "mean": arith.AddiOp,
                    "prod": arith.AndIOp if elem.bitwidth == 1 else arith.MuliOp,
                    "amin": arith.MinUIOp if elem.bitwidth == 1 else arith.MinSIOp,
                    "amax": arith.MaxUIOp if elem.bitwidth == 1 else arith.MaxSIOp,
                }[mode]
            else:
                cls = {
                    "sum": arith.AddfOp,
                    "mean": arith.AddfOp,
                    "prod": arith.MulfOp,
                    "amin": arith.MinimumfOp,
                    "amax": arith.MaximumfOp,
                }[mode]
            value = u(cls(acc, value))
            if not include:
                value = u(arith.SelectOp(u(arith.CmpiOp(n, z, "eq")), val, value))
        u(scf.YieldOp(value, u(arith.AddiOp(n, o))))
        branch = scf.IfOp(matches, [elem, IndexType()], Region(update), Region(Block([scf.YieldOp(acc, n)])))
        body.add_ops([branch, scf.YieldOp(*branch.results)])
        loop = scf.ForOp(z, count, o, [original, z], Region(body))
        value = b(loop)
        if mode == "mean":
            n = loop.results[1]
            if include:
                n = b(arith.AddiOp(n, o))
            denom = b(arith.IndexCastOp(b(arith.MaxSIOp(n, o)), elem if isinstance(elem, IntegerType) else i64))
            if isinstance(elem, AnyFloat):
                denom = b(arith.SIToFPOp(denom, elem))
            # Integer scatter mean rounds down, including negative sums.
            value = b((arith.DivfOp if isinstance(elem, AnyFloat) else arith.FloorDivSIOp)(value, denom))
        return value

    ops, result = generate(shape, elem, build)
    return DecompResult(ops=ops, result=result, pattern_hint="scatter")


def mean(operands, meta, node_name):
    import torch
    from m2m.ir.decompositions import (
        DecompResult,
        _bin_build,
        _elementwise,
        _fx_arg,
        _keepdim_reshape,
        _opaque_decomp,
        _reduce,
        _reduction_dims,
        _reduction_input,
        _reduction_result,
        _reduction_cast,
        _element_type_from_meta,
        _shape_of,
        _splat_scalar,
        _tag_role,
    )

    x = operands[0]
    shape = _shape_of(x)
    if shape is None or any(d < 0 for d in shape):
        return _opaque_decomp("aten_mean_dim", operands, meta, "reduce", pattern_hint="reduce_mean")
    raw_dims = None if meta.get("_aten_target") == "aten.mean.default" else _fx_arg(meta, 1, None)
    dims = _reduction_dims(raw_dims, len(shape))
    source_node = _fx_arg(meta, 0)
    source = getattr(source_node, "meta", {}).get("val", source_node)
    if getattr(getattr(source, 'device', None), 'type', None) == 'cpu':
        elem = _element_type_from_meta(meta)
        acc = Float32Type() if isinstance(elem, (Float16Type, BFloat16Type)) else elem
        # CPU mean sums in the temporary output dtype, then converts once to
        # storage. Casting to Half/BF16 first loses tiny negative results' sign.
        prepared = _reduction_cast(x, shape, acc, meta, source_unsigned=source.dtype == torch.uint8)
        if prepared is None:
            return _opaque_decomp('aten_mean_dim', operands, meta, 'reduce', pattern_hint='reduce_mean')
        ops, x = prepared
        if source.dtype == torch.float16 and acc == Float32Type():
            from .cpu_pointwise import half_to_float_copy
            converted, x = half_to_float_copy(x, source)
            ops += converted
    else:
        prepared = _reduction_input(x, shape, meta)
        if prepared is None:
            return _opaque_decomp('aten_mean_dim', operands, meta, 'reduce', pattern_hint='reduce_mean')
        ops, x, acc, elem = prepared
    if (getattr(getattr(source, "device", None), "type", None) == "cpu"
            and (source.dtype in (torch.float16, torch.bfloat16)
                 or isinstance(elem, (Float16Type, BFloat16Type)))):
        from .cpu_reductions import sum_tensor
        rops, value = sum_tensor(x, dims, source)
        reduced = [shape[d] for d in range(len(shape)) if d not in dims]
    else:
        rops, value, reduced = _reduce(x, shape, dims, 0, arith.AddfOp, acc)
    _tag_role(rops, "sum")
    ops += rops
    n = math.prod(shape[d] for d in dims)
    const, denom = _splat_scalar(n, TensorType(acc, reduced))
    ops += const
    divide, value = _elementwise([value, denom], TensorType(acc, reduced), _bin_build(arith.DivfOp))
    _tag_role([*const, *divide], "scale")
    ops += divide
    out = list(meta["val"].shape)
    value = _keepdim_reshape(ops, value, reduced, out, acc)
    casts, value = _reduction_result(value, out, elem, meta)
    ops += casts
    if (getattr(getattr(source, 'device', None), 'type', None) == 'cpu'
            and isinstance(elem, BFloat16Type)):
        # CPU mean's BF16 scalar writeback differs from tensor conversion.
        nan_bits = 0xffff if out else 0x7fc0
        def canonical_nan(args, oe):
            b = Builder()
            bits = b(arith.ConstantOp(IntegerAttr(nan_bits if nan_bits < 32768 else nan_bits - 65536, IntegerType(16))))
            nan = b(arith.BitcastOp(bits, oe))
            missing = b(arith.CmpfOp(args[0], args[0], 'uno'))
            return b.ops, b(arith.SelectOp(missing, nan, args[0]))
        normalized, value = _elementwise([value], TensorType(elem, out), canonical_nan)
        ops += normalized
    return DecompResult(ops=ops, result=value, pattern_hint="reduce_mean")


def embedding_bag(operands, meta, node_name):
    """CPU forward bags, including all four auxiliary result tensors.

    Backward-only sparse/frequency flags do not affect forward values. CPU
    include_last_offset has distinct sum/mean/max tail rules, retained here.
    """
    from m2m.ir.decompositions import DecompResult, _cast_tensor, _fx_arg, _shape_of, _t_elem

    weight, indices, offsets = operands[:3]
    elem = _t_elem(weight)
    acc_elem = Float32Type() if isinstance(elem, (Float16Type, BFloat16Type)) else elem
    casts, weight = _cast_tensor(weight, _shape_of(weight), acc_elem)
    values = meta["val"]
    bags = values[0].shape[0]
    length = _shape_of(indices)[0]
    mode = int(_fx_arg(meta, 4, 0))
    last = bool(_fx_arg(meta, 7, False))
    padding = int(_fx_arg(meta, 8, -1))
    sample = operands[3] if len(operands) > 3 else None
    source = _fx_arg(meta, 0)
    source_value = getattr(source, "meta", {}).get("val", source)
    sample_source = _fx_arg(meta, 6, None)
    sample_value = getattr(sample_source, "meta", {}).get("val", sample_source)
    from .embedding_bag_cpu import fast_path_eligible
    fast = fast_path_eligible(source_value, sample_value, padding_idx=padding)
    ops = list(casts)
    top = Builder(ops=ops)
    z, o, n, limit, pad = top.ix(0), top.ix(1), top.ix(length), top.ix(bags), top.ix(padding)
    aux = max((indices.type.element_type, offsets.type.element_type), key=lambda t: t.bitwidth)

    def offset_at(b, bag):
        return b(arith.IndexCastOp(b(tensor.ExtractOp(offsets, [bag], offsets.type.element_type)), IndexType()))

    def row_at(b, pos):
        return b(arith.IndexCastOp(b(tensor.ExtractOp(indices, [pos], indices.type.element_type)), IndexType()))

    def bounds(block, bag, for_count=False):
        b = Builder(block)
        start = offset_at(b, bag)
        following = b(arith.AddiOp(bag, o))
        exists = b(arith.CmpiOp(following, limit, "slt"))
        yes, no = Block(), Block()
        yes.add_op(scf.YieldOp(offset_at(Builder(yes), following)))
        if last and mode != 2 and fast and not for_count:
            no.add_op(scf.YieldOp(offset_at(Builder(no), following)))
        else:
            no.add_op(scf.YieldOp(n))
        end = b(scf.IfOp(exists, [IndexType()], Region(yes), Region(no)))
        from xdsl.dialects.cf import AssertOp

        low, ordered, high = b(arith.CmpiOp(start, z, "sge")), b(arith.CmpiOp(start, end, "sle")), b(arith.CmpiOp(end, n, "sle"))
        b(AssertOp(b(arith.AndIOp(b(arith.AndIOp(low, ordered)), high)), "embedding_bag offsets out of bounds"))
        return start, end

    def count_active(block, bag):
        start, end = bounds(block, bag, True)
        body = Block(arg_types=[IndexType(), IndexType()])
        pos, count = body.args
        e = Builder(body)
        active = e(arith.CmpiOp(row_at(e, pos), pad, "ne"))
        e(scf.YieldOp(e(arith.AddiOp(count, e(arith.SelectOp(active, o, z))))))
        return Builder(block)(scf.ForOp(start, end, o, [z], Region(body)))

    def result(block, coords, return_index=False):
        start, end = bounds(block, coords[0])
        b = Builder(block)
        init, iz = b(scalar(0, acc_elem)), b(scalar(0, aux))
        body = Block(arg_types=[IndexType(), acc_elem, aux, IndexType()])
        pos, current, argmax, count = body.args
        e = Builder(body)
        row = row_at(e, pos)
        active = e(arith.CmpiOp(row, pad, "ne"))
        yes = Block()
        y = Builder(yes)
        from xdsl.dialects.cf import AssertOp

        rows = y.ix(weight.type.get_shape()[0])
        valid = y(arith.AndIOp(y(arith.CmpiOp(row, z, "sge")), y(arith.CmpiOp(row, rows, "slt"))))
        y(AssertOp(valid, "embedding_bag index out of bounds"))
        contribution = y(tensor.ExtractOp(weight, [row, coords[1]], acc_elem))
        if sample is not None:
            scale = y(tensor.ExtractOp(sample, [pos], elem))
            if acc_elem != elem:
                scale = y(arith.ExtFOp(scale, acc_elem))
            contribution = y(arith.MulfOp(contribution, scale))
        if mode == 2:
            first, greater = y(arith.CmpiOp(count, z, "eq")), y(arith.CmpfOp(contribution, current, "ogt"))
            take = y(arith.OrIOp(first, greater))
            chosen = y(arith.SelectOp(take, contribution, current))
            newarg = y(arith.SelectOp(take, y(arith.IndexCastOp(row, aux)), argmax))
        else:
            chosen = y(arith.AddfOp(current, contribution))
            newarg = argmax
        y(scf.YieldOp(chosen, newarg, y(arith.AddiOp(count, o))))
        branch = scf.IfOp(active, [acc_elem, aux, IndexType()], Region(yes),
                          Region(Block([scf.YieldOp(current, argmax, count)])))
        body.add_ops([branch, scf.YieldOp(*branch.results)])
        loop = scf.ForOp(start, end, o, [init, iz, z], Region(body))
        value = b(loop)
        if return_index:
            return loop.results[1]
        if mode == 1:
            count = b(arith.IndexCastOp(b(arith.MaxSIOp(count_active(block, coords[0]), o)), i64))
            value = b(arith.DivfOp(value, b(arith.SIToFPOp(count, acc_elem))))
        return value

    fast_bf16 = fast and mode == 0 and isinstance(elem, BFloat16Type)
    def stored_result(block, coords):
        value = result(block, coords)
        if fast_bf16:
            from .embedding_bag_cpu import fbgemm_bfloat16_store
            value = fbgemm_bfloat16_store(value, block)
        return value

    extra, out = generate(list(values[0].shape), elem if fast_bf16 else acc_elem, stored_result)
    ops += extra
    cast, out = _cast_tensor(out, list(values[0].shape), elem)
    ops += cast
    results = [out]
    for ordinal, value in enumerate(values[1:], 1):

        def build(block, coords):
            b = Builder(block)
            if ordinal == 1:
                body = Block(arg_types=[IndexType(), aux])
                bag, current = body.args
                e = Builder(body)
                belongs = e(arith.CmpiOp(offset_at(e, bag), coords[0], "sle"))
                e(scf.YieldOp(e(arith.SelectOp(belongs, e(arith.IndexCastOp(bag, aux)), current))))
                init = b(scalar(0, aux))
                return b(scf.ForOp(z, limit, o, [init], Region(body)))
            if mode == 0:
                return b(scalar(0, aux))
            if ordinal == 3 and mode == 2:
                return result(block, coords, True)
            return b(arith.IndexCastOp(count_active(block, coords[0]), aux))

        extra, ssa = generate(list(value.shape), aux, build)
        ops += extra
        results.append(ssa)
    return DecompResult(ops=ops, result=results[0], results=results, pattern_hint="embedding_bag")


def avg_pool_backward(operands, meta, node_name):
    """Accumulate each overlapping output gradient with its actual pool divisor."""
    from m2m.ir.decompositions import DecompResult, _fx_arg, _shape_of, _t_elem

    grad, x = operands[:2]
    shape = _shape_of(x)
    gshape = _shape_of(grad)
    elem = _t_elem(grad)
    adaptive = "adaptive" in meta["_aten_target"]
    ndim = 2
    prefix = len(shape) - ndim

    def pair(raw, default):
        if raw is None or raw == []:
            raw = default
        if isinstance(raw, int):
            return [raw] * 2
        return list(raw) * 2 if len(raw) == 1 else list(raw)

    if not adaptive:
        kernel = pair(_fx_arg(meta, 2, None), [1, 1])
        stride = pair(_fx_arg(meta, 3, None), kernel)
        padding = pair(_fx_arg(meta, 4, None), [0, 0])
        include = bool(_fx_arg(meta, 6, True))
        override = _fx_arg(meta, 7, None)

    def build(block, coords):
        b = Builder(block)
        z, o, n = b.ix(0), b.ix(1), b.ix(math.prod(gshape[-2:]))
        init = b(scalar(0, elem))
        body = Block(arg_types=[IndexType(), elem])
        gc = coordinates(body, body.args[0], gshape[-2:])
        e = Builder(body)
        valid = None
        divisor = o
        for axis in range(2):
            size = e.ix(shape[prefix + axis])
            if adaptive:
                outs = e.ix(gshape[prefix + axis])
                start = e(arith.DivUIOp(e(arith.MuliOp(gc[axis], size)), outs))
                scaled_end = e(arith.MuliOp(e(arith.AddiOp(gc[axis], o)), size))
                end = e(arith.DivUIOp(e(arith.AddiOp(scaled_end, e(arith.SubiOp(outs, o)))), outs))
            else:
                st, pad, k = e.ix(stride[axis]), e.ix(padding[axis]), e.ix(kernel[axis])
                start = e(arith.SubiOp(e(arith.MuliOp(gc[axis], st)), pad))
                end = e(arith.MinSIOp(e(arith.AddiOp(start, k)), e(arith.AddiOp(size, pad))))
            low, high = e(arith.MaxSIOp(start, z)), e(arith.MinSIOp(end, size))
            coord = coords[prefix + axis]
            both = e(arith.AndIOp(e(arith.CmpiOp(coord, low, "sge")), e(arith.CmpiOp(coord, high, "slt"))))
            valid = both if valid is None else e(arith.AndIOp(valid, both))
            span = (end, start) if not adaptive and include else (high, low)
            divisor = e(arith.MuliOp(divisor, e(arith.SubiOp(*span))))
        yes = Block()
        y = Builder(yes)
        value = y(tensor.ExtractOp(grad, [*coords[:prefix], *gc], elem))
        if not adaptive and override is not None:
            denom = y(scalar(override, elem))
        else:
            denom = y(arith.IndexCastOp(divisor, elem if isinstance(elem, IntegerType) else i64))
            if isinstance(elem, AnyFloat):
                denom = y(arith.SIToFPOp(denom, elem))
        div = y((arith.DivfOp if isinstance(elem, AnyFloat) else arith.DivSIOp)(value, denom))
        y(scf.YieldOp(y((arith.AddfOp if isinstance(elem, AnyFloat) else arith.AddiOp)(body.args[1], div))))
        branch = scf.IfOp(valid, [elem], Region(yes), Region(Block([scf.YieldOp(body.args[1])])))
        e(scf.YieldOp(e(branch)))
        return b(scf.ForOp(z, n, o, [init], Region(body)))

    ops, result = generate(shape, elem, build)
    return DecompResult(ops=ops, result=result, pattern_hint="avg_pool_backward")


def batch_norm_functional(operands, meta, node_name):
    from m2m.ir.decompositions import _fx_arg, decompose_batch_norm_inference, decompose_batch_norm_training

    if bool(_fx_arg(meta, 5, False)):
        return decompose_batch_norm_training(operands, meta, node_name)
    rewritten = dict(meta)
    args = meta["_fx_args"]
    rewritten["_fx_args"] = (*args[:5], *args[6:])
    rewritten["_aten_target"] = "aten._native_batch_norm_legit_no_training.default"
    rewritten["val"] = meta["val"][:3]
    result = decompose_batch_norm_inference(operands, rewritten, node_name)
    slots = meta.get("_fx_ssa_args")
    if slots is not None:
        stats = [slots[3], slots[4]]
    else:
        stats = operands[-2:]
    result.results += stats
    return result
