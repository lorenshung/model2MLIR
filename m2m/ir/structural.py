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


def index(n):
    return arith.ConstantOp(IntegerAttr(n, IndexType()), IndexType())


def scalar(n, elem):
    return arith.ConstantOp(
        IntegerAttr(int(bool(n)) if elem.bitwidth == 1 else int(n), elem)
        if isinstance(elem, IntegerType)
        else FloatAttr(float(n), elem),
        elem,
    )


def coordinates(block, linear, shape):
    result = [None] * len(shape)
    for axis in reversed(range(len(shape))):
        size = index(max(shape[axis], 1))
        rem = arith.RemUIOp(linear, size.result)
        block.add_ops([size, rem])
        result[axis] = rem.result
        if axis:
            div = arith.DivUIOp(linear, size.result)
            block.add_op(div)
            linear = div.result
    return result


def generate(shape, elem, build):
    from m2m.ir.decompositions import _make_empty

    rt = TensorType(elem, shape)
    empty = _make_empty(rt)
    block = Block(arg_types=[elem])
    coords = []
    for axis in range(len(shape)):
        ix = IndexOp(axis)
        block.add_op(ix)
        coords.append(ix.result)
    result = build(block, coords)
    block.add_op(YieldOp(result))
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
    a, b = (int(_fx_arg(meta, 2, 0)) % len(shape), int(_fx_arg(meta, 3, 1)) % len(shape))

    def build(block, coords):
        src = []
        remaining = iter(coords[:-1])
        for axis in range(len(shape)):
            if axis in (a, b):
                shift = index(max(-offset, 0) if axis == a else max(offset, 0))
                pos = arith.AddiOp(coords[-1], shift.result)
                block.add_ops([shift, pos])
                src.append(pos.result)
            else:
                src.append(next(remaining))
        picked = tensor.ExtractOp(x, src, _t_elem(x))
        block.add_op(picked)
        return picked.result

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
        zero, one, features = index(0), index(1), index(xs[-1])
        init = scalar(0, elem)
        block.add_ops([zero, one, features, init])
        if cdist:
            batch = coords[:-2]

            def batch_coords(shape):
                result = []
                for size, c in zip(shape[:-2], batch[len(batch) - len(shape[:-2]) :]):
                    if size == 1:
                        result.append(zero.result)
                    else:
                        result.append(c)
                return result

            xc, yc = batch_coords(xs) + [coords[-2]], batch_coords(ys) + [coords[-1]]
        else:
            # Locate the triangular row by subtracting successive row lengths.
            search = Block(arg_types=[IndexType(), IndexType(), IndexType()])
            row, remainder, chosen = search.args
            n = index(xs[0] - 1)
            width = arith.SubiOp(n.result, row)
            large = arith.CmpiOp(remainder, width.result, "sge")
            active = arith.CmpiOp(row, chosen, "eq")
            fits = arith.AndIOp(large.result, active.result)
            reduced = arith.SubiOp(remainder, width.result)
            nextrow = arith.AddiOp(row, one.result)
            rem = arith.SelectOp(fits.result, reduced.result, remainder)
            pick = arith.SelectOp(fits.result, nextrow.result, chosen)
            search.add_ops(
                [n, width, large, active, fits, reduced, nextrow, rem, pick, scf.YieldOp(rem.result, pick.result)]
            )
            limit = index(max(xs[0] - 1, 0))
            loop = scf.ForOp(zero.result, limit.result, one.result, [coords[0], zero.result], Region(search))
            col0 = arith.AddiOp(loop.results[0], loop.results[1])
            col = arith.AddiOp(col0.result, one.result)
            block.add_ops([limit, loop, col0, col])
            xc, yc = [loop.results[1]], [col.result]
        body = Block(arg_types=[IndexType(), elem])
        k, acc = body.args
        a, b = tensor.ExtractOp(x, [*xc, k], elem), tensor.ExtractOp(y, [*yc, k], elem)
        sub = arith.SubfOp(a.result, b.result)
        absval = mm.AbsFOp(sub.result)
        body.add_ops([a, b, sub, absval])
        value = absval.result
        if p == 0:
            nz = arith.CmpfOp(a.result, b.result, "une")
            unit = scalar(1, elem)
            contribution = arith.SelectOp(nz.result, unit.result, init.result)
            nan = arith.CmpfOp(absval.result, absval.result, "uno")
            preserved = arith.SelectOp(nan.result, absval.result, contribution.result)
            body.add_ops([nz, unit, contribution, nan, preserved])
            value = preserved.result
        elif p == 2:
            sq = arith.MulfOp(value, value)
            body.add_op(sq)
            value = sq.result
        elif p not in (1, float("inf")):
            exponent = scalar(p, elem)
            power = mm.PowFOp(value, exponent.result)
            body.add_ops([exponent, power])
            value = power.result
        # CPU infinity norms ignore unordered differences, as their strict
        # comparison does; other p norms propagate NaNs through addition.
        if p == float("inf"):
            if cdist:
                bigger = arith.CmpfOp(value, acc, "ogt")
                comb = arith.SelectOp(bigger.result, value, acc)
                body.add_ops([bigger, comb])
            else:
                comb = arith.MaximumfOp(value, acc)
                body.add_op(comb)
        else:
            comb = arith.AddfOp(acc, value)
            body.add_op(comb)
        body.add_op(scf.YieldOp(comb.result))
        loop = scf.ForOp(zero.result, features.result, one.result, [init.result], Region(body))
        block.add_op(loop)
        value = loop.results[0]
        if p == 2:
            root = mm.SqrtOp(value)
            block.add_op(root)
            value = root.result
        elif p not in (0, 1, float("inf")):
            exponent = scalar(1 / p, elem)
            root = mm.PowFOp(value, exponent.result)
            block.add_ops([exponent, root])
            value = root.result
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
        if rank:
            cast = arith.IndexCastOp(coords[dim], i64)
            block.add_op(cast)
            return cast.result
        zero = scalar(0, i64)
        block.add_op(zero)
        return zero.result

    ops, ix = generate(shape, i64, indices)
    if not rank or shape[dim] < 2 or math.prod(shape) == 0:
        results = [x, ix]
        if topk and rank and list(meta["val"][0].shape) != shape:
            out = list(meta["val"][0].shape)
            for j in range(2):
                sliced = tensor.ExtractSliceOp.from_static_parameters(results[j], [0] * rank, out, [1] * rank)
                ops.append(sliced)
                results[j] = sliced.result
        return DecompResult(ops=ops, result=results[0], results=results, pattern_hint="sort")
    zero, one, extent, total = index(0), index(1), index(shape[dim]), index(math.prod(shape) // shape[dim])
    ops += [zero, one, extent, total]
    vt, it = x.type, ix.type
    outer = Block(arg_types=[IndexType(), vt, it])
    outer_coords = coordinates(outer, outer.args[0], shape[:dim] + shape[dim + 1 :])
    passes = Block(arg_types=[IndexType(), vt, it])
    limit = arith.SubiOp(extent.result, passes.args[0])
    stop = arith.SubiOp(limit.result, one.result)
    passes.add_ops([limit, stop])
    body = Block(arg_types=[IndexType(), vt, it])
    pos, v, i = body.args
    nextpos = arith.AddiOp(pos, one.result)
    body.add_op(nextpos)
    lc = outer_coords[:dim] + [pos] + outer_coords[dim:]
    rc = outer_coords[:dim] + [nextpos.result] + outer_coords[dim:]
    left, right = tensor.ExtractOp(v, lc, elem), tensor.ExtractOp(v, rc, elem)
    li, ri = tensor.ExtractOp(i, lc, i64), tensor.ExtractOp(i, rc, i64)
    body.add_ops([left, right, li, ri])
    if isinstance(elem, AnyFloat):
        cmp = arith.CmpfOp(left.result, right.result, "olt" if descending else "ogt")
        nan = right if descending else left
        number = left if descending else right
        isnan = arith.CmpfOp(nan.result, nan.result, "uno")
        isnum = arith.CmpfOp(number.result, number.result, "ord")
        ns = arith.AndIOp(isnan.result, isnum.result)
        swap = arith.OrIOp(cmp.result, ns.result)
        body.add_ops([cmp, isnan, isnum, ns, swap])
    else:
        unsigned = elem.bitwidth == 1
        swap = arith.CmpiOp(left.result, right.result, ("u" if unsigned else "s") + ("lt" if descending else "gt"))
        body.add_op(swap)
    lv, rv = (
        arith.SelectOp(swap.result, right.result, left.result),
        arith.SelectOp(swap.result, left.result, right.result),
    )
    liv, riv = arith.SelectOp(swap.result, ri.result, li.result), arith.SelectOp(swap.result, li.result, ri.result)
    vl = tensor.InsertOp(lv.result, v, lc)
    vr = tensor.InsertOp(rv.result, vl.result, rc)
    il = tensor.InsertOp(liv.result, i, lc)
    ir = tensor.InsertOp(riv.result, il.result, rc)
    body.add_ops([lv, rv, liv, riv, vl, vr, il, ir, scf.YieldOp(vr.result, ir.result)])
    inner = scf.ForOp(zero.result, stop.result, one.result, list(passes.args[1:]), Region(body))
    passes.add_ops([inner, scf.YieldOp(*inner.results)])
    loop = scf.ForOp(zero.result, extent.result, one.result, list(outer.args[1:]), Region(passes))
    outer.add_ops([loop, scf.YieldOp(*loop.results)])
    loop = scf.ForOp(zero.result, total.result, one.result, [x, ix], Region(outer))
    ops.append(loop)
    results = list(loop.results)
    if topk:
        out = list(meta["val"][0].shape)
        for j in range(2):
            sliced = tensor.ExtractSliceOp.from_static_parameters(results[j], [0] * rank, out, [1] * rank)
            ops.append(sliced)
            results[j] = sliced.result
    return DecompResult(ops=ops, result=results[0], results=results, pattern_hint="sort")


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
        z, o = index(0), index(1)
        init = scalar(0, acc_elem)
        block.add_ops([z, o, init])
        starts = []
        ends = []
        divisor = o.result
        for axis in range(ndim):
            size = index(shape[prefix + axis])
            block.add_op(size)
            if adaptive:
                ins, outs = index(shape[prefix + axis]), index(out[prefix + axis])
                scaled = arith.MuliOp(coords[prefix + axis], ins.result)
                start = arith.DivUIOp(scaled.result, outs.result)
                nxt = arith.AddiOp(coords[prefix + axis], o.result)
                scaled_end = arith.MuliOp(nxt.result, ins.result)
                lessone = arith.SubiOp(outs.result, o.result)
                roundup = arith.AddiOp(scaled_end.result, lessone.result)
                end = arith.DivUIOp(roundup.result, outs.result)
                block.add_ops([ins, outs, scaled, start, nxt, scaled_end, lessone, roundup, end])
                ds, de = start.result, end.result
            else:
                st, pad, k = index(stride[axis]), index(padding[axis]), index(kernel[axis])
                mul = arith.MuliOp(coords[prefix + axis], st.result)
                start = arith.SubiOp(mul.result, pad.result)
                end0 = arith.AddiOp(start.result, k.result)
                padded_end = arith.AddiOp(size.result, pad.result)
                end = arith.MinSIOp(end0.result, padded_end.result)
                block.add_ops([st, pad, k, mul, start, end0, padded_end, end])
                ds, de = start.result, end.result
            low = arith.MaxSIOp(start.result, z.result)
            high = arith.MinSIOp(end.result, size.result)
            block.add_ops([low, high])
            starts.append(low.result)
            ends.append(high.result)
            if adaptive or not include:
                ds, de = low.result, high.result
            width = arith.SubiOp(de, ds)
            product = arith.MuliOp(divisor, width.result)
            block.add_ops([width, product])
            divisor = product.result

        def nested(axis, body, spatial, acc):
            if axis == ndim:
                value = tensor.ExtractOp(x, [*coords[:prefix], *spatial], acc_elem)
                add = (arith.AddfOp if isinstance(acc_elem, AnyFloat) else arith.AddiOp)(acc, value.result)
                body.add_ops([value, add])
                return add.result
            inner = Block(arg_types=[IndexType(), acc_elem])
            val = nested(axis + 1, inner, [*spatial, inner.args[0]], inner.args[1])
            inner.add_op(scf.YieldOp(val))
            loop = scf.ForOp(starts[axis], ends[axis], o.result, [acc], Region(inner))
            body.add_op(loop)
            return loop.results[0]

        summ = nested(0, block, [], init.result)
        if override is not None:
            dc = scalar(override, acc_elem)
            block.add_op(dc)
            denom = dc.result
        else:
            cast = arith.IndexCastOp(divisor, i64)
            block.add_op(cast)
            if isinstance(acc_elem, AnyFloat):
                dc = arith.SIToFPOp(cast.result, acc_elem)
                block.add_op(dc)
                denom = dc.result
            else:
                dc = arith.IndexCastOp(divisor, acc_elem)
                block.add_op(dc)
                denom = dc.result
        divided = (arith.DivfOp if isinstance(acc_elem, AnyFloat) else arith.DivSIOp)(summ, denom)
        block.add_op(divided)
        return divided.result

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
        z, o, count = index(0), index(1), index(math.prod(ishape))
        original = tensor.ExtractOp(x, coords, elem)
        block.add_ops([z, o, count, original])
        body = Block(arg_types=[IndexType(), elem, IndexType()])
        pos, acc, n = body.args
        ic = coordinates(body, pos, ishape)
        picked = tensor.ExtractOp(indices, ic, indices.type.element_type)
        dst = arith.IndexCastOp(picked.result, IndexType())
        body.add_ops([picked, dst])
        # scatter indices are nonnegative; unlike advanced indexing they do not wrap.
        from xdsl.dialects.cf import AssertOp

        size = index(shape[dim] if rank else 1)
        low = arith.CmpiOp(dst.result, z.result, "sge")
        high = arith.CmpiOp(dst.result, size.result, "slt")
        valid = arith.AndIOp(low.result, high.result)
        check = AssertOp(valid.result, "scatter index out of bounds")
        body.add_ops([size, low, high, valid, check])
        matches = None
        for axis in range(rank):
            eq = arith.CmpiOp(dst.result if axis == dim else ic[axis], coords[axis], "eq")
            body.add_op(eq)
            if matches is None:
                matches = eq.result
            else:
                both = arith.AndIOp(matches, eq.result)
                body.add_op(both)
                matches = both.result
        if not rank:
            yes = scalar(1, IntegerType(1))
            body.add_op(yes)
            matches = yes.result
        update = Block()
        if source is not None:
            val = tensor.ExtractOp(source, ic, elem)
            update.add_op(val)
        else:
            val = scalar(_fx_arg(meta, 3, 0), elem)
            update.add_op(val)
        value = val.result
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
            combined = cls(acc, value)
            update.add_op(combined)
            value = combined.result
            if not include:
                first = arith.CmpiOp(n, z.result, "eq")
                chosen = arith.SelectOp(first.result, val.result, value)
                update.add_ops([first, chosen])
                value = chosen.result
        nn = arith.AddiOp(n, o.result)
        update.add_ops([nn, scf.YieldOp(value, nn.result)])
        no = Block()
        no.add_op(scf.YieldOp(acc, n))
        branch = scf.IfOp(matches, [elem, IndexType()], Region(update), Region(no))
        body.add_ops([branch, scf.YieldOp(*branch.results)])
        loop = scf.ForOp(z.result, count.result, o.result, [original.result, z.result], Region(body))
        block.add_op(loop)
        value = loop.results[0]
        if mode == "mean":
            n = loop.results[1]
            if include:
                plus = arith.AddiOp(n, o.result)
                block.add_op(plus)
                n = plus.result
            safe = arith.MaxSIOp(n, o.result)
            block.add_op(safe)
            cast = arith.IndexCastOp(safe.result, elem if isinstance(elem, IntegerType) else i64)
            block.add_op(cast)
            denom = cast.result
            if isinstance(elem, AnyFloat):
                fp = arith.SIToFPOp(denom, elem)
                block.add_op(fp)
                denom = fp.result
            # Integer scatter mean rounds down, including negative sums.
            div = (arith.DivfOp if isinstance(elem, AnyFloat) else arith.FloorDivSIOp)(value, denom)
            block.add_op(div)
            value = div.result
        return value

    ops, result = generate(shape, elem, build)
    return DecompResult(ops=ops, result=result, pattern_hint="scatter")


def mean(operands, meta, node_name):
    import torch
    from m2m.ir.decompositions import (
        DecompResult,
        _bin_build,
        _cast_tensor,
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
            bits = arith.ConstantOp(IntegerAttr(nan_bits if nan_bits < 32768 else nan_bits - 65536, IntegerType(16)))
            nan = arith.BitcastOp(bits.result, oe)
            missing = arith.CmpfOp(args[0], args[0], 'uno')
            chosen = arith.SelectOp(missing.result, nan.result, args[0])
            return [bits, nan, missing, chosen], chosen.result
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
    z, o, n, limit, pad = index(0), index(1), index(length), index(bags), index(padding)
    ops += [z, o, n, limit, pad]
    aux = max((indices.type.element_type, offsets.type.element_type), key=lambda t: t.bitwidth)

    def bounds(block, bag, for_count=False):
        start = tensor.ExtractOp(offsets, [bag], offsets.type.element_type)
        si = arith.IndexCastOp(start.result, IndexType())
        nb = arith.AddiOp(bag, o.result)
        exists = arith.CmpiOp(nb.result, limit.result, "slt")
        block.add_ops([start, si, nb, exists])
        yes = Block()
        nextoffset = tensor.ExtractOp(offsets, [nb.result], offsets.type.element_type)
        ei = arith.IndexCastOp(nextoffset.result, IndexType())
        yes.add_ops([nextoffset, ei, scf.YieldOp(ei.result)])
        no = Block()
        if last and mode != 2 and fast and not for_count:
            nextoffset = tensor.ExtractOp(offsets, [nb.result], offsets.type.element_type)
            ei = arith.IndexCastOp(nextoffset.result, IndexType())
            no.add_ops([nextoffset, ei, scf.YieldOp(ei.result)])
        else:
            no.add_op(scf.YieldOp(n.result))
        end = scf.IfOp(exists.result, [IndexType()], Region(yes), Region(no))
        block.add_op(end)
        from xdsl.dialects.cf import AssertOp

        low = arith.CmpiOp(si.result, z.result, "sge")
        ordered = arith.CmpiOp(si.result, end.results[0], "sle")
        high = arith.CmpiOp(end.results[0], n.result, "sle")
        both = arith.AndIOp(low.result, ordered.result)
        valid = arith.AndIOp(both.result, high.result)
        check = AssertOp(valid.result, "embedding_bag offsets out of bounds")
        block.add_ops([low, ordered, high, both, valid, check])
        return si.result, end.results[0]

    def count_active(block, bag):
        start, end = bounds(block, bag, True)
        body = Block(arg_types=[IndexType(), IndexType()])
        pos, count = body.args
        picked = tensor.ExtractOp(indices, [pos], indices.type.element_type)
        row = arith.IndexCastOp(picked.result, IndexType())
        active = arith.CmpiOp(row.result, pad.result, "ne")
        inc = arith.SelectOp(active.result, o.result, z.result)
        plus = arith.AddiOp(count, inc.result)
        body.add_ops([picked, row, active, inc, plus, scf.YieldOp(plus.result)])
        loop = scf.ForOp(start, end, o.result, [z.result], Region(body))
        block.add_op(loop)
        return loop.results[0]

    def result(block, coords, return_index=False):
        start, end = bounds(block, coords[0])
        init = scalar(0, acc_elem)
        iz = scalar(0, aux)
        block.add_ops([init, iz])
        body = Block(arg_types=[IndexType(), acc_elem, aux, IndexType()])
        pos, current, argmax, count = body.args
        picked = tensor.ExtractOp(indices, [pos], indices.type.element_type)
        row = arith.IndexCastOp(picked.result, IndexType())
        active = arith.CmpiOp(row.result, pad.result, "ne")
        body.add_ops([picked, row, active])
        yes = Block()
        from xdsl.dialects.cf import AssertOp

        rows = index(weight.type.get_shape()[0])
        low = arith.CmpiOp(row.result, z.result, "sge")
        high = arith.CmpiOp(row.result, rows.result, "slt")
        valid = arith.AndIOp(low.result, high.result)
        check = AssertOp(valid.result, "embedding_bag index out of bounds")
        value = tensor.ExtractOp(weight, [row.result, coords[1]], acc_elem)
        yes.add_ops([rows, low, high, valid, check, value])
        contribution = value.result
        if sample is not None:
            scale = tensor.ExtractOp(sample, [pos], elem)
            yes.add_op(scale)
            scaleval = scale.result
            if acc_elem != elem:
                ext = arith.ExtFOp(scaleval, acc_elem)
                yes.add_op(ext)
                scaleval = ext.result
            mul = arith.MulfOp(contribution, scaleval)
            yes.add_op(mul)
            contribution = mul.result
        if mode == 2:
            first = arith.CmpiOp(count, z.result, "eq")
            greater = arith.CmpfOp(contribution, current, "ogt")
            take = arith.OrIOp(first.result, greater.result)
            chosen = arith.SelectOp(take.result, contribution, current)
            ri = arith.IndexCastOp(row.result, aux)
            arg = arith.SelectOp(take.result, ri.result, argmax)
            yes.add_ops([first, greater, take, chosen, ri, arg])
            newarg = arg.result
        else:
            chosen = arith.AddfOp(current, contribution)
            yes.add_op(chosen)
            newarg = argmax
        plus = arith.AddiOp(count, o.result)
        yes.add_ops([plus, scf.YieldOp(chosen.result, newarg, plus.result)])
        no = Block()
        no.add_op(scf.YieldOp(current, argmax, count))
        branch = scf.IfOp(active.result, [acc_elem, aux, IndexType()], Region(yes), Region(no))
        body.add_ops([branch, scf.YieldOp(*branch.results)])
        loop = scf.ForOp(start, end, o.result, [init.result, iz.result, z.result], Region(body))
        block.add_op(loop)
        if return_index:
            return loop.results[1]
        value = loop.results[0]
        if mode == 1:
            count = count_active(block, coords[0])
            safe = arith.MaxSIOp(count, o.result)
            ci = arith.IndexCastOp(safe.result, i64)
            fp = arith.SIToFPOp(ci.result, acc_elem)
            div = arith.DivfOp(value, fp.result)
            block.add_ops([safe, ci, fp, div])
            value = div.result
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
            if ordinal == 1:
                body = Block(arg_types=[IndexType(), aux])
                bag, current = body.args
                offset = tensor.ExtractOp(offsets, [bag], offsets.type.element_type)
                oi = arith.IndexCastOp(offset.result, IndexType())
                belongs = arith.CmpiOp(oi.result, coords[0], "sle")
                bi = arith.IndexCastOp(bag, aux)
                chosen = arith.SelectOp(belongs.result, bi.result, current)
                body.add_ops([offset, oi, belongs, bi, chosen, scf.YieldOp(chosen.result)])
                init = scalar(0, aux)
                block.add_op(init)
                loop = scf.ForOp(z.result, limit.result, o.result, [init.result], Region(body))
                block.add_op(loop)
                return loop.results[0]
            if mode == 0:
                init = scalar(0, aux)
                block.add_op(init)
                return init.result
            if ordinal == 3 and mode == 2:
                return result(block, coords, True)
            count = count_active(block, coords[0])
            cast = arith.IndexCastOp(count, aux)
            block.add_op(cast)
            return cast.result

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
        z, o, n = index(0), index(1), index(math.prod(gshape[-2:]))
        init = scalar(0, elem)
        block.add_ops([z, o, n, init])
        body = Block(arg_types=[IndexType(), elem])
        gc = coordinates(body, body.args[0], gshape[-2:])
        valid = None
        divisor = o.result
        for axis in range(2):
            size = index(shape[prefix + axis])
            body.add_op(size)
            if adaptive:
                outs = index(gshape[prefix + axis])
                scaled = arith.MuliOp(gc[axis], size.result)
                start = arith.DivUIOp(scaled.result, outs.result)
                nxt = arith.AddiOp(gc[axis], o.result)
                mul = arith.MuliOp(nxt.result, size.result)
                less = arith.SubiOp(outs.result, o.result)
                rounded = arith.AddiOp(mul.result, less.result)
                end = arith.DivUIOp(rounded.result, outs.result)
                body.add_ops([outs, scaled, start, nxt, mul, less, rounded, end])
            else:
                st, pad, k = index(stride[axis]), index(padding[axis]), index(kernel[axis])
                mul = arith.MuliOp(gc[axis], st.result)
                start = arith.SubiOp(mul.result, pad.result)
                end0 = arith.AddiOp(start.result, k.result)
                bound = arith.AddiOp(size.result, pad.result)
                end = arith.MinSIOp(end0.result, bound.result)
                body.add_ops([st, pad, k, mul, start, end0, bound, end])
            low = arith.MaxSIOp(start.result, z.result)
            high = arith.MinSIOp(end.result, size.result)
            lower = arith.CmpiOp(coords[prefix + axis], low.result, "sge")
            upper = arith.CmpiOp(coords[prefix + axis], high.result, "slt")
            both = arith.AndIOp(lower.result, upper.result)
            body.add_ops([low, high, lower, upper, both])
            if valid is None:
                valid = both.result
            else:
                allvalid = arith.AndIOp(valid, both.result)
                body.add_op(allvalid)
                valid = allvalid.result
            width = arith.SubiOp(
                end.result if not adaptive and include else high.result,
                start.result if not adaptive and include else low.result,
            )
            product = arith.MuliOp(divisor, width.result)
            body.add_ops([width, product])
            divisor = product.result
        yes = Block()
        value = tensor.ExtractOp(grad, [*coords[:prefix], *gc], elem)
        yes.add_op(value)
        if not adaptive and override is not None:
            dc = scalar(override, elem)
            yes.add_op(dc)
            denom = dc.result
        else:
            ci = arith.IndexCastOp(divisor, elem if isinstance(elem, IntegerType) else i64)
            yes.add_op(ci)
            denom = ci.result
            if isinstance(elem, AnyFloat):
                fp = arith.SIToFPOp(denom, elem)
                yes.add_op(fp)
                denom = fp.result
        div = (arith.DivfOp if isinstance(elem, AnyFloat) else arith.DivSIOp)(value.result, denom)
        add = (arith.AddfOp if isinstance(elem, AnyFloat) else arith.AddiOp)(body.args[1], div.result)
        yes.add_ops([div, add, scf.YieldOp(add.result)])
        no = Block()
        no.add_op(scf.YieldOp(body.args[1]))
        branch = scf.IfOp(valid, [elem], Region(yes), Region(no))
        body.add_ops([branch, scf.YieldOp(branch.results[0])])
        loop = scf.ForOp(z.result, n.result, o.result, [init.result], Region(body))
        block.add_op(loop)
        return loop.results[0]

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
