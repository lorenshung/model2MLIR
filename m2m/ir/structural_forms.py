"""Structural (window, movement) forms for pools, pads, repeat, select_scatter and col2im.

The gather-loop kernels in ``m2m.ir.structural`` (``linalg.index`` + ``scf.for`` +
``tensor.extract``) and the upstream composites (arange/abs/``index_gather``) are
semantically exact but opaque to a structural family matcher: nothing in them says
"sum over a window" or "copy". These lowerings emit the same values with
recognizable forms:

* average pools: a zero pad, then one ``linalg.generic`` per window run whose input
  map is affine ``x[o*stride + k + offset]`` with the kernel dims as reduction
  iterators and an ``add`` body (a windowed sum reduction), then a division by a
  constant divisor (scalar or per-position tensor). Non-divisible adaptive pools are
  split into runs of equal window size and step and reassembled with ``insert_slice``;
* pool backwards and col2im: a zero accumulator updated tap by tap with a strided
  ``extract_slice`` / elementwise add / ``insert_slice`` (an accumulate of shifted
  windows), then the interior ``extract_slice``;
* reflection/replication pads: width-1 ``extract_slice`` mirror/edge copies and
  ``tensor.concat`` per padded axis;
* repeat: ``linalg.broadcast`` then a row-major reshape;
* select_scatter: ``tensor.insert_slice`` of the unsqueezed source.

Every arithmetic step keeps the CPU kernel's operand order: window sums add in
lexicographic tap order from zero, backward gradients and col2im accumulate their
contributions in the CPU loop order, and the divisors are exactly the CPU
divisors. ``M2M_WINDOW_LOWERING=gather`` restores the previous kernels.
"""

from __future__ import annotations

import math
import os
from itertools import product

from xdsl.dialects import arith, tensor
from xdsl.dialects.builtin import (
    AffineMapAttr,
    AnyFloat,
    BFloat16Type,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    Float16Type,
    Float32Type,
    IntegerType,
    StringAttr,
    TensorType,
    i64,
)
from xdsl.dialects.linalg import BroadcastOp, GenericOp, IteratorType, IteratorTypeAttr, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineExpr, AffineMap

MAX_WINDOW_RUNS = 64


def structural_enabled() -> bool:
    return os.environ.get("M2M_WINDOW_LOWERING", "structural") != "gather"


def _d():
    from m2m.ir import decompositions

    return decompositions


def _tag(ops, role, movement=None):
    for op in ops:
        if "prov.role" not in op.attributes:
            op.attributes["prov.role"] = StringAttr(role)
        if movement is not None and "prov.movement_form" not in op.attributes:
            op.attributes["prov.movement_form"] = StringAttr(movement)


def _add_cls(elem):
    if isinstance(elem, IntegerType) and elem.bitwidth == 1:
        return arith.OrIOp
    return arith.AddfOp if isinstance(elem, AnyFloat) else arith.AddiOp


def _div_cls(elem):
    return arith.DivfOp if isinstance(elem, AnyFloat) else arith.DivSIOp


def _constant_tensor(values, shape, elem):
    """A dense constant (exact small integers), built in f32/i64 and cast if needed."""
    d = _d()
    ops = []
    if isinstance(elem, IntegerType):
        typ = TensorType(elem, shape)
        const = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(typ, [int(v) for v in values]), typ)
        return [const], const.result
    typ = TensorType(Float32Type(), shape)
    const = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(typ, [float(v) for v in values]), typ)
    ops.append(const)
    value = const.result
    if not isinstance(elem, Float32Type):
        casts, value = d._cast_tensor(value, shape, elem)
        ops += casts
    return ops, value


def _divide(ops, value, shape, divisors, elem, *, prefix):
    """value / divisor, where divisors is a scalar or a tensor over the trailing dims."""
    d = _d()
    result_type = TensorType(elem, shape)
    flat = divisors if isinstance(divisors, list) else None
    if flat is None or len(set(flat)) == 1:
        scalar = divisors if flat is None else flat[0]
        made = d._splat_scalar(scalar, result_type)
        ops += made[0]
        denominator = made[1]
        maps = None
    else:
        trailing = shape[prefix:]
        const_ops, denominator = _constant_tensor(flat, trailing, elem)
        ops += const_ops
        rank = len(shape)
        maps = [AffineMap.identity(rank),
                AffineMap(rank, 0, tuple(AffineExpr.dimension(prefix + i) for i in range(len(trailing))))]
    built = d._elementwise([value, denominator], result_type, d._bin_build(_div_cls(elem)), input_maps=maps)
    if built is None:
        return None
    _tag(built[0], "divide")
    ops += built[0]
    return built[1]


# ---------------------------------------------------------------------------
# Window geometry
# ---------------------------------------------------------------------------


def _runs(starts, sizes):
    """Split output positions into runs of equal window size and constant step.

    Returns ``[(out_offset, length, first_start, step, size)]``.
    """
    runs = []
    o = 0
    count = len(starts)
    while o < count:
        length = 1
        step = 1
        if o + 1 < count and sizes[o + 1] == sizes[o]:
            step = starts[o + 1] - starts[o]
            while (o + length < count and sizes[o + length] == sizes[o]
                   and starts[o + length] - starts[o + length - 1] == step):
                length += 1
        runs.append((o, length, starts[o], step if length > 1 else 1, sizes[o]))
        o += length
    return runs


def _pool_axes(target, shape, out, meta, ndim, prefix, *, backward=False):
    """Per spatial axis: (runs, divisor widths per output position) or None."""
    d = _d()
    adaptive = "adaptive" in target

    base = 1 if backward else 0
    override = None
    if not adaptive:
        from m2m.ir.structural import pool_vector
        kernel = pool_vector(d._fx_arg(meta, base + 1, None), 1, ndim)
        stride = pool_vector(d._fx_arg(meta, base + 2, None), kernel, ndim)
        padding = pool_vector(d._fx_arg(meta, base + 3, None), 0, ndim)
        include = bool(d._fx_arg(meta, base + 5, True))
        override = d._fx_arg(meta, base + 6, None)
        if (len(kernel), len(stride), len(padding)) != (ndim,) * 3:
            return None
    axes = []
    for axis in range(ndim):
        size, osize = shape[prefix + axis], out[prefix + axis]
        starts, sizes, widths = [], [], []
        for o in range(osize):
            if adaptive:
                start = (o * size) // osize
                end = -(-((o + 1) * size) // osize)
                starts.append(start)
                sizes.append(end - start)
                widths.append(end - start)
            else:
                start = o * stride[axis] - padding[axis]
                end = min(start + kernel[axis], size + padding[axis])
                low, high = max(start, 0), min(end, size)
                # The window reads [start, start + kernel); positions past the
                # clipped end lie beyond the input and read the zero pad.
                starts.append(start)
                sizes.append(kernel[axis])
                widths.append(end - start if include else high - low)
        axes.append((_runs(starts, sizes), widths))
    return axes, override


def _divisor_sequence(axes, combo, override, *, adaptive):
    """The CPU divisions of a window block, in order.

    A regular pool divides once by the product of the window widths (or the
    override); an adaptive pool divides by each axis width in turn
    (``sum / kh / kw``), which rounds differently from one product division.
    """
    if override is not None:
        return [override]
    slices = [axes[a][1][run[0]:run[0] + run[1]] for a, run in enumerate(combo)]
    if not adaptive:
        return [[math.prod(widths) for widths in product(*slices)]]
    sequence = []
    for axis in range(len(combo)):
        sequence.append([widths[axis] for widths in product(*slices)])
    return sequence


def _pad_for(runs, size):
    low = max(0, -min(run[2] for run in runs))
    high = max(0, max(run[2] + (run[1] - 1) * run[3] + run[4] for run in runs) - size)
    return low, high


def _acc_elem(target, elem, ndim):
    adaptive = "adaptive" in target
    # Native adaptive 3D pooling keeps its running sum in scalar_t, the others in opmath.
    if isinstance(elem, (Float16Type, BFloat16Type)) and not (adaptive and ndim == 3):
        return Float32Type()
    return elem


# ---------------------------------------------------------------------------
# Average pools
# ---------------------------------------------------------------------------


def avg_pool_windows(operands, meta, node_name):
    """Regular and adaptive average pooling as windowed sum reductions."""
    from m2m.ir.structural import avg_pool

    d = _d()
    if not structural_enabled():
        return avg_pool(operands, meta, node_name)
    x = operands[0]
    shape = d._shape_of(x)
    elem = d._t_elem(x)
    target = meta["_aten_target"]
    ndim = 3 if "3d" in target else 2 if "2d" in target else 1
    out = list(meta["val"].shape)
    if shape is None or any(v <= 0 for v in (*shape, *out)) or len(shape) != len(out):
        return avg_pool(operands, meta, node_name)
    prefix = len(shape) - ndim
    geometry = _pool_axes(target, shape, out, meta, ndim, prefix)
    if geometry is None:
        return avg_pool(operands, meta, node_name)
    axes, override = geometry
    if math.prod(len(runs) for runs, _ in axes) > MAX_WINDOW_RUNS:
        return avg_pool(operands, meta, node_name)
    acc = _acc_elem(target, elem, ndim)
    ops = []
    casts, x = d._cast_tensor(x, shape, acc)
    _tag(casts, "cast")
    ops += casts
    pads = [_pad_for(runs, shape[prefix + a]) for a, (runs, _) in enumerate(axes)]
    if any(lo or hi for lo, hi in pads):
        padded = d._zero_pad_ops(x, acc, list(shape), [(0, 0)] * prefix + pads)
        if padded is None:
            return avg_pool(operands, meta, node_name)
        _tag(padded[0], "pad")
        ops += padded[0]
        x = padded[1]
    combos = list(product(*(runs for runs, _ in axes)))
    result = None
    if len(combos) > 1:
        splat = d._splat_scalar(0, TensorType(acc, out))
        _tag(splat[0], "acc_init")
        ops += splat[0]
        result = splat[1]
    D = AffineExpr.dimension
    rank = prefix + 2 * ndim
    for combo in combos:
        block_shape = [*out[:prefix], *(run[1] for run in combo)]
        init = d._splat_scalar(0, TensorType(acc, block_shape))
        _tag(init[0], "acc_init")
        ops += init[0]
        in_exprs = [D(i) for i in range(prefix)]
        for a, run in enumerate(combo):
            offset = run[2] + pads[a][0]
            in_exprs.append(D(prefix + a) * run[3] + D(prefix + ndim + a) + offset)
        # The shape-only window operand bounds the kernel (reduction) dims, as in
        # linalg.pooling_*_sum: ins(input, window) with the window read by (k...).
        shape_operand = d._make_empty(TensorType(acc, [run[4] for run in combo]))
        _tag([shape_operand], "window_shape")
        ops.append(shape_operand)
        block = Block(arg_types=[acc, acc, acc])
        add = _add_cls(acc)(block.args[2], block.args[0])
        block.add_ops([add, YieldOp(add.result)])
        window = GenericOp(
            inputs=[x, shape_operand.results[0]],
            outputs=[init[1]],
            body=Region(block),
            indexing_maps=[AffineMapAttr(AffineMap(rank, 0, tuple(in_exprs))),
                           AffineMapAttr(AffineMap(rank, 0, tuple(D(prefix + ndim + a) for a in range(ndim)))),
                           AffineMapAttr(AffineMap(rank, 0, tuple(D(i) for i in range(prefix + ndim))))],
            iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)] * (prefix + ndim)
                           + [IteratorTypeAttr(IteratorType.REDUCTION)] * ndim,
            result_types=[TensorType(acc, block_shape)],
        )
        window.attributes["prov.window"] = StringAttr(
            "kernel=" + "x".join(str(run[4]) for run in combo)
            + ",stride=" + "x".join(str(run[3]) for run in combo))
        _tag([window], "window_sum")
        ops.append(window)
        quotient = window.results[0]
        for divisors in _divisor_sequence(axes, combo, override, adaptive="adaptive" in target):
            quotient = _divide(ops, quotient, block_shape, divisors, acc, prefix=prefix)
            if quotient is None:
                return avg_pool(operands, meta, node_name)
        if result is None:
            result = quotient
        else:
            insert = tensor.InsertSliceOp.from_static_parameters(
                quotient, result, [0] * prefix + [run[0] for run in combo], block_shape)
            _tag([insert], "assemble", "copy")
            ops.append(insert)
            result = insert.results[0]
    final, result = d._cast_tensor(result, out, elem)
    _tag(final, "cast")
    ops += final
    return d.DecompResult(ops=ops, result=result, pattern_hint="avg_pool")


def avg_pool_backward_windows(operands, meta, node_name):
    """avg_pool2d_backward / _adaptive_avg_pool2d_backward as an accumulate of shifted windows."""
    from m2m.ir.structural import avg_pool_backward

    d = _d()
    if not structural_enabled():
        return avg_pool_backward(operands, meta, node_name)
    grad, x = operands[:2]
    shape, gshape = d._shape_of(x), d._shape_of(grad)
    elem = d._t_elem(grad)
    target = meta["_aten_target"]
    ndim = 2
    if (shape is None or gshape is None or len(shape) != len(gshape) or len(shape) < ndim
            or any(v <= 0 for v in (*shape, *gshape))):
        return avg_pool_backward(operands, meta, node_name)
    prefix = len(shape) - ndim
    geometry = _pool_axes(target, shape, gshape, meta, ndim, prefix, backward=True)
    if geometry is None:
        return avg_pool_backward(operands, meta, node_name)
    axes, override = geometry
    if math.prod(sum(run[4] * 1 for run in runs) for runs, _ in axes) > 4096:
        return avg_pool_backward(operands, meta, node_name)
    ops = []
    delta = grad
    # Divisors per output position: one product (regular) or one per axis (adaptive).
    whole = [(0, gshape[prefix + a], 0, 1, 1) for a in range(ndim)]
    for divisors in _divisor_sequence(axes, whole, override, adaptive="adaptive" in target):
        delta = _divide(ops, delta, list(gshape), divisors, elem, prefix=prefix)
        if delta is None:
            return avg_pool_backward(operands, meta, node_name)
    pads = [_pad_for(runs, shape[prefix + a]) for a, (runs, _) in enumerate(axes)]
    padded_shape = [*shape[:prefix], *(shape[prefix + a] + lo + hi for a, (lo, hi) in enumerate(pads))]
    zero = d._splat_scalar(0, TensorType(elem, padded_shape))
    _tag(zero[0], "acc_init")
    ops += zero[0]
    acc = zero[1]
    # CPU adds each output's delta to its window in lexicographic output order. Per
    # axis, iterate runs ascending and, inside a run, taps descending: one input
    # element then receives its contributions in exactly that order.
    per_axis = [[(run, tap) for run in runs for tap in reversed(range(run[4]))] for runs, _ in axes]
    for steps in product(*per_axis):
        block = [*gshape[:prefix], *(run[1] for run, _ in steps)]
        offsets = [0] * prefix + [run[2] + pads[a][0] + tap for a, (run, tap) in enumerate(steps)]
        strides = [1] * prefix + [run[3] for run, _ in steps]
        update = delta
        if block != list(gshape):
            piece = tensor.ExtractSliceOp.from_static_parameters(
                delta, [0] * prefix + [run[0] for run, _ in steps], block, [1] * len(block))
            _tag([piece], "window_block", "copy")
            ops.append(piece)
            update = piece.results[0]
        acc = _accumulate(ops, acc, update, offsets, block, strides, elem)
    interior = tensor.ExtractSliceOp.from_static_parameters(
        acc, [0] * prefix + [lo for lo, _ in pads], list(shape), [1] * len(shape))
    _tag([interior], "interior", "copy")
    ops.append(interior)
    return d.DecompResult(ops=ops, result=interior.results[0], pattern_hint="avg_pool_backward")


def _accumulate(ops, acc, update, offsets, sizes, strides, elem, opmath=None):
    """acc[offsets::strides] += update, as extract_slice / add / insert_slice.

    With ``opmath`` the add widens both storage values, adds, and narrows back."""
    d = _d()
    window = tensor.ExtractSliceOp.from_static_parameters(acc, offsets, sizes, strides)
    if opmath is not None and opmath != elem:
        def build(args, out_elem):
            body = []
            wide = []
            for value in args:
                cast, value = d._cast_scalar_arg(value, opmath)
                body += cast
                wide.append(value)
            add = _add_cls(opmath)(wide[0], wide[1])
            narrow, value = d._cast_scalar_arg(add.result, out_elem)
            return [*body, add, *narrow], value
        added = d._elementwise([window.results[0], update], TensorType(elem, sizes), build)
    else:
        added = d._elementwise([window.results[0], update], TensorType(elem, sizes), d._bin_build(_add_cls(elem)))
    insert = tensor.InsertSliceOp.from_static_parameters(added[1], acc, offsets, sizes, strides)
    _tag([window, *added[0], insert], "tap_accumulate")
    ops += [window, *added[0], insert]
    return insert.results[0]


def conv_input_gradient(grad, weight, shape, *, stride, padding, dilation, groups, storage_elem):
    """convolution_backward grad_input as a column contraction plus tap accumulation.

    CPU computes the backward GEMM columns ``cols[c, k, o] = sum_f w[f, c, k] * dy[f, o]``
    (f ascending, opmath), stores them in input storage, then col2im adds each tap
    into input storage in ascending tap order. This emits exactly that: one
    single-reduction ``linalg.generic`` contraction over the group's filters
    (``prov.role = "contraction"``), a cast to storage, and per-tap strided
    accumulation (``prov.role = "tap_accumulate"``). Only real forward edges ever
    contribute a product. Returns ``(ops, value)`` or None.
    """
    from xdsl.dialects.builtin import AffineMapAttr

    d = _d()
    elem = grad.type.element_type
    gshape, wshape = list(grad.type.get_shape()), list(weight.type.get_shape())
    if len(shape) != 4 or len(gshape) != 4 or len(wshape) != 4 or not isinstance(elem, AnyFloat):
        return None
    n, c_in, h, w = shape
    _, f_out, oh, ow = gshape
    _, c_per_g, kh, kw = wshape
    g = int(groups)
    fg = f_out // g
    D = AffineExpr.dimension
    # dims: n=0, g=1, cg=2, kh=3, kw=4, oh=5, ow=6, f=7
    cols_shape = [n, g, c_per_g, kh, kw, oh, ow]
    zero_ops, zero = d._zero_acc(TensorType(elem, cols_shape), elem)
    block = Block(arg_types=[elem, elem, elem])
    multiply = arith.MulfOp(block.args[1], block.args[0])
    total = arith.AddfOp(block.args[2], multiply.result)
    block.add_ops([multiply, total, YieldOp(total.result)])
    # Split F into (G, F/G) so every loop dim is read directly (invertible maps).
    w_split = d._emit_reshape(weight, [g, fg, c_per_g, kh, kw], elem)
    g_split = d._emit_reshape(grad, [n, g, fg, oh, ow], elem)
    if w_split is None or g_split is None:
        return None
    _tag([*w_split[0], *g_split[0]], "operand_layout")
    columns = GenericOp(
        inputs=[w_split[1], g_split[1]],
        outputs=[zero],
        body=Region(block),
        indexing_maps=[AffineMapAttr(AffineMap(8, 0, (D(1), D(7), D(2), D(3), D(4)))),
                       AffineMapAttr(AffineMap(8, 0, (D(0), D(1), D(7), D(5), D(6)))),
                       AffineMapAttr(AffineMap(8, 0, tuple(D(i) for i in range(7))))],
        iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)] * 7 + [IteratorTypeAttr(IteratorType.REDUCTION)],
        result_types=[TensorType(elem, cols_shape)],
    )
    _tag(zero_ops, "acc_init")
    _tag([columns], "contraction")
    ops = [*w_split[0], *g_split[0], *zero_ops, columns]
    stored = columns.results[0]
    if storage_elem != elem:
        casts, stored = d._cast_tensor(stored, cols_shape, storage_elem)
        _tag(casts, "cast")
        ops += casts
    padded_shape = [n, c_in, h + 2 * padding[0], w + 2 * padding[1]]
    acc_zero = d._splat_scalar(0, TensorType(storage_elem, padded_shape))
    _tag(acc_zero[0], "acc_init")
    ops += acc_zero[0]
    acc = acc_zero[1]
    sizes = [n, c_in, oh, ow]
    for ky, kx in product(range(kh), range(kw)):
        tap = tensor.ExtractSliceOp.from_static_parameters(
            stored, [0, 0, 0, ky, kx, 0, 0], [n, g, c_per_g, 1, 1, oh, ow], [1] * 7)
        squeezed = d._emit_reshape(tap.results[0], sizes, storage_elem)
        if squeezed is None:
            return None
        _tag([tap, *squeezed[0]], "column_tap", "copy")
        ops += [tap, *squeezed[0]]
        acc = _accumulate(ops, acc, squeezed[1], [0, 0, ky * dilation[0], kx * dilation[1]], sizes,
                          [1, 1, stride[0], stride[1]], storage_elem, opmath=elem)
    interior = tensor.ExtractSliceOp.from_static_parameters(acc, [0, 0, *padding], list(shape), [1] * 4)
    _tag([interior], "interior", "copy")
    ops.append(interior)
    return ops, interior.results[0]


# ---------------------------------------------------------------------------
# col2im
# ---------------------------------------------------------------------------


def col2im_accumulate(operands, meta, node_name):
    """aten.col2im: each kernel tap's column block is added into a strided window."""
    d = _d()
    fallback = _upstream_fallback("aten.col2im.default")
    if not structural_enabled():
        return fallback(operands, meta, node_name)
    cols = operands[0]
    cshape = d._shape_of(cols)
    elem = d._t_elem(cols)

    def pair(index):
        raw = d._fx_arg(meta, index, None)
        values = [raw, raw] if isinstance(raw, int) else list(raw or ())
        return values if len(values) == 2 and all(isinstance(v, int) for v in values) else None

    output, kernel, dilation, padding, stride = (pair(i) for i in range(1, 6))
    if cshape is None or None in (output, kernel, dilation, padding, stride) or len(cshape) not in (2, 3):
        return fallback(operands, meta, node_name)
    batched = len(cshape) == 3
    batch = cshape[0] if batched else None
    rows, positions = cshape[-2:]
    taps = kernel[0] * kernel[1]
    if rows % taps:
        return fallback(operands, meta, node_name)
    channels = rows // taps
    blocks = [(output[i] + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1) // stride[i] + 1 for i in range(2)]
    if min(blocks) <= 0 or blocks[0] * blocks[1] != positions:
        return fallback(operands, meta, node_name)
    lead = [batch] if batched else []
    ops = []
    reshaped = d._emit_reshape(cols, [*lead, channels, kernel[0], kernel[1], *blocks], elem)
    if reshaped is None:
        return fallback(operands, meta, node_name)
    _tag(reshaped[0], "column_layout")
    ops += reshaped[0]
    columns = reshaped[1]
    padded_shape = [*lead, channels, output[0] + 2 * padding[0], output[1] + 2 * padding[1]]
    zero = d._splat_scalar(0, TensorType(elem, padded_shape))
    _tag(zero[0], "acc_init")
    ops += zero[0]
    acc = zero[1]
    prefix = len(lead) + 1
    sizes = [*lead, channels, *blocks]
    for kh, kw in product(range(kernel[0]), range(kernel[1])):
        tap = tensor.ExtractSliceOp.from_static_parameters(
            columns, [0] * prefix + [kh, kw, 0, 0], [*lead, channels, 1, 1, *blocks], [1] * (prefix + 4))
        squeezed = d._emit_reshape(tap.results[0], sizes, elem)
        _tag([tap, *squeezed[0]], "column_tap", "copy")
        ops += [tap, *squeezed[0]]
        acc = _accumulate(ops, acc, squeezed[1], [0] * prefix + [kh * dilation[0], kw * dilation[1]],
                          sizes, [1] * prefix + stride, elem)
    interior = tensor.ExtractSliceOp.from_static_parameters(
        acc, [0] * prefix + padding, [*lead, channels, *output], [1] * (prefix + 2))
    _tag([interior], "interior", "copy")
    ops.append(interior)
    return d.DecompResult(ops=ops, result=interior.results[0], pattern_hint="col2im")


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------


def _upstream_fallback(target):
    def lower(operands, meta, node_name):
        import torch
        from torch._decomp import decomposition_table

        from m2m.ir.core_aten import upstream_lowering

        overload = eval("torch.ops." + target, {"torch": torch})  # noqa: S307 - fixed op names
        return upstream_lowering(decomposition_table[overload])(operands, meta, node_name)
    return lower


def _edge_pad(operands, meta, node_name, *, reflect):
    d = _d()
    target = meta["_aten_target"]
    fallback = _upstream_fallback(target)
    if not structural_enabled():
        return fallback(operands, meta, node_name)
    x = operands[0]
    shape = d._shape_of(x)
    elem = d._t_elem(x)
    pad = d._fx_arg(meta, 1, None)
    if shape is None or not isinstance(pad, (list, tuple)) or len(pad) % 2 or len(pad) // 2 > len(shape):
        return fallback(operands, meta, node_name)
    ops = []
    current, current_shape = x, list(shape)
    for i in range(len(pad) // 2):
        axis = len(shape) - 1 - i
        low, high = int(pad[2 * i]), int(pad[2 * i + 1])
        size = current_shape[axis]
        if low < 0 or high < 0 or (reflect and (low >= size or high >= size)) or size <= 0:
            return fallback(operands, meta, node_name)
        if not low and not high:
            continue
        if reflect:
            indices = [*range(low, 0, -1), None, *range(size - 2, size - 2 - high, -1)]
        else:
            indices = [*([0] * low), None, *([size - 1] * high)]
        pieces = []
        for index in indices:
            if index is None:
                pieces.append(current)
                continue
            offsets = [0] * len(shape)
            offsets[axis] = index
            sizes = list(current_shape)
            sizes[axis] = 1
            piece = tensor.ExtractSliceOp.from_static_parameters(current, offsets, sizes, [1] * len(shape))
            _tag([piece], "mirror_copy" if reflect else "edge_copy", "copy")
            ops.append(piece)
            pieces.append(piece.results[0])
        current_shape[axis] = size + low + high
        concat = tensor.ConcatOp(inputs=pieces, dim=axis, result_type=TensorType(elem, list(current_shape)))
        _tag([concat], "concat", "copy")
        ops.append(concat)
        current = concat.results[0]
    return d.DecompResult(ops=ops, result=current, pattern_hint="pad")


def reflection_pad(operands, meta, node_name):
    return _edge_pad(operands, meta, node_name, reflect=True)


def replication_pad(operands, meta, node_name):
    return _edge_pad(operands, meta, node_name, reflect=False)


def repeat_broadcast(operands, meta, node_name):
    """aten.repeat as linalg.broadcast to (r0, s0, r1, s1, ...) and a row-major reshape."""
    d = _d()
    if not structural_enabled() or not operands or not isinstance(operands[0].type, TensorType):
        return d.decompose_repeat(operands, meta, node_name)
    x = operands[0]
    shape = d._shape_of(x)
    repeats = d._fx_arg(meta, 1, None)
    out = list(meta["val"].shape)
    if (shape is None or not isinstance(repeats, (list, tuple)) or len(repeats) < len(shape)
            or any(v < 0 for v in shape) or any(not isinstance(r, int) or r < 0 for r in repeats)
            or 0 in (*shape, *repeats)):
        return d.decompose_repeat(operands, meta, node_name)
    elem = d._t_elem(x)
    extra = len(repeats) - len(shape)
    logical = [1] * extra + list(shape)
    if all(r == 1 for r in repeats) and not extra:
        return d.decompose_repeat(operands, meta, node_name)
    tiled = [v for r, s in zip(repeats, logical) for v in (r, s)]
    added = [2 * i for i in range(len(repeats))] + [2 * i + 1 for i in range(extra)]
    ops = []
    empty = d._make_empty(TensorType(elem, tiled))
    broadcast = BroadcastOp(x, empty.results[0], DenseArrayBase.from_list(i64, sorted(added)), TensorType(elem, tiled))
    _tag([empty, broadcast], "tile", "broadcast")
    ops += [empty, broadcast]
    reshaped = d._emit_reshape(broadcast.results[0], out, elem)
    if reshaped is None or [r * s for r, s in zip(repeats, logical)] != out:
        return d.decompose_repeat(operands, meta, node_name)
    _tag(reshaped[0], "tile_layout", "copy")
    ops += reshaped[0]
    return d.DecompResult(ops=ops, result=reshaped[1], pattern_hint="repeat")


def select_scatter_insert(operands, meta, node_name):
    """aten.select_scatter(x, src, dim, index) as one tensor.insert_slice."""
    d = _d()
    fallback = _upstream_fallback("aten.select_scatter.default")
    if not structural_enabled() or len(operands) < 2:
        return fallback(operands, meta, node_name)
    x, src = operands[:2]
    shape, sshape = d._shape_of(x), d._shape_of(src)
    dim, index = d._fx_arg(meta, 2, None), d._fx_arg(meta, 3, None)
    if (shape is None or sshape is None or not isinstance(dim, int) or not isinstance(index, int)
            or not shape or d._t_elem(src) != d._t_elem(x)):
        return fallback(operands, meta, node_name)
    dim %= len(shape)
    if not -shape[dim] <= index < shape[dim]:
        return fallback(operands, meta, node_name)
    index %= shape[dim]
    sizes = list(shape)
    sizes[dim] = 1
    if [v for i, v in enumerate(shape) if i != dim] != list(sshape):
        return fallback(operands, meta, node_name)
    elem = d._t_elem(x)
    ops = []
    expanded = d._emit_reshape(src, sizes, elem) if sizes else None
    if expanded is None:
        return fallback(operands, meta, node_name)
    _tag(expanded[0], "unsqueeze")
    ops += expanded[0]
    offsets = [0] * len(shape)
    offsets[dim] = index
    insert = tensor.InsertSliceOp.from_static_parameters(expanded[1], x, offsets, sizes)
    _tag([insert], "scatter", "copy")
    ops.append(insert)
    return d.DecompResult(ops=ops, result=insert.results[0], pattern_hint="select_scatter")


STRUCTURAL_DECOMPOSITIONS = {
    **{f"aten.{name}.default": avg_pool_windows for name in (
        "avg_pool1d", "avg_pool2d", "avg_pool3d", "adaptive_avg_pool1d",
        "_adaptive_avg_pool2d", "_adaptive_avg_pool3d", "adaptive_avg_pool2d", "adaptive_avg_pool3d")},
    "aten.avg_pool2d_backward.default": avg_pool_backward_windows,
    "aten._adaptive_avg_pool2d_backward.default": avg_pool_backward_windows,
    "aten.col2im.default": col2im_accumulate,
    "aten.reflection_pad1d.default": reflection_pad,
    "aten.reflection_pad2d.default": reflection_pad,
    "aten.reflection_pad3d.default": reflection_pad,
    "aten.replication_pad1d.default": replication_pad,
    "aten.replication_pad2d.default": replication_pad,
    "aten.replication_pad3d.default": replication_pad,
    "aten.repeat.default": repeat_broadcast,
    "aten.select_scatter.default": select_scatter_insert,
}
