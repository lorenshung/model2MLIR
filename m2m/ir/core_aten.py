"""Core ATen lowerings whose result ABI includes auxiliary tensors."""
from __future__ import annotations
import itertools
import math
from xdsl.dialects import arith, tensor, scf
from xdsl.dialects.builtin import AffineMapAttr, AnyFloat, IndexType, IntegerAttr, TensorType, i64
from xdsl.dialects.linalg import GenericOp, IndexOp, IteratorType, IteratorTypeAttr, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

def _index(value):
    return arith.ConstantOp(IntegerAttr(value, IndexType()), IndexType())

def normalize_index(value, extent):
    """Normalize a signed ATen index and diagnose invalid accesses before extraction."""
    from xdsl.dialects.cf import AssertOp
    zero, size = (_index(0), _index(extent))
    negative = arith.CmpiOp(value, zero.result, 'slt')
    wrapped = arith.AddiOp(value, size.result)
    chosen = arith.SelectOp(negative.result, wrapped.result, value)
    low = arith.CmpiOp(chosen.result, zero.result, 'sge')
    high = arith.CmpiOp(chosen.result, size.result, 'slt')
    valid = arith.AndIOp(low.result, high.result)
    check = AssertOp(valid.result, 'ATen index out of bounds')
    return ([zero, size, negative, wrapped, chosen, low, high, valid, check], chosen.result)

def max_pool_with_indices(operands, meta, node_name):
    """Window argmax with eager's first numeric tie / last NaN index policy.

    Coordinates outside the input are skipped, including ceil-mode edge windows;
    the returned indices address the flattened spatial input, independently of N/C.
    """
    from m2m.ir.decompositions import DecompResult, _fx_arg, _opaque_decomp, _shape_of, _splat_scalar, _t_elem
    x = operands[0]
    shape = _shape_of(x)
    values = meta.get('val')
    ndim = 3 if '3d' in meta.get('_aten_target', '') else 2

    def bail():
        return _opaque_decomp('aten_max_pool_with_indices', operands, meta, 'pool')
    if shape is None or len(shape) not in (ndim + 1, ndim + 2) or any((d < 0 for d in shape)) or (not isinstance(values, (tuple, list))) or (len(values) != 2) or (not isinstance(_t_elem(x), AnyFloat)):
        return bail()
    out_shape = list(values[0].shape)

    def control(pos, default):
        raw = _fx_arg(meta, pos, default)
        if raw is None or raw == []:
            raw = default
        if isinstance(raw, int):
            return [raw] * ndim
        if len(raw) == 1:
            return list(raw) * ndim
        return list(raw) if len(raw) == ndim else None
    kernel = control(1, None)
    if kernel is None:
        return bail()
    stride, padding, dilation = (control(2, kernel), control(3, [0] * ndim), control(4, [1] * ndim))
    if any((c is None for c in (stride, padding, dilation))) or math.prod(kernel) > 4096:
        return bail()
    elem = _t_elem(x)
    vt, it = (TensorType(elem, out_shape), TensorType(i64, out_shape))
    vops, initv = _splat_scalar(float('-inf'), vt)
    iops, initi = _splat_scalar(-1, it)
    block = Block(arg_types=[elem, i64])
    rank = len(shape)
    coordinates = []
    for axis in range(rank):
        ix = IndexOp(axis)
        block.add_op(ix)
        coordinates.append(ix.result)
    current, argmax = block.args
    for window in itertools.product(*(range(k) for k in kernel)):
        body = []
        source_coords = list(coordinates[:rank - ndim])
        valid = None
        spatial_linear = None
        for j in range(ndim):
            axis = rank - ndim + j
            step, offset, zero, size = (_index(stride[j]), _index(window[j] * dilation[j] - padding[j]), _index(0), _index(shape[axis]))
            scaled = arith.MuliOp(coordinates[axis], step.result)
            position = arith.AddiOp(scaled.result, offset.result)
            low = arith.CmpiOp(position.result, zero.result, 'sge')
            high = arith.CmpiOp(position.result, size.result, 'slt')
            inside = arith.AndIOp(low.result, high.result)
            body += [step, offset, zero, size, scaled, position, low, high, inside]
            if valid is None:
                valid = inside.result
            else:
                both = arith.AndIOp(valid, inside.result)
                body.append(both)
                valid = both.result
            source_coords.append(position.result)
            if spatial_linear is None:
                spatial_linear = position.result
            else:
                mul = arith.MuliOp(spatial_linear, size.result)
                add = arith.AddiOp(mul.result, position.result)
                body += [mul, add]
                spatial_linear = add.result
        block.add_ops(body)
        yes = Block()
        picked = tensor.ExtractOp(x, source_coords, elem)
        greater = arith.CmpfOp(picked.result, current, 'ogt')
        nan = arith.CmpfOp(picked.result, picked.result, 'uno')
        minus_one = arith.ConstantOp(IntegerAttr(-1, i64), i64)
        first = arith.CmpiOp(argmax, minus_one.result, 'eq')
        better = arith.OrIOp(greater.result, nan.result)
        take = arith.OrIOp(first.result, better.result)
        cast = arith.IndexCastOp(spatial_linear, i64)
        v = arith.SelectOp(take.result, picked.result, current)
        i = arith.SelectOp(take.result, cast.result, argmax)
        yes.add_ops([picked, greater, nan, minus_one, first, better, take, cast, v, i, scf.YieldOp(v.result, i.result)])
        no = Block()
        no.add_op(scf.YieldOp(current, argmax))
        branch = scf.IfOp(valid, [elem, i64], Region(yes), Region(no))
        block.add_op(branch)
        current, argmax = branch.results
    block.add_op(YieldOp(current, argmax))
    generic = GenericOp(inputs=[], outputs=[initv, initi], body=Region(block), indexing_maps=[AffineMapAttr(AffineMap.identity(rank))] * 2, iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)] * rank, result_types=[vt, it])
    return DecompResult(ops=[*vops, *iops, generic], result=generic.results[0], results=list(generic.results), pattern_hint='max_pool')

def max_pool_backward(operands, meta, node_name):
    """Scatter-add every upstream gradient at its saved spatial argmax."""
    from m2m.ir.decompositions import DecompResult, _shape_of, _splat_scalar, _t_elem, _opaque_decomp
    if len(operands) < 3:
        return _opaque_decomp('aten_max_pool_backward', operands, meta, 'pool')
    grad, source, indices = operands[:3]
    shape, gshape = (_shape_of(source), _shape_of(grad))
    ndim = 3 if '3d' in meta.get('_aten_target', '') else 2
    if shape is None or gshape is None or any((d < 0 for d in [*shape, *gshape])):
        return _opaque_decomp('aten_max_pool_backward', operands, meta, 'pool')
    elem = _t_elem(grad)
    rt = TensorType(elem, shape)
    ops, initial = _splat_scalar(0, rt)
    zero, one, count = (_index(0), _index(1), _index(math.prod(gshape)))
    ops += [zero, one, count]
    body = Block(arg_types=[IndexType(), rt])
    iv, accumulator = body.args
    coords = [None] * len(gshape)
    q = iv
    for axis in range(len(gshape) - 1, -1, -1):
        size = _index(gshape[axis])
        rem = arith.RemUIOp(q, size.result)
        body.add_ops([size, rem])
        coords[axis] = rem.result
        if axis:
            div = arith.DivUIOp(q, size.result)
            body.add_op(div)
            q = div.result
    picked = tensor.ExtractOp(indices, coords, indices.type.element_type)
    cast = arith.IndexCastOp(picked.result, IndexType())
    body.add_ops([picked, cast])
    norm_ops, linear = normalize_index(cast.result, math.prod(shape[-ndim:]))
    body.add_ops(norm_ops)
    spatial = [None] * ndim
    for j in range(ndim - 1, -1, -1):
        size = _index(shape[len(shape) - ndim + j])
        rem = arith.RemUIOp(linear, size.result)
        body.add_ops([size, rem])
        spatial[j] = rem.result
        if j:
            div = arith.DivUIOp(linear, size.result)
            body.add_op(div)
            linear = div.result
    destination = coords[:-ndim] + spatial
    incoming = tensor.ExtractOp(grad, coords, elem)
    prior = tensor.ExtractOp(accumulator, destination, elem)
    add = arith.AddfOp(prior.result, incoming.result) if isinstance(elem, AnyFloat) else arith.AddiOp(prior.result, incoming.result)
    insert = tensor.InsertOp(add.result, accumulator, destination)
    body.add_ops([incoming, prior, add, insert, scf.YieldOp(insert.result)])
    loop = scf.ForOp(zero.result, count.result, one.result, [initial], Region(body))
    ops.append(loop)
    return DecompResult(ops=ops, result=loop.results[0], pattern_hint='max_pool_backward')

def upstream_lowering(decomposition, local_decompositions=None):
    """Inline a framework decomposition at the IR boundary, retaining source identity.

    Trace only FakeTensor metadata, never sample values. Import the resulting
    primitive graph through the same owned emitters as an ordinary exported graph.
    """

    def lower(operands, meta, node_name):
        from types import SimpleNamespace
        import torch
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
        from m2m.ir.decompositions import DECOMPOSITION_TABLE, DecompResult
        from m2m.ir.import_fx import FXImporter
        from xdsl.dialects.func import FuncOp, ReturnOp
        raw = (meta['_fx_args'], meta.get('_fx_kwargs', {}))
        fake = tree_map(lambda x: x.meta['val'] if isinstance(x, torch.fx.Node) else x, raw)
        leaves, spec = tree_flatten(fake)
        positions = [i for i, v in enumerate(leaves) if isinstance(v, torch.Tensor)]
        tensors = [leaves[i] for i in positions]
        if len(tensors) != len(operands):
            raise ValueError('decomposition input SSA does not match its schema')

        def wrapper(*args):
            bound = list(leaves)
            for i, value in zip(positions, args, strict=True):
                bound[i] = value
            args, kwargs = tree_unflatten(bound, spec)
            return decomposition(*args, **kwargs)
        from torch._decomp import decomposition_table
        owned = {**DECOMPOSITION_TABLE, **(local_decompositions or {})}
        table = {k: v for k, v in decomposition_table.items() if str(k) not in owned}
        graph = make_fx(wrapper, decomposition_table=table, tracing_mode='fake')(*tensors)
        importer = FXImporter(emit_named_ops=meta.get('_emit_named_ops', False), materialize_get_attr=True,
                              dynamic_decompositions=dict(local_decompositions or {}))
        module = importer.import_graph(SimpleNamespace(graph=graph.graph, graph_module=graph, graph_signature=None))
        if importer.opaque_count or any((d.level == 'error' for d in importer.diagnostics)):
            raise ValueError('upstream decomposition contains unsupported primitives')
        from xdsl.dialects.builtin import StringAttr
        functions = [o for o in module.body.block.ops if isinstance(o, FuncOp)]
        if len(functions) != 1 or len(list(module.body.block.ops)) != 1:
            raise ValueError('decomposition requires module-level state')
        block = functions[0].body.block
        for arg, value in zip(block.args, operands, strict=True):
            arg.replace_all_uses_with(value)
        returned = next((o for o in block.ops if isinstance(o, ReturnOp)))
        results = list(returned.operands)
        ops = [o for o in list(block.ops) if o is not returned]
        for op in ops:
            op.detach()
            if 'prov.aten' in op.attributes:
                op.attributes['prov.lowered_aten'] = op.attributes['prov.aten']
            op.attributes['prov.aten'] = StringAttr(meta['_aten_target'])
        logical = meta.get('val')
        if isinstance(logical,(tuple,list)):
            remaining = iter(results)
            slots = [next(remaining) if isinstance(value,torch.Tensor) else None for value in logical]
        else:
            slots = []
        return DecompResult(ops=ops, result=results[0] if results else None,
                            results=slots, pattern_hint='composite')
    return lower

def register_core_decompositions(table):
    """Build importer-local lowerings for pure Core source nodes.

    Mutable overloads still need export functionalization so input mutations
    become explicit results. Registration is schema/tag driven, not corpus driven.
    """
    import torch
    from m2m.ir.decompositions import DECOMPOSITION_TABLE
    from torch._decomp import decomposition_table
    local = {}
    candidates = {**decomposition_table, **{k: v for k, v in table.items() if callable(v)}}
    for target, decomposition in candidates.items():
        if callable(decomposition) and torch.Tag.core in getattr(target, 'tags', ()) and (not target._schema.is_mutable) and (str(target) not in DECOMPOSITION_TABLE):
            local[str(target)] = upstream_lowering(decomposition, local)
    return local

def embedding_backward(operands, meta, node_name):
    """Dense embedding gradient with runtime frequency counts and padding skip."""
    from m2m.ir.decompositions import DecompResult, _fx_arg, _shape_of, _t_elem, _splat_scalar, _cast_tensor
    from xdsl.dialects.builtin import Float32Type, Float16Type, BFloat16Type
    from xdsl.dialects.cf import AssertOp
    grad, indices = operands[:2]
    ishape, gshape = (_shape_of(indices), _shape_of(grad))
    rows = int(_fx_arg(meta, 2))
    padding = int(_fx_arg(meta, 3, -1))
    frequency = bool(_fx_arg(meta, 4, False))
    if ishape is None or gshape is None or gshape[:-1] != ishape or any((d < 0 for d in [*ishape, *gshape])):
        raise ValueError('embedding gradient requires static compatible shapes')
    elem = _t_elem(grad)
    acc_elem = Float32Type() if isinstance(elem, (Float16Type, BFloat16Type)) else elem
    cast_ops, grad = _cast_tensor(grad, gshape, acc_elem)
    rt = TensorType(acc_elem, [rows, gshape[-1]])
    ops, initial = _splat_scalar(0, rt)
    ops = [*cast_ops, *ops]
    z, o, count, width = (_index(0), _index(1), _index(math.prod(ishape)), _index(gshape[-1]))
    ops += [z, o, count, width]

    def coordinates(block, linear):
        coords = [None] * len(ishape)
        for axis in range(len(ishape) - 1, -1, -1):
            size = _index(ishape[axis])
            rem = arith.RemUIOp(linear, size.result)
            block.add_ops([size, rem])
            coords[axis] = rem.result
            if axis:
                div = arith.DivUIOp(linear, size.result)
                block.add_op(div)
                linear = div.result
        return coords
    body = Block(arg_types=[IndexType(), rt])
    iv, acc = body.args
    coords = coordinates(body, iv)
    picked = tensor.ExtractOp(indices, coords, indices.type.element_type)
    row = arith.IndexCastOp(picked.result, IndexType())
    pad = _index(padding)
    active = arith.CmpiOp(row.result, pad.result, 'ne')
    body.add_ops([picked, row, pad, active])
    yes = Block()
    limit = _index(rows)
    low = arith.CmpiOp(row.result, z.result, 'sge')
    high = arith.CmpiOp(row.result, limit.result, 'slt')
    valid = arith.AndIOp(low.result, high.result)
    check = AssertOp(valid.result, 'embedding index out of bounds')
    yes.add_ops([limit, low, high, valid, check])
    divisor = None
    if frequency:
        counter = Block(arg_types=[IndexType(), IndexType()])
        j, n = counter.args
        others = coordinates(counter, j)
        other = tensor.ExtractOp(indices, others, indices.type.element_type)
        equal = arith.CmpiOp(other.result, picked.result, 'eq')
        step = arith.SelectOp(equal.result, o.result, z.result)
        added = arith.AddiOp(n, step.result)
        counter.add_ops([other, equal, step, added, scf.YieldOp(added.result)])
        freq = scf.ForOp(z.result, count.result, o.result, [z.result], Region(counter))
        cast = arith.IndexCastOp(freq.results[0], i64)
        fp = arith.SIToFPOp(cast.result, acc_elem)
        yes.add_ops([freq, cast, fp])
        divisor = fp.result
    columns = Block(arg_types=[IndexType(), rt])
    col, dst = columns.args
    value = tensor.ExtractOp(grad, [*coords, col], acc_elem)
    prior = tensor.ExtractOp(dst, [row.result, col], acc_elem)
    columns.add_ops([value, prior])
    contribution = value.result
    if divisor is not None:
        divided = arith.DivfOp(contribution, divisor)
        columns.add_op(divided)
        contribution = divided.result
    add = arith.AddfOp(prior.result, contribution)
    put = tensor.InsertOp(add.result, dst, [row.result, col])
    columns.add_ops([add, put, scf.YieldOp(put.result)])
    col_loop = scf.ForOp(z.result, width.result, o.result, [acc], Region(columns))
    yes.add_ops([col_loop, scf.YieldOp(col_loop.results[0])])
    no = Block()
    no.add_op(scf.YieldOp(acc))
    branch = scf.IfOp(active.result, [rt], Region(yes), Region(no))
    body.add_ops([branch, scf.YieldOp(branch.results[0])])
    loop = scf.ForOp(z.result, count.result, o.result, [initial], Region(body))
    ops.append(loop)
    casts, result = _cast_tensor(loop.results[0], [rows, gshape[-1]], elem)
    ops += casts
    return DecompResult(ops=ops, result=result, pattern_hint='embedding_backward')

def embedding_bag_sum(operands, meta, node_name):
    """CPU forward sum bags, including the slow-path offset-to-bag mapping.

    Mean/max, backward flags and include_last_offset are explicitly refused.
    The latter has kernel-dependent behavior for a nonterminal last offset.
    """
    from m2m.ir.decompositions import DecompResult, _fx_arg, _shape_of, _t_elem, _splat_scalar
    from xdsl.dialects.cf import AssertOp
    weight, indices, offsets = operands[:3]
    if (int(_fx_arg(meta, 4, 0)) != 0 or bool(_fx_arg(meta, 3, False))
            or bool(_fx_arg(meta, 5, False)) or bool(_fx_arg(meta, 7, False))):
        raise ValueError('embedding_bag lowering supports forward sum mode without include_last_offset')
    values = meta['val']
    shape = list(values[0].shape)
    elem = _t_elem(weight)
    from xdsl.dialects.builtin import Float32Type, Float64Type
    if not isinstance(elem, (Float32Type, Float64Type)):
        raise ValueError('embedding_bag lowering supports float32/float64 accumulation')
    padding = int(_fx_arg(meta, 8, -1))
    length = _shape_of(indices)[0]
    n_offsets = _shape_of(offsets)[0]
    if length < 0 or n_offsets <= 0 or any(d < 0 for d in _shape_of(weight)):
        raise ValueError('embedding_bag lowering requires static nonempty offsets')
    sample = operands[3] if _fx_arg(meta, 6, None) is not None else None
    ops, initial = _splat_scalar(0, TensorType(elem, shape))
    z, o, n, rows = (_index(0), _index(1), _index(length), _index(_shape_of(weight)[0]))
    ops += [z, o, n, rows]
    block = Block(arg_types=[elem])
    bag, feature = (IndexOp(0), IndexOp(1))
    block.add_ops([bag, feature])
    start = tensor.ExtractOp(offsets, [bag.result], offsets.type.element_type)
    starti = arith.IndexCastOp(start.result, IndexType())
    nextbag = arith.AddiOp(bag.result, o.result)
    limit = _index(n_offsets)
    exists = arith.CmpiOp(nextbag.result, limit.result, 'slt')
    block.add_ops([start, starti, nextbag, limit, exists])
    yes = Block()
    last = tensor.ExtractOp(offsets, [nextbag.result], offsets.type.element_type)
    lasti = arith.IndexCastOp(last.result, IndexType())
    yes.add_ops([last, lasti, scf.YieldOp(lasti.result)])
    no = Block()
    no.add_op(scf.YieldOp(n.result))
    end = scf.IfOp(exists.result, [IndexType()], Region(yes), Region(no))
    block.add_op(end)
    low = arith.CmpiOp(starti.result, z.result, 'sge')
    order = arith.CmpiOp(starti.result, end.results[0], 'sle')
    high = arith.CmpiOp(end.results[0], n.result, 'sle')
    both = arith.AndIOp(low.result, order.result)
    valid = arith.AndIOp(both.result, high.result)
    check = AssertOp(valid.result, 'embedding_bag offsets out of bounds')
    block.add_ops([low, order, high, both, valid, check])
    body = Block(arg_types=[IndexType(), elem])
    position, acc = body.args
    picked = tensor.ExtractOp(indices, [position], indices.type.element_type)
    row = arith.IndexCastOp(picked.result, IndexType())
    pad = _index(padding)
    active = arith.CmpiOp(row.result, pad.result, 'ne')
    body.add_ops([picked, row, pad, active])
    yes = Block()
    low = arith.CmpiOp(row.result, z.result, 'sge')
    high = arith.CmpiOp(row.result, rows.result, 'slt')
    valid = arith.AndIOp(low.result, high.result)
    check = AssertOp(valid.result, 'embedding_bag index out of bounds')
    value = tensor.ExtractOp(weight, [row.result, feature.result], elem)
    yes.add_ops([low, high, valid, check, value])
    contribution = value.result
    if sample is not None:
        scale = tensor.ExtractOp(sample, [position], elem)
        mul = arith.MulfOp(contribution, scale.result)
        yes.add_ops([scale, mul])
        contribution = mul.result
    added = arith.AddfOp(acc, contribution)
    yes.add_ops([added, scf.YieldOp(added.result)])
    no = Block()
    no.add_op(scf.YieldOp(acc))
    branch = scf.IfOp(active.result, [elem], Region(yes), Region(no))
    body.add_ops([branch, scf.YieldOp(branch.results[0])])
    loop = scf.ForOp(starti.result, end.results[0], o.result, [block.args[0]], Region(body))
    block.add_ops([loop, YieldOp(loop.results[0])])
    generic = GenericOp(inputs=[], outputs=[initial], body=Region(block), indexing_maps=[AffineMapAttr(AffineMap.identity(2))], iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)] * 2, result_types=[TensorType(elem, shape)])
    ops.append(generic)
    results = [generic.results[0]]
    # Eager promotes mixed int32/int64 indices and offsets. FakeTensor's CPU
    # metadata currently uses offsets.dtype alone, so derive the ABI here.
    aux_elem = max((indices.type.element_type, offsets.type.element_type),
                   key=lambda dtype: dtype.bitwidth)
    for ordinal, value in enumerate(values[1:], start=1):
        extra, ssa = _splat_scalar(0, TensorType(aux_elem, list(value.shape)))
        ops += extra
        if ordinal == 1 and value.numel():
            # Sum's fast path returns an empty offset2bag. On the slow path
            # every input position belongs to the last offset <= its position,
            # including padding positions and duplicate offsets (empty bags).
            mapping = Block(arg_types=[ssa.type.element_type])
            position = IndexOp(0)
            mapping.add_op(position)
            scan = Block(arg_types=[IndexType(), ssa.type.element_type])
            bag_index, current = scan.args
            offset = tensor.ExtractOp(offsets, [bag_index], offsets.type.element_type)
            cast = arith.IndexCastOp(offset.result, IndexType())
            belongs = arith.CmpiOp(cast.result, position.result, 'sle')
            bag_value = arith.IndexCastOp(bag_index, ssa.type.element_type)
            selected = arith.SelectOp(belongs.result, bag_value.result, current)
            scan.add_ops([offset, cast, belongs, bag_value, selected, scf.YieldOp(selected.result)])
            limit = _index(n_offsets)
            loop = scf.ForOp(z.result, limit.result, o.result, [mapping.args[0]], Region(scan))
            mapping.add_ops([loop, YieldOp(loop.results[0])])
            result = GenericOp(inputs=[], outputs=[ssa], body=Region(mapping),
                               indexing_maps=[AffineMapAttr(AffineMap.identity(1))],
                               iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)],
                               result_types=[ssa.type])
            ops += [limit, result]
            ssa = result.results[0]
        # In CPU sum mode bag_size and max_indices are defined zero tensors.
        results.append(ssa)
    return DecompResult(ops=ops, result=results[0], results=results, pattern_hint='embedding_bag')


def static_query(operands,meta,node_name):
    """Lower static metadata to constants and dynamic dimensions to tensor.dim."""
    from m2m.ir.decompositions import DecompResult, _fx_arg
    from xdsl.dialects.builtin import i1
    value=meta['val']
    if isinstance(value,(bool,int)):
        dtype=i1 if isinstance(value,bool) else i64
        constant=arith.ConstantOp(IntegerAttr(int(value),dtype),dtype)
        return DecompResult(ops=[constant],result=constant.result,pattern_hint='local_scalar')
    target=meta['_aten_target']
    source=operands[0]
    rank=len(source.type.get_shape())
    if target not in ('aten.sym_size.int','aten.sym_numel.default'):
        raise ValueError('runtime storage-layout query needs a descriptor ABI')
    axes=[int(_fx_arg(meta,1)) % rank] if target=='aten.sym_size.int' else list(range(rank))
    ops=[]; result=None
    for axis in axes:
        coordinate=_index(axis)
        dim=tensor.DimOp(source,coordinate.result)
        cast=arith.IndexCastOp(dim.result,i64)
        ops += [coordinate,dim,cast]
        if result is None: result=cast.result
        else:
            multiplied=arith.MuliOp(result,cast.result);ops.append(multiplied);result=multiplied.result
    if result is None:
        constant=arith.ConstantOp(IntegerAttr(1,i64),i64);ops.append(constant);result=constant.result
    return DecompResult(ops=ops,result=result,pattern_hint='local_scalar')
