"""Core ATen lowerings whose result ABI includes auxiliary tensors."""
from __future__ import annotations
import itertools
import math
from xdsl.dialects import arith, tensor, scf
from xdsl.dialects.builtin import AffineMapAttr, AnyFloat, IndexType, IntegerAttr, TensorType, IntegerType, i64
from xdsl.dialects.linalg import GenericOp, IndexOp, IteratorType, IteratorTypeAttr, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from m2m.ir.builder import Builder
from m2m.ir.structural import delinearize

def normalize_index(b, value, extent):
    """Normalize a signed ATen index and diagnose invalid accesses before extraction."""
    from xdsl.dialects.cf import AssertOp
    zero, size = (b.ix(0), b.ix(extent))
    chosen = b(arith.SelectOp(b(arith.CmpiOp(value, zero, 'slt')), b(arith.AddiOp(value, size)), value))
    valid = b(arith.AndIOp(b(arith.CmpiOp(chosen, zero, 'sge')), b(arith.CmpiOp(chosen, size, 'slt'))))
    b(AssertOp(valid, 'ATen index out of bounds'))
    return chosen

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
    if shape is None or len(shape) not in (ndim + 1, ndim + 2) or any((d < 0 for d in shape)) or (not isinstance(values, (tuple, list))) or (len(values) != 2) or (not isinstance(_t_elem(x), (AnyFloat, IntegerType))):
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
    vops, initv = _splat_scalar(-(1 << (elem.bitwidth - 1)) if isinstance(elem, IntegerType) else float('-inf'), vt)
    iops, initi = _splat_scalar(-1, it)
    block = Block(arg_types=[elem, i64])
    b = Builder(block)
    rank = len(shape)
    coordinates = [b(IndexOp(axis)) for axis in range(rank)]
    current, argmax = block.args
    integer = isinstance(elem, IntegerType)
    for window in itertools.product(*(range(k) for k in kernel)):
        source_coords = list(coordinates[:rank - ndim])
        valid = None
        spatial_linear = None
        for j in range(ndim):
            axis = rank - ndim + j
            step, offset, zero, size = (b.ix(stride[j]), b.ix(window[j] * dilation[j] - padding[j]), b.ix(0), b.ix(shape[axis]))
            position = b(arith.AddiOp(b(arith.MuliOp(coordinates[axis], step)), offset))
            inside = b(arith.AndIOp(b(arith.CmpiOp(position, zero, 'sge')), b(arith.CmpiOp(position, size, 'slt'))))
            valid = inside if valid is None else b(arith.AndIOp(valid, inside))
            source_coords.append(position)
            spatial_linear = position if spatial_linear is None else b(arith.AddiOp(b(arith.MuliOp(spatial_linear, size)), position))
        yes = Block()
        y = Builder(yes)
        picked = y(tensor.ExtractOp(x, source_coords, elem))
        greater = y(arith.CmpiOp(picked, current, 'sgt') if integer else arith.CmpfOp(picked, current, 'ogt'))
        nan = y(arith.CmpiOp(picked, picked, 'ne') if integer else arith.CmpfOp(picked, picked, 'uno'))
        first = y(arith.CmpiOp(argmax, y(arith.ConstantOp(IntegerAttr(-1, i64), i64)), 'eq'))
        take = y(arith.OrIOp(first, y(arith.OrIOp(greater, nan))))
        cast = y(arith.IndexCastOp(spatial_linear, i64))
        y(scf.YieldOp(y(arith.SelectOp(take, picked, current)), y(arith.SelectOp(take, cast, argmax))))
        branch = scf.IfOp(valid, [elem, i64], Region(yes), Region(Block([scf.YieldOp(current, argmax)])))
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
    top = Builder(ops=ops)
    zero, one, count = (top.ix(0), top.ix(1), top.ix(math.prod(gshape)))
    body = Block(arg_types=[IndexType(), rt])
    iv, accumulator = body.args
    b = Builder(body)
    coords = delinearize(b, iv, gshape)
    picked = b(tensor.ExtractOp(indices, coords, indices.type.element_type))
    linear = normalize_index(b, b(arith.IndexCastOp(picked, IndexType())), math.prod(shape[-ndim:]))
    destination = coords[:-ndim] + delinearize(b, linear, shape[-ndim:])
    incoming = b(tensor.ExtractOp(grad, coords, elem))
    prior = b(tensor.ExtractOp(accumulator, destination, elem))
    add = b(arith.AddfOp(prior, incoming) if isinstance(elem, AnyFloat) else arith.AddiOp(prior, incoming))
    b(scf.YieldOp(b(tensor.InsertOp(add, accumulator, destination))))
    loop = scf.ForOp(zero, count, one, [initial], Region(body))
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
    top = Builder(ops=ops)
    z, o, count, width = (top.ix(0), top.ix(1), top.ix(math.prod(ishape)), top.ix(gshape[-1]))
    body = Block(arg_types=[IndexType(), rt])
    iv, acc = body.args
    b = Builder(body)
    coords = delinearize(b, iv, ishape)
    picked = b(tensor.ExtractOp(indices, coords, indices.type.element_type))
    row = b(arith.IndexCastOp(picked, IndexType()))
    active = b(arith.CmpiOp(row, b.ix(padding), 'ne'))
    yes = Block()
    y = Builder(yes)
    limit = y.ix(rows)
    y(AssertOp(y(arith.AndIOp(y(arith.CmpiOp(row, z, 'sge')), y(arith.CmpiOp(row, limit, 'slt')))), 'embedding index out of bounds'))
    divisor = None
    if frequency and not (isinstance(acc_elem, IntegerType) and acc_elem.bitwidth == 1):
        counter = Block(arg_types=[IndexType(), IndexType()])
        j, n = counter.args
        c = Builder(counter)
        other = c(tensor.ExtractOp(indices, delinearize(c, j, ishape), indices.type.element_type))
        equal = c(arith.CmpiOp(other, picked, 'eq'))
        c(scf.YieldOp(c(arith.AddiOp(n, c(arith.SelectOp(equal, o, z))))))
        cast = y(arith.IndexCastOp(y(scf.ForOp(z, count, o, [z], Region(counter))), i64))
        if isinstance(acc_elem, IntegerType):
            divisor = y(arith.DivSIOp(y(arith.ConstantOp(IntegerAttr(1, i64), i64)), cast))
            # Counts have full integer width even when gradients are int8.
            if acc_elem.bitwidth < 64:
                divisor = y(arith.TruncIOp(divisor, acc_elem))
        else:
            divisor = y(arith.SIToFPOp(cast, acc_elem))
    columns = Block(arg_types=[IndexType(), rt])
    col, dst = columns.args
    c = Builder(columns)
    contribution = c(tensor.ExtractOp(grad, [*coords, col], acc_elem))
    prior = c(tensor.ExtractOp(dst, [row, col], acc_elem))
    if divisor is not None:
        contribution = c(arith.MuliOp(contribution, divisor) if isinstance(acc_elem, IntegerType) else arith.DivfOp(contribution, divisor))
    add = c((arith.OrIOp if isinstance(acc_elem, IntegerType) and acc_elem.bitwidth == 1 else arith.AddiOp if isinstance(acc_elem, IntegerType) else arith.AddfOp)(prior, contribution))
    c(scf.YieldOp(c(tensor.InsertOp(add, dst, [row, col]))))
    y(scf.YieldOp(y(scf.ForOp(z, width, o, [acc], Region(columns)))))
    branch = scf.IfOp(active, [rt], Region(yes), Region(Block([scf.YieldOp(acc)])))
    b(scf.YieldOp(b(branch)))
    loop = scf.ForOp(z, count, o, [initial], Region(body))
    ops.append(loop)
    casts, result = _cast_tensor(loop.results[0], [rows, gshape[-1]], elem)
    ops += casts
    return DecompResult(ops=ops, result=result, pattern_hint='embedding_backward')

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
    b=Builder(); result=None
    for axis in axes:
        cast=b(arith.IndexCastOp(b(tensor.DimOp(source,b.ix(axis))),i64))
        result=cast if result is None else b(arith.MuliOp(result,cast))
    if result is None:
        result=b(arith.ConstantOp(IntegerAttr(1,i64),i64))
    ops=b.ops
    return DecompResult(ops=ops,result=result,pattern_hint='local_scalar')
