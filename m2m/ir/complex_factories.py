"""Pair ABI factories and scalar bridges, independent of complex FFT lowering."""
from types import SimpleNamespace

import torch
from xdsl.dialects import arith, cf, tensor
from xdsl.dialects.builtin import Float32Type, FloatAttr, IndexType, IntegerAttr, TensorType

from .complex_aten import _Scalar, _finish, _pack, _split
from .decompositions import (
    _cast_scalar_arg, _elementwise, _fx_arg,
    decompose_native_dropout,
)

TARGETS = {
    'aten.empty.memory_format', 'aten.empty_strided.default',
    'aten.full.default', 'aten.scalar_tensor.default',
    'aten.rand.default', 'aten.randn.default', 'aten.rand_like.default',
    'aten.randn_like.default', 'aten.native_dropout.default',
    'aten._local_scalar_dense.default',
}


def decomposition(node, target):
    if target not in TARGETS:
        return None
    values = node.meta.get('val')
    values = list(values) if isinstance(values, (list, tuple)) else [values]
    values += [n.meta.get('val') for n in node.all_input_nodes]
    dtypes = [v.dtype for v in values if hasattr(v, 'dtype') and v.dtype.is_complex]
    return dispatch if dtypes and all(d == torch.complex64 for d in dtypes) else None


def dispatch(operands, meta, node_name):
    target = meta['_aten_target']
    value = meta['val']
    elem = Float32Type()
    if target == 'aten._local_scalar_dense.default':
        ops, planes = _split(operands[0])
        indices = []
        for _ in planes[0].type.get_shape():
            zero = arith.ConstantOp(IntegerAttr(0, IndexType()))
            ops.append(zero); indices.append(zero.result)
        scalars = []
        for plane in planes:
            extract = tensor.ExtractOp(plane, indices, elem)
            ops.append(extract); scalars.append(extract.result)
        pair = tensor.FromElementsOp(*scalars, result_type=TensorType(elem, [2]))
        return _finish([*ops, pair], [pair.result], meta, 'local_scalar')
    if target == 'aten.native_dropout.default':
        return dropout(operands, meta, node_name)
    shape = list(value.shape)
    physical = TensorType(elem, [*shape, 2])
    if target.startswith(('aten.empty.', 'aten.empty_strided.')):
        # Undefined contents stay undefined, including both components.
        empty = tensor.EmptyOp([], physical)
        return _finish([empty], [empty.tensor], meta, 'empty')
    if target.startswith(('aten.rand.', 'aten.randn.', 'aten.rand_like.', 'aten.randn_like.')):
        from .torch_cpu_rng import normal_tensor, uniform_tensor
        normal = 'randn' in target
        draw = normal_tensor(physical, std=2.**-.5) if normal else uniform_tensor(physical)
        return _finish(draw[0], [draw[1]], meta, 'randn' if normal else 'rand')
    position = 1 if target == 'aten.full.default' else 0
    scalar = meta.get('_fx_ssa_args', {}).get(position)
    ops, planes = [], []
    if scalar is not None and isinstance(scalar.type, TensorType):
        # _local_scalar_dense carries one complex scalar as a two-element pair.
        for i in range(2):
            index = arith.ConstantOp(IntegerAttr(i, IndexType()))
            extract = tensor.ExtractOp(scalar, [index.result], elem)
            ops.extend([index, extract]); planes.append(extract.result)
    elif scalar is not None:
        cast, real = _cast_scalar_arg(scalar, elem)
        zero = arith.ConstantOp(FloatAttr(0., elem))
        ops.extend([*cast, zero]); planes = [real, zero.result]
    else:
        raw = complex(_fx_arg(meta, position, 0))
        for component in (raw.real, raw.imag):
            constant = arith.ConstantOp(FloatAttr(component, elem))
            ops.append(constant); planes.append(constant.result)
    if value.dtype != torch.complex64:
        # Eager's checked complex-to-real Scalar conversion accepts only zero imag.
        from .decompositions import _element_type_from_meta
        zero = arith.ConstantOp(FloatAttr(0., elem))
        destination = _element_type_from_meta(meta)
        from xdsl.dialects.builtin import IntegerType
        if isinstance(destination, IntegerType) and destination.width.data == 1:
            real_truth = arith.CmpfOp(planes[0], zero.result, 'une')
            imag_truth = arith.CmpfOp(planes[1], zero.result, 'une')
            truth = arith.OrIOp(real_truth.result, imag_truth.result)
            splat = tensor.SplatOp(truth.result, [], TensorType(destination, shape))
            return _finish([*ops, zero, real_truth, imag_truth, truth, splat], [splat.result], meta, 'fill')
        valid = arith.CmpfOp(planes[1], zero.result, 'oeq')
        check = cf.AssertOp(valid.result, 'complex Scalar has nonzero imaginary component')
        cast, real = _cast_scalar_arg(planes[0], destination)
        splat = tensor.SplatOp(real, [], TensorType(real.type, shape))
        return _finish([*ops, zero, valid, check, *cast, splat], [splat.result], meta, 'fill')
    splats = [tensor.SplatOp(p, [], TensorType(elem, shape)) for p in planes]
    packed = _pack(*(p.result for p in splats))
    return _finish([*ops, *splats, *packed[0]], [packed[1]], meta, 'fill')


def dropout(operands, meta, node_name):
    source = operands[0]
    logical = meta['val'][0]
    ops, planes = _split(source)
    # Obtain one Bernoulli mask for a logical element, never two RNG streams.
    adapted = dict(meta, val=(SimpleNamespace(shape=logical.shape, dtype=torch.float32), meta['val'][1]))
    real = decompose_native_dropout([planes[0]], adapted, node_name)
    ops += real.ops
    mask = real.results[1]
    if not bool(_fx_arg(meta, 2, False)):
        return _finish(ops, [source, mask], meta, 'native_dropout')
    p = float(_fx_arg(meta, 1, .5))
    scale = 0. if p == 1. else 1. / (1. - p)
    outputs = []
    for component in range(2):
        def body(args, out_elem, component=component):
            b = _Scalar()
            factor = b.cast(args[2])
            pair = b.cmul(args[:2], (factor, b.c(0)))
            # aten multiplies by mask, then by its real scalar scale.
            result = b.mul(pair[component], b.c(scale))
            return b.ops, result
        emitted = _elementwise([*planes, mask], TensorType(Float32Type(), list(logical.shape)), body)
        ops += emitted[0]; outputs.append(emitted[1])
    packed = _pack(*outputs)
    return _finish([*ops, *packed[0]], [packed[1], mask], meta, 'native_dropout')
