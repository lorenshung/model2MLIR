"""Execute Core ATen lowerings against eager, including every auxiliary result."""
import itertools
import numpy as np
import pytest
import torch
from xdsl.dialects import arith, cf, math as mlir_math
from xdsl.dialects.builtin import ModuleOp, TensorType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.dialects.linalg.ops import GenericOp, IndexOp
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.cf import CfFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.linalg import LinalgFunctions
from xdsl.interpreters.math import MathFunctions
from xdsl.interpreters.scf import ScfFunctions
from xdsl.interpreters.tensor import TensorFunctions
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.utils.ptr import TypedPtr
from xdsl.interpreters.builtin import xtype_for_el_type
from xdsl.ir import Block, Region
from m2m.ir.core_aten import max_pool_backward, max_pool_with_indices
from m2m.ir.decompositions import _element_type_from_meta
from test_core_aten_review_regressions import _ReviewInterpreterExtras
from test_native_batch_norm_training import _BatchNormInterpreterExtras, _as_shaped_array

@register_impls
class _LoopExtras(InterpreterFunctions):
    indices = ()

    @impl(GenericOp)
    def generic(self, interpreter, op, args):
        n = len(op.inputs)
        outputs = [a.copy() for a in args[n:]]
        arrays = [*args[:n], *outputs]
        maps = [m.data for m in op.indexing_maps]
        previous = self.indices
        for coords in itertools.product(*(range(r) for r in op.get_static_loop_ranges())):
            self.indices = coords
            scalars = tuple((a.load(m.eval(coords, ())) for a, m in zip(arrays, maps, strict=True)))
            values = interpreter.run_ssacfg_region(op.body, scalars, 'generic')
            for a, m, v in zip(outputs, maps[n:], values, strict=True):
                a.store(m.eval(coords, ()), v)
        self.indices = previous
        return tuple(outputs)

    @impl(cf.AssertOp)
    def assertion(self, interpreter, op, args):
        assert args[0], op.msg.data
        return ()

    @impl(arith.MaxSIOp)
    def maxsi(self, interpreter, op, args):
        return (max(args),)

    @impl(arith.MinSIOp)
    def minsi(self, interpreter, op, args):
        return (min(args),)

    @impl(mlir_math.AbsFOp)
    def absf(self, interpreter, op, args):
        return (abs(args[0]),)

    @impl(arith.SIToFPOp)
    def sitofp(self, interpreter, op, args):
        return (float(args[0]),)

    @impl(arith.UIToFPOp)
    def uitofp(self, interpreter, op, args):
        width = op.input.type.width.data
        return (float(args[0] % (1 << width)),)

    @impl(arith.ExtFOp)
    def extf(self, interpreter, op, args):
        return args

    @impl(arith.ExtSIOp)
    def extsi(self, interpreter, op, args):
        return args

    @impl(arith.ExtUIOp)
    def extui(self, interpreter, op, args):
        return (args[0] % (1 << op.input.type.width.data),)

    @impl(arith.RemUIOp)
    def rem(self, interpreter, op, args):
        return (args[0] % args[1],)

    @impl(arith.DivUIOp)
    def div(self, interpreter, op, args):
        return (args[0] // args[1],)

    @impl(IndexOp)
    def index(self, interpreter, op, args):
        return (self.indices[op.dim.value.data],)

def execute(lowering, target, inputs, args, expected, kwargs=None):
    types = [TensorType(_element_type_from_meta({'val': x}), list(x.shape)) for x in inputs]
    block = Block(arg_types=types)
    meta = {'val': expected, '_aten_target': target, '_fx_args': args, '_fx_kwargs': kwargs or {}}
    result = lowering(list(block.args), meta, 'independent')
    results = result.results or [result.result]
    block.add_ops([*result.ops, ReturnOp(*results)])
    module = ModuleOp([FuncOp('run', (types, [r.type for r in results]), Region(block))])
    module.verify()
    interpreter = Interpreter(module)
    for impl in [ArithFunctions(), CfFunctions(), FuncFunctions(), LinalgFunctions(), MathFunctions(), ScfFunctions(), TensorFunctions(), _BatchNormInterpreterExtras()]:
        interpreter.register_implementations(impl)
    interpreter.register_implementations(_ReviewInterpreterExtras(), override=True)
    interpreter.register_implementations(_LoopExtras(), override=True)
    actual = interpreter.call_op('run', tuple((ShapedArray(TypedPtr.new((x.to(torch.int8) if x.dtype == torch.uint8 else x).flatten().tolist(), xtype=xtype_for_el_type(t.element_type, interpreter.index_bitwidth)), list(x.shape)) for x, t in zip(inputs, types, strict=True))))
    expected = list(expected) if isinstance(expected, (tuple, list)) else [expected]
    return [torch.tensor(x.data, dtype=y.dtype).reshape(x.shape) for x, y in zip(actual, expected, strict=True)]

@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('unbatched', [False, True])
@pytest.mark.parametrize('ceil', [False, True])
def test_pool_values_indices_and_backward(ndim, unbatched, ceil):
    shape = ([2] if unbatched else [1, 2]) + [4] * ndim
    x = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
    x.flatten()[3:5] = float('nan')
    x.flatten()[10:13] = float('-inf')
    target = f'aten.max_pool{ndim}d_with_indices.default'
    op = getattr(torch.ops.aten, f'max_pool{ndim}d_with_indices').default
    args = (None, [2] * ndim, [2] * ndim, [1] * ndim, [1] * ndim, ceil)
    expected = op(x, *args[1:])
    actual = execute(max_pool_with_indices, target, [x], args, expected)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, equal_nan=True)
    grad = torch.arange(expected[0].numel(), dtype=torch.float32).reshape(expected[0].shape)
    back = getattr(torch.ops.aten, f'max_pool{ndim}d_with_indices_backward').default
    oracle = back(grad, x, *args[1:], expected[1])
    actual, = execute(max_pool_backward, f'aten.max_pool{ndim}d_with_indices_backward.default', [grad, x, expected[1]], (None, None, *args[1:], None), oracle)
    torch.testing.assert_close(actual, oracle)

@pytest.mark.parametrize('dtype', [torch.int8, torch.int16, torch.int32, torch.int64, torch.float32])
def test_batched_contraction_uses_the_input_arithmetic(dtype):
    from m2m.ir.decompositions import decompose_bmm
    a = torch.arange(12, dtype=dtype).reshape(2, 2, 3) % 3
    b = torch.arange(18, dtype=dtype).reshape(2, 3, 3) % 4
    expected = torch.bmm(a, b)
    actual, = execute(decompose_bmm, 'aten.bmm.default', [a, b], (None, None), expected)
    torch.testing.assert_close(actual, expected)

@pytest.mark.parametrize('target', ['aten.ne.Tensor', 'aten.lt.Tensor', 'aten.ge.Tensor'])
def test_comparison_promotes_inputs_and_handles_nan(target):
    from m2m.ir.decompositions import DECOMPOSITION_TABLE
    a = torch.tensor([[0, 2]], dtype=torch.int32)
    b = torch.tensor([0.5, float('nan')], dtype=torch.float64)
    op = getattr(getattr(torch.ops.aten, target.split('.')[1]), 'Tensor')
    expected = op(a, b)
    actual, = execute(DECOMPOSITION_TABLE[target], target, [a, b], (a, b), expected)
    torch.testing.assert_close(actual, expected)

@pytest.mark.parametrize('target', ['aten.index.Tensor', 'aten.index_put.default'])
def test_negative_advanced_indices_and_bounds(target):
    from m2m.ir.decompositions import DECOMPOSITION_TABLE
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    indices = torch.tensor([-1, 0, -3])
    if target == 'aten.index.Tensor':
        inputs = [x, indices]
        args = (x, [indices])
        expected = x[indices]
    else:
        values = torch.full((3, 3), -2.0)
        inputs = [x, indices, values]
        args = (x, [indices], values, False)
        expected = torch.index_put(x, [indices], values)
    actual, = execute(DECOMPOSITION_TABLE[target], target, inputs, args, expected)
    torch.testing.assert_close(actual, expected)
    inputs[1] = torch.tensor([-5, 0, 1])
    with pytest.raises(AssertionError, match='ATen index out of bounds'):
        execute(DECOMPOSITION_TABLE[target], target, inputs, args, expected)

@pytest.mark.parametrize('frequency', [False, True])
@pytest.mark.parametrize('padding', [-1, 1])
def test_embedding_backward_counts_duplicates_and_skips_padding(frequency, padding):
    from m2m.ir.core_aten import embedding_backward
    indices = torch.tensor([[2, 1], [2, 0]], dtype=torch.int32)
    grad = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    expected = torch.ops.aten.embedding_dense_backward.default(grad, indices, 4, padding, frequency)
    actual, = execute(embedding_backward, 'aten.embedding_dense_backward.default', [grad, indices], (None, None, 4, padding, frequency), expected)
    torch.testing.assert_close(actual, expected)

@pytest.mark.parametrize('case', ['hardtanh', 'leaky_relu', 'isinf', 'reflection_pad2d', 'replication_pad2d', 'index_select'])
def test_upstream_composite_ir_executes_and_retains_source(case):
    import m2m
    from m2m.capture.torch_export import capture_frontend_artifact
    from m2m.ir.decompositions import DECOMPOSITION_TABLE
    x = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4) - 5
    if case in ('reflection_pad2d', 'replication_pad2d'):
        op = getattr(torch.ops.aten, case).default
        args = (x, [1, 1, 1, 1])
    elif case == 'hardtanh':
        op = torch.ops.aten.hardtanh.default
        args = (x, -1.5, 2.0)
    elif case == 'leaky_relu':
        op = torch.ops.aten.leaky_relu.default
        args = (x, 0.25)
    elif case == 'isinf':
        op = torch.ops.aten.isinf.default
        x.flatten()[0] = float('inf')
        args = (x,)
    else:
        op = torch.ops.aten.index_select.default
        args = (x, -1, torch.tensor([3, 1]))

    class Module(torch.nn.Module):

        def forward(self, *inputs):
            return op(inputs[0], *args[1:])
    result = m2m.convert(Module(), (x,), backend='fx_importer')
    assert result.ok and f'prov.aten = "{op}"' in result.mlir_text
    fake_args = args
    inputs = [v for v in args if isinstance(v, torch.Tensor)]
    expected = op(*args)
    actual, = execute(DECOMPOSITION_TABLE[str(op)], str(op), inputs, fake_args, expected)
    torch.testing.assert_close(actual, expected, equal_nan=True)

@pytest.mark.parametrize('index_dtype,offset_dtype', [(torch.int32, torch.int32), (torch.int32, torch.int64), (torch.int64, torch.int32)])
def test_embedding_bag_int32_auxiliary_abi_and_empty_bag(index_dtype, offset_dtype):
    from m2m.ir.decompositions import decompose_embedding_bag
    weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    indices = torch.tensor([2, 1, 2], dtype=index_dtype)
    offsets = torch.tensor([0, 2, 2], dtype=offset_dtype)
    args = (weight, indices, offsets, False, 0, False, None, False, -1)
    expected = torch.ops.aten._embedding_bag.default(*args)
    actual = execute(decompose_embedding_bag, 'aten._embedding_bag.default', [weight, indices, offsets], args, expected)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)

def test_functionalized_out_retains_source_and_mutation_result():
    import m2m

    class Out(torch.nn.Module):

        def forward(self, x, y, out):
            return torch.ops.aten.atan2.out(x, y, out=out)
    result = m2m.convert(Out(), (torch.ones(2), torch.ones(2), torch.zeros(2)), backend='fx_importer')
    assert result.ok and 'prov.aten = "aten.atan2.out"' in result.mlir_text
    assert '-> (tensor<2xf32>, tensor<2xf32>)' in result.mlir_text
    assert result.exported_program.graph_signature.output_specs[0].kind.name == 'USER_INPUT_MUTATION'

@pytest.mark.parametrize('source,dtype',[(torch.tensor([0.,2.,-3.,float('nan')]),torch.bool),(torch.tensor([False,True]),torch.float32)])
def test_truth_cast_and_boolean_float_cast(source,dtype):
    from m2m.ir.decompositions import DECOMPOSITION_TABLE
    expected=source.to(dtype)
    actual,=execute(DECOMPOSITION_TABLE['aten._to_copy.default'],'aten._to_copy.default',
                   [source],(source,),expected,{'dtype':dtype})
    torch.testing.assert_close(actual,expected)


def test_unsigned_comparison_promotes_without_sign_extension():
    from m2m.ir.decompositions import DECOMPOSITION_TABLE
    a=torch.tensor([127,128,255],dtype=torch.uint8)
    b=torch.tensor([126.,129.,254.],dtype=torch.float32)
    actual,=execute(DECOMPOSITION_TABLE['aten.gt.Tensor'],'aten.gt.Tensor',[a,b],(a,b),torch.gt(a,b))
    torch.testing.assert_close(actual,torch.gt(a,b))


def test_metadata_resize_preserves_values_and_explicit_input_update():
    import m2m
    class Resize(torch.nn.Module):
        def forward(self,x):return torch.ops.aten.resize_.default(x,[3,2])
    x=torch.arange(6.).reshape(2,3)
    result=m2m.convert(Resize(),(x,),backend='fx_importer')
    assert result.ok and 'func.call' not in result.mlir_text
    assert 'prov.aten = "aten.resize_.default"' in result.mlir_text
    assert 'prov.input_metadata_mutation = "x"' in result.mlir_text
    assert '-> (tensor<3x2xf32>, tensor<3x2xf32>)' in result.mlir_text
    assert x.shape==(2,3)
    result.module.verify()
    signature=result.exported_program.graph_signature
    assert signature.output_specs[0].kind.name=='USER_INPUT_MUTATION'


def test_unsigned_arithmetic_promotion_matches_eager():
    from m2m.ir.decompositions import DECOMPOSITION_TABLE
    a=torch.tensor([128,255],dtype=torch.uint8);b=torch.tensor([2.,3.])
    expected=a*b
    actual,=execute(DECOMPOSITION_TABLE['aten.mul.Tensor'],'aten.mul.Tensor',[a,b],(a,b),expected)
    torch.testing.assert_close(actual,expected)


@pytest.mark.parametrize('name,extra',[('sym_numel',()),('sym_size',(1,)),('sym_stride',(0,)),('sym_storage_offset',()),('sym_is_contiguous',())])
def test_static_metadata_query_retains_exact_provenance(name,extra):
    import m2m
    overload='int' if name in ('sym_size','sym_stride') else 'default'
    op=getattr(getattr(torch.ops.aten,name),overload)
    class Query(torch.nn.Module):
        def forward(self,x):return torch.ops.aten.scalar_tensor.default(op(x,*extra))
    x=torch.arange(12.).reshape(3,4).transpose(0,1)
    result=m2m.convert(Query(),(x,),backend='fx_importer')
    assert result.ok and 'func.call' not in result.mlir_text
    assert f'prov.aten = "{op}"' in result.mlir_text
    result.module.verify()
    # The query's scalar constant is derived from static layout metadata.
    expected=op(x,*extra)
    from m2m.ir.core_aten import static_query
    from xdsl.dialects.builtin import i1,i64
    query=static_query([],{'val':expected},'metadata')
    assert query.result.type==(i1 if isinstance(expected,bool) else i64)
    assert query.result.owner.value.value.data==int(expected)


def test_bfloat16_scalar_operands_use_splat_without_opaque_cast():
    import m2m
    class Add(torch.nn.Module):
        def forward(self,x):return x+0.5
    result=m2m.convert(Add(),(torch.ones(2,dtype=torch.bfloat16),),backend='fx_importer')
    assert result.ok and 'func.call' not in result.mlir_text
    result.module.verify()


def test_upstream_backward_preserves_none_tuple_slots():
    import m2m
    class Backward(torch.nn.Module):
        def forward(self,g,x,mean,rstd,w,b):
            return torch.ops.aten.native_layer_norm_backward.default(g,x,[3],mean,rstd,w,b,[False,True,False])[1]
    x=torch.arange(6.).reshape(2,3);w=torch.ones(3);b=torch.zeros(3)
    _,mean,rstd=torch.ops.aten.native_layer_norm.default(x,[3],w,b,1e-5)
    result=m2m.convert(Backward(),(torch.ones_like(x),x,mean,rstd,w,b),backend='fx_importer')
    assert result.ok and 'func.call' not in result.mlir_text
    assert '-> tensor<3xf32>' in result.mlir_text
    result.module.verify()


def test_failed_export_attempt_cannot_mutate_retry_or_caller_inputs(monkeypatch):
    from m2m.capture.torch_export import capture_model
    class Resize(torch.nn.Module):
        def forward(self,x):return torch.ops.aten.resize_.default(x,[3,2])
    real_export=torch.export.export
    attempts=[]
    def export(model,inputs,**kwargs):
        attempts.append(list(inputs[0].shape))
        if len(attempts)==1:
            inputs[0].resize_(3,2)
            raise AssertionError('sources must not be empty for symbol s1')
        assert kwargs['strict'] is True
        return real_export(model,inputs,**kwargs)
    monkeypatch.setattr(torch.export,'export',export)
    x=torch.arange(6.).reshape(2,3)
    exported=capture_model(Resize(),(x,))
    assert attempts==[[2,3],[2,3]]
    assert x.shape==(2,3)
    first=next(n for n in exported.graph.nodes if n.op=='placeholder')
    assert first.meta['val'].shape==(2,3)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bool])
@pytest.mark.parametrize('value_shape',[(),(1,),(3,)])
@pytest.mark.parametrize('accumulate',[False,True])
def test_boolean_mask_updates_broadcast_and_accumulate(dtype,value_shape,accumulate):
    from m2m.ir.decompositions import decompose_index_put
    x=torch.tensor([1,0,1,0],dtype=dtype)
    mask=torch.tensor([True,True,False,True])
    values=torch.ones(value_shape,dtype=dtype)
    expected=torch.ops.aten.index_put.default(x,[mask],values,accumulate)
    actual,=execute(decompose_index_put,'aten.index_put.default',[x,mask,values],(x,[mask],values,accumulate),expected)
    torch.testing.assert_close(actual,expected)


def test_boolean_integer_index_accumulation_uses_or():
    from m2m.ir.decompositions import decompose_index_put
    x=torch.tensor([True,False]); index=torch.tensor([0,0,1]); values=torch.tensor([True,True,True])
    expected=torch.ops.aten.index_put.default(x,[index],values,True)
    actual,=execute(decompose_index_put,'aten.index_put.default',[x,index,values],(x,[index],values,True),expected)
    torch.testing.assert_close(actual,expected)


def test_masked_scatter_flattens_source_and_checks_consumption():
    from m2m.ir.decompositions import decompose_index_put
    x=torch.zeros(3); mask=torch.tensor([True,False,True]); values=torch.tensor([[4.,5.],[6.,7.]])
    expected=torch.masked_scatter(x,mask,values)
    actual,=execute(decompose_index_put,'aten.masked_scatter.default',[x,mask,values],(x,mask,values),expected)
    torch.testing.assert_close(actual,expected)
    with pytest.raises(AssertionError,match='too short'):
        execute(decompose_index_put,'aten.masked_scatter.default',[x,mask,torch.ones(1)],(x,mask,torch.ones(1)),expected)
