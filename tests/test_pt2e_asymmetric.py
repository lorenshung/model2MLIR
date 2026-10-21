"""Centered byte contractions against independent exact Fraction arithmetic."""
from fractions import Fraction
import itertools

import numpy as np
import pytest
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401

from m2m.capture.pt2e_integerize import integerize_pt2e
from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
from m2m.capture.pt2e_compat import allow_exported_model_train_eval
from m2m.ir.quant_attrs import fp32
from tests.native_execution import run_native
from tests.test_exact_quant_arithmetic import oracle_float32
from tests.test_quant_schema import _check_capture
import m2m


class Affine(torch.nn.Module):
    def __init__(self, kind, dx, dw, zx, zw, channel, quantized=True):
        super().__init__()
        self.kind, self.dx, self.dw, self.zx, self.channel, self.quantized = kind, dx, dw, zx, channel, quantized
        self.zw = zw
        self.register_buffer('scales', torch.tensor([.03, .125, .5], dtype=torch.float64))
        self.register_buffer('zeros', torch.tensor(zw if isinstance(zw, list) else [zw] * 3, dtype=torch.int64))
        self.register_buffer('bias', torch.tensor([.125, -.25, .3]))
    def forward(self, x, w):
        qd = torch.ops.quantized_decomposed
        def bounds(dt):return (-128,127) if dt==torch.int8 else (0,255)
        x = qd.dequantize_per_tensor.default(x,.03,self.zx,*bounds(self.dx),self.dx)
        axis = w.ndim-1 if self.kind in ('addmm','bmm','matmul') else 0
        w = (qd.dequantize_per_channel.default(w,self.scales,self.zeros,axis,*bounds(self.dw),self.dw)
             if self.channel else qd.dequantize_per_tensor.default(w,.125,self.zw,*bounds(self.dw),self.dw))
        if self.kind=='linear':y=torch.ops.aten.linear.default(x,w,self.bias)
        elif self.kind=='addmm':y=torch.addmm(self.bias,x,w,alpha=-1.3,beta=-.5)
        elif self.kind in ('bmm','matmul'):y=getattr(torch,self.kind)(x,w)
        elif self.kind=='conv':y=torch.conv2d(x,w,self.bias,padding=1,groups=3)
        else:y=torch.conv1d(x,w,self.bias,padding=1,groups=3)
        return qd.quantize_per_tensor.default(y,.0371,129,1,254,torch.uint8) if self.quantized else y


def oracle(model,x,w):
    xx=x.to(torch.int64)-model.zx
    axis=w.ndim-1 if model.kind in ('addmm','bmm','matmul') else 0
    zeros=model.zeros if model.channel else torch.tensor([model.zw])
    dims=[1]*w.ndim;dims[axis]=-1
    ww=w.to(torch.int64)-zeros.reshape(dims)
    if model.kind=='linear':acc=xx@ww.T
    elif model.kind in ('addmm','bmm','matmul'):acc=xx@ww
    else:
        rank=x.ndim-2;spatial=list(x.shape[2:]);kernel=list(w.shape[2:])
        out=[spatial[i]+2-kernel[i]+1 for i in range(rank)]
        acc=torch.zeros((x.shape[0],3,*out),dtype=torch.int64)
        for n,c,*p in itertools.product(range(x.shape[0]),range(3),*(range(s) for s in out)):
            total=0
            for tap in itertools.product(*(range(s) for s in kernel)):
                pos=[p[i]+tap[i]-1 for i in range(rank)]
                if all(0<=v<spatial[i] for i,v in enumerate(pos)):
                    total+=int(xx[(n,c,*pos)])*int(ww[(c,0,*tap)])
            acc[(n,c,*p)]=total
    result=np.empty(tuple(acc.shape),dtype=np.uint8 if model.quantized else np.float32)
    for ix in itertools.product(*(range(s) for s in acc.shape)):
        c=ix[1] if model.kind.startswith('conv') else ix[-1]
        sw=fp32(float(model.scales[c])) if model.channel else .125
        v=Fraction(int(acc[ix]))*Fraction(fp32(.03))*Fraction(sw)
        if model.kind=='addmm':v*=Fraction(-1.3)
        if model.kind not in ('bmm','matmul'):
            v+=Fraction(float(model.bias[c]))*Fraction(-.5 if model.kind=='addmm' else 1.)
        result[ix]=(min(254,max(1,round(v/Fraction(fp32(.0371)))+129)) if model.quantized else oracle_float32(v))
    return result


@pytest.mark.parametrize('kind',['linear','addmm','bmm','matmul','conv','conv1'])
@pytest.mark.parametrize('dx,dw,zx,zw,channel',[
    (torch.int8,torch.int8,-17,23,False),
    (torch.uint8,torch.uint8,255,0,False),
    (torch.uint8,torch.int8,129,[-128,3,127],True),
    (torch.int8,torch.uint8,127,[0,128,255],True),
])
@pytest.mark.parametrize('quantized',[False,True])
def test_centered_contractions_exact_fraction_native(tmp_path,kind,dx,dw,zx,zw,channel,quantized):
    rng=torch.Generator().manual_seed(310)
    sx,sw=({'linear':((2,5),(3,5)), 'addmm':((2,5),(5,3)),
            'bmm':((2,2,5),(2,5,3)), 'matmul':((2,1,2,5),(2,1,5,3)),
            'conv':((1,3,3,4),(3,1,2,2)), 'conv1':((1,3,5),(3,1,3))}[kind])
    def codes(shape,dt):return torch.randint(-128 if dt==torch.int8 else 0,128 if dt==torch.int8 else 256,shape,generator=rng,dtype=dt)
    x,w=codes(sx,dx),codes(sw,dw)
    model=Affine(kind,dx,dw,zx,zw,channel,quantized)
    want=oracle(model,x,w)
    captured=torch.export.export(model,(x,w)).module()
    ref=run_pt2e_integer_reference(captured,(x,w),expected_contractions=1)
    integer,receipt=integerize_pt2e(captured,(x,w))
    assert not receipt['refusals'] and receipt['quantized_contractions_integerized']==1
    q=receipt['qparams'][0]
    assert q['activation']['zero_point']==zx
    assert q['weight']['zero_point']==(model.zeros.tolist() if channel else zw)
    got=integer(x,w).numpy()
    typ=np.uint8 if quantized else np.uint32
    np.testing.assert_array_equal(got.view(typ),want.view(typ))
    np.testing.assert_array_equal(ref.output.numpy().view(typ),want.view(typ))
    allow_exported_model_train_eval(integer)
    converted=m2m.convert(integer,(x,w),backend='fx_importer',fully_standard=True)
    _check_capture(converted,set())
    native,=run_native(tmp_path,integer,(x,w),convert_options={'fully_standard':True})
    np.testing.assert_array_equal(native.view(typ),want.view(typ))


@pytest.mark.parametrize('dtype,zx,zw',[(torch.int8,0,0),(torch.int8,-128,127),(torch.uint8,255,0)])
@pytest.mark.parametrize('narrow',[False,True])
def test_all_byte_operand_pairs_and_negative_half_ties(tmp_path,dtype,zx,zw,narrow):
    low,high=(-128,127) if dtype==torch.int8 else (0,255)
    qmin,qmax=((-8,7) if dtype==torch.int8 else (3,252)) if narrow else (low,high)
    x=torch.arange(low,high+1,dtype=dtype).reshape(-1,1);w=x.T.contiguous()
    class Model(torch.nn.Module):
        def forward(self,x,w):
            qd=torch.ops.quantized_decomposed
            a=qd.dequantize_per_tensor.default(x,.5,zx,qmin,qmax,dtype)
            b=qd.dequantize_per_tensor.default(w,.25,zw,qmin,qmax,dtype)
            y=torch.addmm(torch.zeros(256),a,b,alpha=-1.,beta=0.)
            return qd.quantize_per_tensor.default(y,.25,3,-128,127,torch.int8)
    captured=torch.export.export(Model(),(x,w)).module()
    integer,receipt=integerize_pt2e(captured,(x,w))
    assert not receipt['refusals']
    want=np.array([[min(127,max(-128,round(Fraction(-(a-zx)*(b-zw),2))+3)) for b in range(low,high+1)]
                   for a in range(low,high+1)],dtype=np.int8)
    np.testing.assert_array_equal(integer(x,w).numpy(),want)
    ref=run_pt2e_integer_reference(torch.export.export(Model(),(x,w)).module(),(x,w),expected_contractions=1)
    np.testing.assert_array_equal(ref.output.numpy(),want)
    allow_exported_model_train_eval(integer)
    got,=run_native(tmp_path,integer,(x,w),convert_options={'fully_standard':True})
    np.testing.assert_array_equal(got,want)


@pytest.mark.parametrize('dtype,zp',[(torch.int8,-128),(torch.uint8,255)])
def test_asymmetric_backward_exact_integer_gradients(tmp_path,dtype,zp):
    rng=torch.Generator().manual_seed(315)
    low,high=(-128,127) if dtype==torch.int8 else (0,255)
    g=torch.randint(low,high+1,(1,4,3,3),dtype=dtype,generator=rng)
    x=torch.randint(low,high+1,(1,2,3,3),dtype=dtype,generator=rng)
    w=torch.randint(low,high+1,(4,1,3,3),dtype=dtype,generator=rng)
    class Backward(torch.nn.Module):
        def forward(self,g,x,w):
            dq=torch.ops.quantized_decomposed.dequantize_per_tensor.default
            g=dq(g,.5,zp,low,high,dtype)
            x=dq(x,.25,zp,low,high,dtype)
            w=dq(w,.125,zp,low,high,dtype)
            return torch.ops.aten.convolution_backward.default(g,x,w,[4],[1,1],[1,1],[1,1],False,[0,0],2,[True,True,True])
    # Small exact integer sums are exactly representable in binary64.
    gg,xx,ww=(v.to(torch.float64)-zp for v in (g,x,w))
    want=[torch.nn.grad.conv2d_input(xx.shape,ww,gg,padding=1,groups=2)*.0625,
          torch.nn.grad.conv2d_weight(xx,ww.shape,gg,padding=1,groups=2)*.125,
          gg.sum((0,2,3))*.5]
    want=[v.float().numpy() for v in want]
    captured=torch.export.export(Backward(),(g,x,w)).module()
    reference=run_pt2e_integer_reference(captured,(g,x,w),expected_contractions=1)
    integer,receipt=integerize_pt2e(captured,(g,x,w))
    assert not receipt['refusals'] and receipt['quantized_contractions_integerized']==1
    for got,expected in zip(integer(g,x,w),want):np.testing.assert_array_equal(got.numpy(),expected)
    for got,expected in zip(reference.output,want):np.testing.assert_array_equal(got.numpy(),expected)
    allow_exported_model_train_eval(integer)
    actual=run_native(tmp_path,integer,(g,x,w),convert_options={'fully_standard':True})
    for got,expected in zip(actual,want):np.testing.assert_array_equal(got,expected)


def test_centered_declaration_bound_refuses_int32_overflow():
    k=((1<<31)-1)//(255*255)+1
    class Overflow(torch.nn.Module):
        def forward(self,x,w):
            dq=torch.ops.quantized_decomposed.dequantize_per_tensor.default
            a=dq(x,.5,0,0,255,torch.uint8)
            b=dq(w,.25,0,0,255,torch.uint8)
            return torch.mm(a,b)
    inputs=(torch.ones((1,k),dtype=torch.uint8),torch.ones((k,1),dtype=torch.uint8))
    captured=torch.export.export(Overflow(),inputs).module()
    _,receipt=integerize_pt2e(captured,inputs)
    assert receipt['quantized_contractions_integerized']==0
    assert 'overflow' in receipt['refusals'][0]['reason']
    with pytest.raises(ValueError,match='overflow'):
        run_pt2e_integer_reference(captured,inputs)


@pytest.mark.parametrize('storage,zero_dtype',[(torch.uint8,torch.int8),(torch.uint8,torch.uint8),(torch.int8,torch.uint8)])
def test_channel_zero_point_storage_and_negative_axis(tmp_path,storage,zero_dtype):
    x=torch.tensor([[1,2,3,4,5]],dtype=torch.uint8)
    w=torch.arange(15,dtype=storage).reshape(5,3)
    model=Affine('addmm',torch.uint8,storage,3,[0,7,127],True)
    model.zeros=model.zeros.to(zero_dtype)
    want=oracle(model,x,w)
    captured=torch.export.export(model,(x,w)).module()
    for node in captured.graph.nodes:
        if node.target==torch.ops.quantized_decomposed.dequantize_per_channel.default:
            args=list(node.args);args[3]=-1;node.args=tuple(args)
    captured.recompile()
    ref=run_pt2e_integer_reference(captured,(x,w),expected_contractions=1)
    integer,receipt=integerize_pt2e(captured,(x,w))
    assert not receipt['refusals']
    assert receipt['qparams'][0]['weight']['axis']==1
    np.testing.assert_array_equal(integer(x,w).numpy(),want)
    np.testing.assert_array_equal(ref.output.numpy(),want)
    allow_exported_model_train_eval(integer)
    got,=run_native(tmp_path,integer,(x,w),convert_options={'fully_standard':True})
    np.testing.assert_array_equal(got,want)


@pytest.mark.parametrize('storage,invalid',[(torch.int8,-129),(torch.int8,128),(torch.uint8,-1),(torch.uint8,256)])
def test_invalid_channel_zero_points_refuse_cleanly(storage,invalid):
    model=Affine('linear',torch.uint8,storage,3,[0,3,7],True)
    x=torch.ones((2,5),dtype=torch.uint8);w=torch.ones((3,5),dtype=storage)
    captured=torch.export.export(model,(x,w)).module()
    zero_node=next(n for n in captured.graph.nodes if n.target==torch.ops.quantized_decomposed.dequantize_per_channel.default).args[2]
    getattr(captured,zero_node.target).fill_(invalid)
    _,receipt=integerize_pt2e(captured,(x,w))
    assert receipt['quantized_contractions_integerized']==0
    assert 'zero points' in receipt['refusals'][0]['reason']
    with pytest.raises(ValueError,match='zero points'):
        run_pt2e_integer_reference(captured,(x,w))
