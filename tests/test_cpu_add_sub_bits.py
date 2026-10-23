"""CPU add/sub storage bits, including overload, Scalar type, SIMD and tails."""
import itertools

import numpy as np
import pytest
import torch

from tests.native_execution import padded, run_native

DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64]
ALPHAS = [0, 0.0, -0.0, 1, 2, -1., .3, float('inf'), -float('inf'), float('nan')]
VALUES = [-0., 0., -1., 1., -float('inf'), float('inf'), float('nan'), -float('nan')]


def assert_bits(tmp_path, fn, inputs):
    expected = fn(*inputs)
    if isinstance(expected, torch.Tensor):expected = (expected,)
    actual = run_native(tmp_path, fn, inputs, convert_options={'fully_standard': True})
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        if want.dtype==torch.bfloat16:want=want.view(torch.uint16)
        np.testing.assert_array_equal(np.ascontiguousarray(got).view(np.uint8),
                                      np.ascontiguousarray(want.numpy()).view(np.uint8),err_msg=f'output {index}')


def values(dtype):
    return [complex(a,b) for a,b in itertools.product(VALUES,repeat=2)] if dtype==torch.complex64 else VALUES


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('operation', ['add','sub'])
@pytest.mark.parametrize('layout', ['dense','padded','transpose','row_broadcast','scalar_broadcast',
                                   'short_tensor','short_lhs','short_rhs','short_strided'])
def test_tensor_signed_zero_infinity_nan_cross_product(tmp_path,dtype,operation,layout):
    v=values(dtype)
    pairs=list(itertools.product(v,repeat=2))
    x,y=torch.tensor(pairs,dtype=dtype).T.contiguous()
    if layout.startswith('short_'):
        def rows(t,scalar):
            if scalar:return t.reshape(-1,1)
            step=2 if layout=='short_strided' else 1
            storage=torch.empty((len(pairs),8*step),dtype=dtype)
            result=storage[:,:7*step:step]
            result.copy_(t[:,None])
            return result
        x,y=rows(x,layout=='short_lhs'),rows(y,layout=='short_rhs')
    else:x,y=x.reshape(-1,len(v)),y.reshape(-1,len(v))
    if layout=='padded':x,y=padded(x),padded(y)
    elif layout=='transpose':x,y=x.T,y.T
    elif layout=='row_broadcast':x,y=x[:1].expand_as(x),y
    elif layout=='scalar_broadcast':x,y=x,y[:1,:1].expand_as(y)
    op=getattr(torch.ops.aten,operation).Tensor
    def fn(x,y):return tuple(op(x,y,alpha=a) for a in ALPHAS)
    assert_bits(tmp_path,fn,(x,y))


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('operation', ['add','sub'])
@pytest.mark.parametrize('other', VALUES)
def test_scalar_signed_zero_infinity_nan(tmp_path,dtype,operation,other):
    x=torch.tensor(values(dtype),dtype=dtype)
    op=getattr(torch.ops.aten,operation).Scalar
    def fn(x):return tuple(op(x,other,alpha=a) for a in ALPHAS)
    assert_bits(tmp_path,fn,(x,))


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('operation', ['add','sub'])
@pytest.mark.parametrize('size',[1,3,4,7,8,17,33,67])
def test_native_finite_rounding_and_scalar_tails(tmp_path,dtype,operation,size):
    rng=torch.Generator().manual_seed(314+size)
    dt=torch.complex128 if dtype==torch.complex64 else torch.float64
    x=torch.randn(size,generator=rng,dtype=dt).to(dtype)
    y=torch.randn(size,generator=rng,dtype=dt).to(dtype)
    op=getattr(torch.ops.aten,operation)
    def fn(x,y):return (op.Tensor(x,y,alpha=.3),op.Scalar(x,.33333333333,alpha=.3))
    assert_bits(tmp_path,fn,(x,y))


@pytest.mark.parametrize('operation',['add','sub'])
@pytest.mark.parametrize('alpha',[complex(0.,-0.),complex(-0.,0.),complex(.3,-.7),complex(float('inf'),1.)])
def test_complex_coefficients_scalar_and_tensor(tmp_path,operation,alpha):
    v=values(torch.complex64)
    x,y=torch.tensor(list(itertools.product(v,v)),dtype=torch.complex64).T.contiguous()
    op=getattr(torch.ops.aten,operation)
    def fn(x,y):return op.Tensor(x,y,alpha=alpha),op.Scalar(x,complex(.5,-0.),alpha=alpha)
    assert_bits(tmp_path,fn,(x,y))


@pytest.mark.parametrize('operation',['add','sub'])
@pytest.mark.parametrize('other',[complex(a,b) for a,b in itertools.product([-0.,0.,float('inf'),float('nan')],repeat=2)])
def test_complex_scalar_components_and_tail(tmp_path,operation,other):
    x=torch.tensor([*values(torch.complex64),complex(-0.,-0.),complex(float('nan'),-float('nan'))],dtype=torch.complex64)
    op=getattr(torch.ops.aten,operation).Scalar
    def fn(x):return tuple(op(x,other,alpha=a) for a in ALPHAS)
    assert_bits(tmp_path,fn,(x,))


@pytest.mark.parametrize('alpha',[0,1,2,-1])
def test_boolean_add_is_overlap_or(tmp_path,alpha):
    x=torch.tensor([False,False,True,True]);y=torch.tensor([False,True,False,True])
    def fn(x,y):return torch.ops.aten.add.Tensor(x,y,alpha=alpha)
    assert_bits(tmp_path,fn,(x,y))


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('layout',['dense','padded'])
def test_reduced_add_sub_every_storage_word(tmp_path,dtype,layout):
    x=torch.arange(65536,dtype=torch.int32).to(torch.uint16).view(dtype)
    y=x.roll(2191)
    if layout=='padded':x,y=padded(x),padded(y)
    def fn(x,y):
        return (torch.add(x,y,alpha=.3),torch.sub(x,y,alpha=0),torch.sub(x,1.,alpha=0))
    assert_bits(tmp_path,fn,(x,y))


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16,torch.complex64])
@pytest.mark.parametrize('operation',['add','sub'])
@pytest.mark.parametrize('sign',[-1,1])
def test_integer_scalar_coefficients_keep_original_bits(tmp_path,dtype,operation,sign):
    coefficient=sign*((1<<60)+(1<<36)+1)
    x=torch.zeros(17,dtype=dtype);y=torch.ones_like(x)
    op=getattr(torch.ops.aten,operation)
    def fn(x,y):return op.Tensor(x,y,alpha=coefficient),op.Scalar(x,coefficient)
    assert_bits(tmp_path,fn,(x,y))


@pytest.mark.parametrize('dtype,midpoint',[(torch.float16,1+2.**-11),(torch.bfloat16,1+2.**-8)])
def test_reduced_scalar_coefficient_conversion_rounds_like_cpu(tmp_path,dtype,midpoint):
    coefficient=midpoint+2.**-40
    x=torch.zeros(67,dtype=dtype);y=torch.ones_like(x)
    def fn(x,y):return torch.add(x,y,alpha=coefficient),torch.add(x,coefficient),torch.sub(x,coefficient)
    assert_bits(tmp_path,fn,(x,y))


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('operation',['add','sub'])
@pytest.mark.parametrize('size',[3,7,8,15,23,24,25])
def test_scalar_broadcast_product_rounding_blocks(tmp_path,dtype,operation,size):
    rng=torch.Generator().manual_seed(1921+size)
    x=torch.randn(size,generator=rng,dtype=torch.float64).to(dtype)
    op=getattr(torch.ops.aten,operation).Scalar
    coefficients=[(.77777777777777,.123456789012345),
                  (-.33333333333,.3),(.123456789,-.777777777)]
    def fn(x):return tuple(op(x,other,alpha=alpha) for other,alpha in coefficients)
    assert_bits(tmp_path,fn,(x,))


@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('operation',['add','sub'])
@pytest.mark.parametrize('left_scalar',[False,True])
def test_zero_dim_tensor_operand(tmp_path,dtype,operation,left_scalar):
    x=torch.tensor(values(dtype)*3+values(dtype)[:3],dtype=dtype)
    y=torch.tensor(-0.,dtype=dtype)
    op=getattr(torch.ops.aten,operation).Tensor
    def fn(x,y):
        a,b=(y,x) if left_scalar else (x,y)
        return tuple(op(a,b,alpha=alpha) for alpha in ALPHAS)
    assert_bits(tmp_path,fn,(x,y))


@pytest.mark.parametrize('left,right',[(a,b) for a,b in itertools.permutations(DTYPES,2)
                                     if torch.promote_types(a,b)!=torch.complex128])
@pytest.mark.parametrize('operation',['add','sub'])
@pytest.mark.parametrize('layout',['dense','padded','expanded'])
def test_mixed_tensor_promotion_bits(tmp_path,left,right,operation,layout):
    x=torch.tensor(values(left)*3,dtype=left)
    y=torch.tensor(values(right)*3,dtype=right)
    if x.numel()!=y.numel():
        if x.numel()<y.numel():x=x.repeat(y.numel()//x.numel())
        else:y=y.repeat(x.numel()//y.numel())
    x,y=torch.cat([x,x[:3]]),torch.cat([y,y[:3]])
    if layout=='padded':x,y=padded(x),padded(y)
    elif layout=='expanded':x=x[:1].expand_as(x)
    op=getattr(torch.ops.aten,operation).Tensor
    def fn(x,y):return tuple(op(x,y,alpha=alpha) for alpha in ALPHAS)
    assert_bits(tmp_path,fn,(x,y))


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64,torch.complex64])
@pytest.mark.parametrize('pad',[False,True])
def test_nan_payload_selection_and_quieting(tmp_path,dtype,pad):
    width=64 if dtype==torch.float64 else 32
    words=([0x7ff0000000000001,0x7ff8123456789abc,0xfff0abcdef012345,0xfff80123456789ab]
           if width==64 else [0x7f800001,0x7fc12345,0xff812345,0xffc54321])
    v=np.array(words,dtype=f'uint{width}').view(f'float{width}')
    if dtype==torch.complex64:
        v=np.array(list(itertools.product(v,v)),dtype=np.float32).reshape(-1).view(np.complex64)
    x,y=(torch.from_numpy(t.copy()) for t in np.array(list(itertools.product(v,v)),dtype=v.dtype).T)
    if pad:x,y=padded(x),padded(y)
    def fn(x,y):
        return tuple(op(x,y,alpha=alpha) for op in (torch.add,torch.sub) for alpha in (0,0.,1.,-.3,float('nan')))
    assert_bits(tmp_path,fn,(x,y))


@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('operation',['add','sub'])
def test_two_broadcast_operands(tmp_path,dtype,operation):
    x=torch.tensor(-0.,dtype=dtype).expand(67)
    y=torch.tensor(float('inf'),dtype=dtype).expand(67)
    op=getattr(torch.ops.aten,operation).Tensor
    def fn(x,y):return tuple(op(x,y,alpha=alpha) for alpha in ALPHAS)
    assert_bits(tmp_path,fn,(x,y))


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32])
@pytest.mark.parametrize('scalar_dtype',[torch.float16,torch.bfloat16,torch.float32,torch.float64])
def test_weak_tensor_dtype_conversion(tmp_path,dtype,scalar_dtype):
    x=torch.tensor(VALUES*4+VALUES[:3],dtype=dtype)
    midpoint=1+2.**(-11 if dtype==torch.float16 else -8)
    y=torch.tensor(midpoint+2.**-40,dtype=scalar_dtype)
    def fn(x,y):return torch.add(x,y,alpha=.3),torch.sub(x,y,alpha=0),torch.sub(y,x,alpha=-.3)
    assert_bits(tmp_path,fn,(x,y))


@pytest.mark.parametrize('threads', [2, 3])
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('operation', ['add', 'sub'])
@pytest.mark.parametrize('layout', ['dense', 'transpose', 'broadcast', 'lhs_broadcast', 'strided'])
def test_parallel_iterator_slices_preserve_storage_bits(tmp_path, num_threads, threads, dtype, operation, layout):
    num_threads(threads)
    generator = torch.Generator().manual_seed(719)
    x = torch.randn((509, 129), generator=generator, dtype=dtype)
    y = torch.randn(x.shape, generator=generator, dtype=dtype)
    # Finite rounding probes and every signed-zero/Inf/NaN combination
    # repeat across chunk starts, partial rows, SIMD prefixes and tails.
    v = values(dtype)
    pairs = torch.tensor(list(itertools.product(v, repeat=2)), dtype=dtype)
    for t, column in [(x, 0), (y, 1)]:
        words = pairs[:, column].repeat((t.numel() + len(pairs) - 1) // len(pairs))[:t.numel()]
        t.flatten()[::3] = words[::3]
    if layout == 'transpose':
        x, y = x.T, y.T
    elif layout == 'broadcast':
        y = y[:, :1]
    elif layout == 'lhs_broadcast':
        x = x[:, :1]
    elif layout == 'strided':
        x, y = padded(x), padded(y)
    op = getattr(torch.ops.aten, operation)
    def fn(x, y):
        return (op.Tensor(x, y), op.Tensor(x, y, alpha=.3),
                op.Tensor(x, y, alpha=-0.), op.Tensor(x, y, alpha=float('nan')),
                op.Scalar(x, .33333333333, alpha=.3))
    assert_bits(tmp_path, fn, (x, y))


@pytest.mark.parametrize('size', [32, 33, 67])
@pytest.mark.parametrize('operation', ['add', 'sub'])
def test_half_broadcast_vector_nan_operand_order(tmp_path, num_threads, size, operation):
    num_threads(1)
    words = torch.tensor([0, 0x8000, 0x3c00, 0x7c00, 0xfc00,
                          0x7e11, 0xfe22, 0x7c01, 0xfc02], dtype=torch.uint16).view(torch.float16)
    x = words.repeat((size + len(words) - 1) // len(words))[:size]
    x = x.expand(len(words), -1).contiguous()
    y = words[:, None]
    op = getattr(torch.ops.aten, operation).Tensor
    def fn(x, y):
        return tuple(op(x, y, alpha=a) for a in [float('nan'), -float('nan'), .3])
    assert_bits(tmp_path, fn, (x, y))


@pytest.mark.parametrize('size', [32768, 32769, 65536, 65537])
@pytest.mark.parametrize('dtype', [torch.float16, torch.complex64])
def test_parallel_grain_boundaries(tmp_path, num_threads, size, dtype):
    num_threads(7)
    generator = torch.Generator().manual_seed(size)
    x = torch.randn(size, generator=generator, dtype=dtype)
    y = torch.randn(size, generator=generator, dtype=dtype)
    special = torch.tensor(values(dtype), dtype=dtype)
    repeated = special.repeat((size + len(special) - 1) // len(special))[:size]
    x[::3], y[::7] = repeated[::3], repeated[::7]
    def fn(x, y):
        return (torch.add(x, y), torch.add(x, y, alpha=float('nan')),
                torch.sub(x, y, alpha=0), torch.sub(x, y, alpha=-.3))
    assert_bits(tmp_path, fn, (x, y))


def test_whole_model_bridge_residual_add_is_not_opaque(monkeypatch, num_threads):
    import m2m
    from m2m.coverage import opaque_report
    import m2m.capture.torch_mlir_bridge as bridge
    class Residual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 1)
            self.bn = torch.nn.BatchNorm2d(3)
            self.relu = torch.nn.ReLU()
        def forward(self, x):
            return self.relu(self.bn(self.conv(x))) + x
    monkeypatch.setattr(bridge, '_try_torch_mlir_import', lambda: None)
    num_threads(2)
    result=m2m.convert(Residual().eval(), (torch.zeros(1, 3, 129, 129),), backend='auto')
    assert result.ok, result.diagnostics
    assert opaque_report(result.mlir_text) == {}
    assert 'math.fma' in result.mlir_text


@pytest.mark.parametrize('dtype,alpha',[(torch.float16,65505.),(torch.bfloat16,3.4e38),
                                       (torch.float32,1e100),(torch.complex64,1e100)])
def test_overflowing_native_alpha_is_cleanly_refused(dtype,alpha):
    import m2m
    from m2m.coverage import opaque_report
    class Add(torch.nn.Module):
        def forward(self,x,y):return torch.add(x,y,alpha=alpha)
    x=torch.ones(17,dtype=dtype)
    with pytest.raises(RuntimeError,match='overflow'):
        Add()(x,x)
    result=m2m.convert(Add(),(x,x),backend='fx_importer')
    assert opaque_report(result.mlir_text)=={'aten_add_Tensor':1}


@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('operation',['add','sub'])
def test_scalar_operand_conversion_can_overflow(tmp_path,dtype,operation):
    x=torch.tensor(values(dtype)+values(dtype)[:3],dtype=dtype)
    op=getattr(torch.ops.aten,operation).Scalar
    def fn(x):return op(x,1e100,alpha=.3)
    assert_bits(tmp_path,fn,(x,))


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32,torch.float64,torch.complex64])
@pytest.mark.parametrize('operation',['add','sub'])
def test_native_integral_scalar_word_boundaries(tmp_path,dtype,operation):
    coefficients=[-(1<<63),-(1<<63)+1,1<<63,(1<<63)+(1<<39)+1,(1<<64)-1]
    x=torch.zeros(17,dtype=dtype);y=torch.ones_like(x)
    op=getattr(torch.ops.aten,operation)
    def fn(x,y):
        return (tuple(op.Tensor(x,y,alpha=a) for a in coefficients)
                +tuple(op.Scalar(x,a) for a in coefficients))
    assert_bits(tmp_path,fn,(x,y))


@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('operation',['add','sub'])
@pytest.mark.parametrize('pad',[False,True])
@pytest.mark.parametrize('nan_self',[False,True])
def test_tensor_nan_payloads_at_scalar_and_vector_tails(tmp_path,dtype,operation,pad,nan_self):
    words={torch.float16:(16,0x7e35,0xfe49),torch.bfloat16:(16,0x7fc5,0xffc7),
           torch.float32:(32,0x7fc12345,0xffc54321),
           torch.float64:(64,0x7ff8123456789abc,0xfff8fedcba987654),
           torch.complex64:(32,0x7fc12345,0xffc54321)}
    width,a,b=words[dtype]
    if dtype==torch.complex64:
        x=torch.from_numpy(np.array([a,b]*17,dtype='uint32').view('complex64'))
        y=torch.from_numpy(np.array([b,a]*17,dtype='uint32').view('complex64'))
    else:
        integer={16:torch.uint16,32:torch.uint32,64:torch.uint64}[width]
        x=torch.tensor([a]*17,dtype=integer).view(dtype)
        y=torch.tensor([b]*17,dtype=integer).view(dtype)
    if not nan_self:x=torch.zeros_like(x)
    if pad:x,y=padded(x),padded(y)
    op=getattr(torch.ops.aten,operation).Tensor
    scalar_op=getattr(torch.ops.aten,operation).Scalar
    other=y[0].item()
    def fn(x,y,ys):
        lhs_other=x if nan_self else y
        return (tuple(op(x[:n],y[:n],alpha=alpha) for n in (1,3,4,7,17) for alpha in (0,float('nan')))
                +tuple(scalar_op(x[:n],other,alpha=alpha) for n in (1,3,4,7,17) for alpha in (0,float('nan')))
                +tuple(op(ys,lhs_other[:n],alpha=alpha) for n in (1,3,4,7,17) for alpha in (0,float('nan'))))
    assert_bits(tmp_path,fn,(x,y,y[0] if nan_self else torch.zeros((),dtype=dtype)))


@pytest.mark.parametrize('operation',['add','sub'])
@pytest.mark.parametrize('padded',[False,True])
@pytest.mark.parametrize('mode',['tensor','lhs','rhs'])
def test_complex_product_fusion_at_native_loop_boundaries(tmp_path,operation,padded,mode):
    rng=torch.Generator().manual_seed(191)
    def rand():
        x=torch.randn(34 if padded else 17,dtype=torch.complex64,generator=rng)
        return x[::2] if padded else x
    x,y=rand(),rand()
    other=y[0].item()
    op=getattr(torch.ops.aten,operation)
    def fn(x,y,xs):
        out=[]
        for n in (1,2,3,4,5,6,7,8,9,12,15,16,17):
            if mode=='tensor':z=op.Tensor(x[:n],y[:n],alpha=complex(.3,-.7))
            elif mode=='lhs':z=op.Tensor(xs,y[:n],alpha=complex(.3,-.7))
            else:z=op.Scalar(x[:n],other,alpha=complex(.3,-.7))
            out.append(z)
        return tuple(out)
    assert_bits(tmp_path,fn,(x,y,x[0]))


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('operation',['add','sub'])
@pytest.mark.parametrize('scalar_dtype',[torch.float64,torch.int64])
def test_runtime_scalar_operand_keeps_cpu_conversion_bridge(tmp_path,dtype,operation,scalar_dtype):
    midpoint=1+2.**(-11 if dtype==torch.float16 else -8)
    scalar=(midpoint+2.**-40 if scalar_dtype==torch.float64 else (1<<60)+(1<<52)+1)
    x=torch.zeros(67,dtype=dtype);z=torch.tensor(scalar,dtype=scalar_dtype)
    op=getattr(torch.ops.aten,operation).Scalar
    def fn(x,z):
        value=torch.ops.aten._local_scalar_dense.default(z)
        return op(x,value),op(x,value,alpha=0),op(x,value,alpha=.3)
    assert_bits(tmp_path,fn,(x,z))
