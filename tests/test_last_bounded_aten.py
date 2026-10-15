"""Execute the remaining bounded families with independent eager references."""
import json
import numpy as np
import pytest
import torch
import m2m
from m2m.coverage import opaque_report
from native_execution import assert_native_matches_eager, run_native, _tools


@pytest.mark.parametrize('op', ['bitwise_and', 'bitwise_or', 'bitwise_xor'])
@pytest.mark.parametrize('scalar', [True, False, 3, -2])
def test_boolean_scalar_bitwise(tmp_path, op, scalar):
    fn = lambda x: getattr(getattr(torch.ops.aten, op), 'Scalar')(x, scalar)
    assert_native_matches_eager(tmp_path, fn, (torch.tensor([False, True]),), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.bool, torch.int8, torch.float32])
@pytest.mark.parametrize('value', [0., 2.5, float('nan')])
def test_fill_truth_conversion(tmp_path, dtype, value):
    if dtype == torch.int8 and np.isnan(value):
        return  # Eager rejects this conversion before capture.
    fn = lambda x: torch.ops.aten.fill.Scalar(x, value)
    assert_native_matches_eager(tmp_path, fn, (torch.ones(2, 3, dtype=dtype),), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.bool, torch.complex64])
@pytest.mark.parametrize('value', [0., 2.5, -3.])
def test_scalar_tensor_factory(tmp_path, dtype, value):
    assert_native_matches_eager(tmp_path, lambda: torch.ops.aten.scalar_tensor(value, dtype=dtype), (), rtol=0, atol=0)


@pytest.mark.parametrize('bounds', [(-.25, .75), (None, .75), (-.25, None)])
def test_boolean_clamp_promotion(tmp_path, bounds):
    assert_native_matches_eager(tmp_path, lambda x: torch.clamp(x, *bounds), (torch.tensor([True, False]),), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('mode', ['floor', 'trunc'])
@pytest.mark.parametrize('form', ['tensor', 'scalar'])
def test_rounded_division_ieee_and_opmath(tmp_path, dtype, mode, form):
    x = torch.tensor([-7., -.5, -0., 0., .5, 7., float('nan'), float('inf'), -float('inf')], dtype=dtype)
    if form == 'scalar':
        fn = lambda a: torch.ops.aten.div.Scalar_mode(a, -.3, rounding_mode=mode)
        inputs = (x,)
    else:
        y = torch.tensor([2., -.3, 2., -2., 0., -0., 2., float('inf'), -float('inf')], dtype=dtype)
        fn = lambda a, b: torch.ops.aten.div.Tensor_mode(a, b, rounding_mode=mode)
        inputs = x, y
    assert_native_matches_eager(tmp_path, fn, inputs, rtol=1e-5, atol=1e-6)


def test_floor_division_corrects_rounded_quotient(tmp_path):
    x = torch.tensor([1., -1., .3, -.3, 0., -0.], dtype=torch.float64)
    fn = lambda a: torch.div(a, .1, rounding_mode='floor')
    actual, = run_native(tmp_path, fn, (x,))
    expected = fn(x).numpy()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('form', ['zero_dim', 'runtime_item'])
@pytest.mark.parametrize('denominator', [.1, -.1, .3])
def test_half_rounded_division_keeps_float32_scalar(tmp_path, dtype, form, denominator):
    x = torch.tensor([1., -1., .3, -.3, 0., -0.], dtype=dtype)
    scalar = torch.tensor(denominator, dtype=torch.float32)
    def fn(a, b):
        if form == 'runtime_item':
            b = torch.ops.aten._local_scalar_dense.default(b)
        return torch.div(a, b, rounding_mode='floor')
    actual, = run_native(tmp_path, fn, (x, scalar))
    got = torch.from_numpy(actual)
    if dtype == torch.bfloat16:
        got = got.view(dtype)
    expected = fn(x, scalar)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert torch.equal(torch.signbit(got), torch.signbit(expected))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('affine', [False, True])
def test_half_layer_norm_all_statistics(tmp_path, dtype, affine):
    x = torch.linspace(-3, 3, 30, dtype=dtype).reshape(2, 3, 5)
    fn = lambda a: torch.ops.aten.native_layer_norm(a, [3, 5], torch.ones_like(a[0]) if affine else None,
                                                   torch.zeros_like(a[0]) if affine else None, .001)
    assert_native_matches_eager(tmp_path, fn, (x,), rtol=.01 if dtype == torch.bfloat16 else .001, atol=.001)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('size', [3, 15, 32, 33, 768])
def test_half_layer_norm_portable_bounded_and_nonfinite_rows(tmp_path, dtype, size):
    row = torch.tensor([1., 2., -2.], dtype=dtype).repeat((size + 2) // 3)[:size]
    special = torch.tensor([float('nan'), float('inf'), -float('inf')], dtype=dtype).repeat((size + 2) // 3)[:size]
    x = torch.stack([row, special])
    fn = lambda a: torch.ops.aten.native_layer_norm.default(a, [size], None, None, .001)
    assert_native_matches_eager(tmp_path, fn, (x,), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_half_layer_norm_mixed_affine_stats_stay_float32(tmp_path, dtype):
    x = torch.linspace(-3, 3, 10, dtype=dtype).reshape(2, 5)
    weight, bias = torch.linspace(.5, 1.5, 5), torch.linspace(-1., 1., 5)
    fn = lambda a, w, b: torch.ops.aten.native_layer_norm.default(a, [5], w, b, .001)
    assert_native_matches_eager(tmp_path, fn, (x, weight, bias), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('value', [2.5, 2.5-3j])
@pytest.mark.parametrize('shape', [(), (0,), (2, 1, 3)])
def test_complex_full(tmp_path, value, shape):
    assert_native_matches_eager(tmp_path, lambda: torch.full(shape, value, dtype=torch.complex64), (), rtol=0, atol=0)


@pytest.mark.parametrize('strided', [False, True])
def test_complex_empty_metadata_only(tmp_path, strided):
    fn = (lambda: torch.empty_strided((2, 3), (7, 2), dtype=torch.complex64)) if strided else (lambda: torch.empty((2, 3), dtype=torch.complex64))
    class Model(torch.nn.Module):
        def forward(self): return fn()
    result = m2m.convert(Model(), (), backend='fx_importer')
    assert result.ok and not opaque_report(result.mlir_text)
    assert 'tensor.empty' in result.mlir_text
    assert 'tensor.splat' not in result.mlir_text
    contract = json.loads(result.module.attributes['m2m.result_layout'].data)['results'][0]
    expected = torch.view_as_real(fn())
    assert contract['shape'] == list(expected.shape)
    assert contract['stride'] == list(expected.stride())
    assert contract['storage_offset'] == expected.storage_offset()
    assert contract['requires_grad'] is False
    actual, = run_native(tmp_path, fn, ())
    # The suite's metadata policy grants no numerical contents to empty tensors.
    assert actual.shape == (2, 3) and actual.dtype == np.complex64


@pytest.mark.parametrize('value', [0j, 2.5+0j, float('inf')+0j, float('nan')+0j])
def test_complex_item_scalar_bridge(tmp_path, value):
    fn = lambda a: torch.ops.aten.scalar_tensor.default(torch.ops.aten._local_scalar_dense.default(a))
    assert_native_matches_eager(tmp_path, fn, (torch.tensor(value, dtype=torch.complex64),), rtol=0, atol=0)


@pytest.mark.parametrize('value', [2j, 0j, complex(0., float('nan'))])
def test_complex_item_boolean_truth(tmp_path, value):
    fn = lambda a: torch.ops.aten.scalar_tensor.default(torch.ops.aten._local_scalar_dense.default(a), dtype=torch.bool)
    assert_native_matches_eager(tmp_path, fn, (torch.tensor(value, dtype=torch.complex64),), rtol=0, atol=0)


def test_real_item_to_complex_scalar_tensor(tmp_path):
    fn = lambda a: torch.ops.aten.scalar_tensor.default(torch.ops.aten._local_scalar_dense.default(a), dtype=torch.complex64)
    assert_native_matches_eager(tmp_path, fn, (torch.tensor(2.5),), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.complex64])
@pytest.mark.parametrize('size', [1, 8, 16, 17, 30, 32, 33])
def test_normal_scalar_vector_tail_stream(tmp_path, dtype, size):
    fn = lambda: (torch.randn(size, dtype=dtype), torch.randn(3), torch.rand(5))
    torch.manual_seed(0)
    expected = fn()
    actual = run_native(tmp_path, fn, ())
    for a, e in zip(actual, expected, strict=True):
        t = torch.from_numpy(a)
        if dtype == torch.bfloat16 and e.dtype == torch.bfloat16: t = t.view(dtype)
        torch.testing.assert_close(t, e, rtol=1e-5, atol=1e-6)


def test_complex_uniform_interleaved_stream(tmp_path):
    fn = lambda: (torch.rand(2, 3, dtype=torch.complex64), torch.rand(5))
    torch.manual_seed(0)
    expected = fn()
    for a, e in zip(run_native(tmp_path, fn, ()), expected, strict=True):
        torch.testing.assert_close(torch.from_numpy(a), e, rtol=0, atol=0)


@pytest.mark.parametrize('p', [0., .25, 1.])
@pytest.mark.parametrize('train', [False, True])
def test_complex_dropout_shared_mask_and_stream(tmp_path, p, train):
    x = torch.tensor([1+2j, -3+4j, complex(-0., -0.), complex(float('inf'), 2), complex(2, float('nan'))])
    fn = lambda a: (*torch.ops.aten.native_dropout.default(a, p, train),
                    torch.ops.aten.rand.default([torch.ops.aten.sym_numel.default(a)]))
    torch.manual_seed(0)
    expected = fn(x)
    for a, e in zip(run_native(tmp_path, fn, (x,)), expected, strict=True):
        torch.testing.assert_close(torch.from_numpy(a), e, equal_nan=True)


@pytest.mark.parametrize('p', [0., .25, 1.])
def test_complex_dropout_nan_components_and_zero_signs(tmp_path, p):
    values = [complex(float('inf'), 0.), complex(float('inf'), 0.),
              complex(-0., -0.), complex(0., float('inf')),
              complex(float('nan'), 2.), complex(2., float('nan')),
              complex(0., -0.), complex(-0., 0.), complex(-3., 4.)]
    x = torch.tensor(values, dtype=torch.complex64)
    fn = lambda a: (*torch.ops.aten.native_dropout.default(a, p, True),
                    torch.ops.aten.rand.default([torch.ops.aten.sym_numel.default(a)]))
    torch.manual_seed(0)
    expected = fn(x)
    actual = run_native(tmp_path, fn, (x,))
    for got, want in zip(actual, expected, strict=True):
        eager = want.numpy()
        pieces = [(got.real, eager.real), (got.imag, eager.imag)] if np.iscomplexobj(got) else [(got, eager)]
        for a, b in pieces:
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
            np.testing.assert_allclose(a, b, rtol=0, atol=0, equal_nan=True)
            zeros = b == 0
            np.testing.assert_array_equal(np.signbit(a[zeros]), np.signbit(b[zeros]))


@pytest.mark.parametrize('structure', ['direct', 'tuple', 'dict'])
@pytest.mark.parametrize('train', [False, True, None])
@pytest.mark.parametrize('capture_trace', [False, True])
def test_complex_dropout_direct_tuple_preserves_all_output_leaves(tmp_path, structure, train, capture_trace):
    x = torch.tensor([[1+2j, complex(-0., -0.), -3+4j],
                      [complex(float('inf'), 0.), complex(0., float('inf')), 5-6j]])
    def fn(a):
        dropped = torch.ops.aten.native_dropout.default(a, .25, train)
        if structure == 'direct':
            return dropped
        following = torch.ops.aten.rand.default([torch.ops.aten.sym_numel.default(a)])
        return (dropped, following) if structure == 'tuple' else {'dropout': dropped, 'following': following}
    class Model(torch.nn.Module):
        def forward(self, a): return fn(a)
    converted = m2m.convert(Model(), (x,), backend='fx_importer', capture_trace=capture_trace)
    assert converted.ok, converted.diagnostics
    assert not opaque_report(converted.mlir_text)
    if capture_trace:
        assert converted.capture_trace['status'] == 'complete', converted.capture_trace['blockers']
        assert all(op['role'] != 'unresolved' for op in converted.capture_trace['mlir']['operations'])
    torch.manual_seed(0)
    expected = torch.utils._pytree.tree_leaves(fn(x))
    assert len(converted.exported_program.graph_signature.output_specs) == len(expected)
    entry = next(op for op in converted.module.body.block.ops
                 if op.name == 'func.func' and not op.is_declaration)
    assert len(entry.function_type.outputs) == len(expected)
    actual = run_native(tmp_path, fn, (x,), capture_trace=capture_trace)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        eager = want.numpy()
        pieces = [(got.real, eager.real), (got.imag, eager.imag)] if np.iscomplexobj(got) else [(got, eager)]
        for a, b in pieces:
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
            np.testing.assert_allclose(a, b, rtol=0, atol=0, equal_nan=True)
            zeros = b == 0
            np.testing.assert_array_equal(np.signbit(a[zeros]), np.signbit(b[zeros]))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('size', [15, 16, 17, 32, 33])
def test_noncontiguous_normal_like_scalar_storage_order_and_cache(tmp_path, dtype, size):
    x = torch.empty(3, size, dtype=dtype).t()
    fn = lambda a: (torch.randn(1), torch.randn_like(a), torch.randn(3), torch.rand(5))
    torch.manual_seed(0)
    expected = fn(x)
    for actual, eager in zip(run_native(tmp_path, fn, (x,)), expected, strict=True):
        got = torch.from_numpy(actual)
        if eager.dtype == torch.bfloat16:
            got = got.view(torch.bfloat16)
        torch.testing.assert_close(got, eager, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('normal', [False, True])
def test_channels_last_random_like_storage_order_and_stream(tmp_path, normal):
    x = torch.empty(2, 3, 2, 3).to(memory_format=torch.channels_last)
    fn = lambda a: ((torch.randn_like(a) if normal else torch.rand_like(a)), torch.randn(3), torch.rand(5))
    torch.manual_seed(0)
    expected = fn(x)
    for actual, eager in zip(run_native(tmp_path, fn, (x,)), expected, strict=True):
        torch.testing.assert_close(torch.from_numpy(actual), eager, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('normal', [False, True])
def test_random_like_zero_stride_singleton_has_no_zero_division(tmp_path, normal):
    from xdsl.dialects.arith import ConstantOp, DivUIOp, RemUIOp
    from xdsl.dialects.builtin import TensorType, f32
    from m2m.ir.torch_cpu_rng import normal_tensor, storage_strides, uniform_tensor
    x = torch.empty_strided((2, 1, 9), (1, 0, 2))
    strides = storage_strides(x)
    assert strides == [1, 0, 2]
    # FakeTensor may normalize a singleton stride during ordinary conversion.
    # Exercise the emitter's explicit stride contract so the structural check
    # detects the undefined division even when that high-level path hides it.
    draw = (normal_tensor if normal else uniform_tensor)(TensorType(f32, [2, 1, 9]), strides=strides)
    for operation in (nested for op in draw[0] for nested in op.walk()):
        if isinstance(operation, (DivUIOp, RemUIOp)):
            divisor = operation.operands[1].owner
            if isinstance(divisor, ConstantOp):
                assert divisor.value.value.data != 0
    fn = lambda a: ((torch.randn_like(a) if normal else torch.rand_like(a)), torch.randn(3), torch.rand(5))
    class Model(torch.nn.Module):
        def forward(self, a): return fn(a)
    converted = m2m.convert(Model(), (x,), backend='fx_importer')
    assert converted.ok, converted.diagnostics
    torch.manual_seed(0)
    expected = fn(x)
    assert expected[0].stride() == (1, 0, 2)
    for actual, eager in zip(run_native(tmp_path, fn, (x,)), expected, strict=True):
        torch.testing.assert_close(torch.from_numpy(actual), eager, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
@pytest.mark.parametrize('train', [False, True, None])
def test_empty_dropout_keeps_dtype_alias_and_rng_stream(tmp_path, dtype, train):
    x = torch.empty(2, 0, 3, dtype=dtype)
    fn = lambda a: (*torch.ops.aten.native_dropout.default(a, .25, train),
                    torch.ops.aten.rand.default([torch.ops.aten.sym_size.int(a, 2)]))
    torch.manual_seed(0)
    expected = fn(x)
    assert expected[1].dtype == dtype
    for actual, eager in zip(run_native(tmp_path, fn, (x,)), expected, strict=True):
        torch.testing.assert_close(torch.from_numpy(actual), eager, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
@pytest.mark.parametrize('train', [True, False, None])
@pytest.mark.parametrize('layout', ['transposed', 'channels_last'])
def test_dropout_storage_order_and_optional_training(tmp_path, dtype, train, layout):
    x = torch.arange(6.).reshape(3, 2).t() if layout == 'transposed' else torch.arange(24.).reshape(1, 2, 3, 4).to(memory_format=torch.channels_last)
    x = x.to(dtype)
    fn = lambda a: (*torch.ops.aten.native_dropout.default(a, .25, train),
                    torch.ops.aten.rand.default([torch.ops.aten.sym_numel.default(a)]))
    torch.manual_seed(0)
    expected = fn(x)
    for actual, eager in zip(run_native(tmp_path, fn, (x,)), expected, strict=True):
        torch.testing.assert_close(torch.from_numpy(actual), eager, rtol=0, atol=0)


def test_noncontiguous_complex_normal_like_stays_visible():
    class Model(torch.nn.Module):
        def forward(self, x): return torch.randn_like(x)
    result = m2m.convert(Model(), (torch.ones(3, 8, dtype=torch.complex64).t(),), backend='fx_importer')
    assert result.ok and opaque_report(result.mlir_text)


def test_resize_intermediate_backing_storage_refuses():
    class Model(torch.nn.Module):
        def forward(self, x):
            x.resize_([1])
            return x.resize_([6])
    x = torch.arange(24.).as_strided((2, 3), (6, 2))
    result = m2m.convert(Model(), (x,), backend='fx_importer', capture_trace=True)
    assert not result.ok or opaque_report(result.mlir_text)


def test_resize_complex128_stays_visible():
    class Model(torch.nn.Module):
        def forward(self, x): return x.resize_([2, 8])
    result = m2m.convert(Model(), (torch.ones(4, 4, dtype=torch.complex128).t(),), backend='fx_importer')
    assert not result.ok or opaque_report(result.mlir_text)


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
@pytest.mark.parametrize('layout', ['holes', 'offset', 'zero_stride'])
def test_resize_reads_preserved_storage(tmp_path, dtype, layout):
    base = torch.arange(24, dtype=torch.float32).to(dtype)
    if dtype.is_complex: base += 1j * base
    x = base.as_strided((2, 3), {'holes': (6, 2), 'offset': (3, 1), 'zero_stride': (0, 1)}[layout], 1 if layout == 'offset' else 0)
    assert_native_matches_eager(tmp_path, lambda a: a.resize_([3, 2]), (x,), rtol=0, atol=0)


def test_resize_growth_initializes_only_existing_storage(tmp_path):
    x = torch.arange(3.)
    fn = lambda a: a.resize_([8])
    actual, = run_native(tmp_path, fn, (x,))
    np.testing.assert_array_equal(actual[:3], x.numpy())
    assert actual.shape == (8,)


def test_configured_broken_native_tools_fail(monkeypatch):
    monkeypatch.setenv('MODEL2MLIR_MLIR_TRANSLATE', '/missing/mlir-translate')
    monkeypatch.setenv('MODEL2MLIR_CLANG', '/missing/clang')
    with pytest.raises(pytest.fail.Exception): _tools()
