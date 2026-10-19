"""A one-element reduction still executes its native seed and dtype policy."""
import itertools

import pytest
import torch

from tests.test_cpu_add_sub_bits import assert_bits

VALUES = [-0., 0., -1., 1., float('inf'), -float('inf'), float('nan'), -float('nan')]
DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64]


def reductions(x):
    outputs = [torch.sum(x, dim=-1), torch.prod(x, dim=-1), torch.mean(x, dim=-1)]
    if not x.is_complex():
        outputs.extend([torch.amax(x, dim=-1), torch.amin(x, dim=-1)])
    return tuple(outputs)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', [(), (1,), (1, 1)])
@pytest.mark.parametrize('value', VALUES)
@pytest.mark.parametrize('default', [False, True])
def test_single_value_reduction_storage_bits(tmp_path, dtype, shape, value, default):
    value = complex(value, -0.) if dtype.is_complex else value
    x = torch.full(shape, value, dtype=dtype)
    if not default:
        fn = reductions
    else:
        def fn(x):
            outputs = [torch.sum(x), torch.prod(x), torch.mean(x)]
            if not x.is_complex():
                outputs += [torch.amax(x), torch.amin(x)]
            return tuple(outputs)
    assert_bits(tmp_path, fn, (x,))


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('layout', ['last', 'transpose', 'padded'])
def test_single_element_fibers_cross_product(tmp_path, dtype, layout):
    words = [complex(a, b) for a, b in itertools.product(VALUES, repeat=2)] if dtype.is_complex else VALUES
    x = torch.tensor(words, dtype=dtype).reshape(-1, 1)
    if layout == 'padded':
        storage = torch.empty((len(words), 3), dtype=dtype)
        storage[:, 1:2] = x
        x = storage[:, 1:2]
    elif layout == 'transpose':
        x = x.T
    def fn(x):
        axis = 0 if layout == 'transpose' else -1
        outputs = [torch.sum(x, dim=axis, keepdim=True), torch.prod(x, dim=axis, keepdim=True),
                   torch.mean(x, dim=axis, keepdim=True)]
        if not x.is_complex():
            outputs += [torch.amax(x, dim=axis, keepdim=True), torch.amin(x, dim=axis, keepdim=True)]
        return tuple(outputs)
    assert_bits(tmp_path, fn, (x,))


@pytest.mark.parametrize('dtype', [torch.int8, torch.uint8, torch.int32, torch.int64, torch.bool])
@pytest.mark.parametrize('shape', [(), (1,), (2, 1)])
def test_single_element_integer_and_boolean_reductions(tmp_path, dtype, shape):
    x = torch.full(shape, 1, dtype=dtype)
    def fn(x):
        result = (torch.sum(x, dim=-1), torch.prod(x, dim=-1), torch.amax(x, dim=-1),
                  torch.amin(x, dim=-1), torch.sum(x, dim=-1, dtype=torch.float32),
                  torch.prod(x, dim=-1, dtype=torch.float32))
        if x.dtype == torch.bool:
            result += (torch.any(x, dim=-1), torch.all(x, dim=-1))
        return result
    assert_bits(tmp_path, fn, (x,))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_single_element_reductions_every_storage_word(tmp_path, dtype):
    x = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(dtype).reshape(-1, 1)
    assert_bits(tmp_path, reductions, (x,))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_single_element_nan_signaling_payloads(tmp_path, dtype):
    if dtype == torch.float32:
        words = [0x7f800001, 0xff800001, 0x7fc00017, 0xffc00027]
        integer = torch.uint32
    else:
        words = [0x7ff0000000000001, 0xfff0000000000001, 0x7ff8000000000017, 0xfff8000000000027]
        integer = torch.uint64
    x = torch.tensor(words, dtype=integer).view(dtype).reshape(-1, 1)
    assert_bits(tmp_path, reductions, (x,))


@pytest.mark.parametrize('shape,default', [(s, False) for s in
    [(), (1,), (1, 1), (7, 1), (8, 1), (9, 1), (17, 1)]] +
    [(s, True) for s in [(), (1,), (1, 1)]])
@pytest.mark.parametrize('offset', [0, 3, 6, 9])
def test_complex_reduction_nan_payloads_and_scalar_tails(tmp_path, shape, default, offset):
    # Distinct signed quiet/signaling payloads expose the div_ SIMD loop's
    # cross-component priority and its scalar tail's primary priority.
    words = [0x7fc00017, 0xffc00027, 0x7f800019, 0xff800029]
    pairs = [(a, b) for a, b in itertools.product(words, repeat=2) if a != b]
    count = 1
    for extent in shape:
        count *= extent
    x = torch.tensor([pairs[(i + offset) % len(pairs)] for i in range(count)], dtype=torch.uint32)
    x = x.view(torch.complex64).reshape(shape)
    def fn(x):
        if default:
            return x.sum(), x.prod(), x.mean()
        return x.sum(dim=-1), x.prod(dim=-1), x.mean(dim=-1)
    assert_bits(tmp_path, fn, (x,))


@pytest.mark.parametrize('dtype', DTYPES[:-1])
@pytest.mark.parametrize('shape', [(), (1,), (1, 1)])
def test_other_single_element_reductions(tmp_path, dtype, shape):
    x = torch.full(shape, -0., dtype=dtype)
    def fn(x):
        return (torch.var(x, dim=-1, correction=0), torch.std(x, dim=-1, correction=0),
                torch.logsumexp(x, dim=-1),
                torch.argmax(x, dim=-1), torch.argmin(x, dim=-1))
    assert_bits(tmp_path, fn, (x,))


@pytest.mark.parametrize('dtype', DTYPES[:-1])
def test_other_nonfinite_single_element_reductions(tmp_path, dtype):
    x = torch.tensor(VALUES, dtype=dtype)[:, None]
    def fn(x):
        return (torch.var(x, dim=-1, correction=0), torch.std(x, dim=-1, correction=0),
                torch.logsumexp(x, dim=-1), torch.argmax(x, dim=-1), torch.argmin(x, dim=-1))
    assert_bits(tmp_path, fn, (x,))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('rows', [7, 8, 9, 17, 33])
@pytest.mark.parametrize('width', [1, 3, 17])
def test_mean_native_writeback_vectors_and_tails(tmp_path, dtype, rows, width):
    generator = torch.Generator().manual_seed(rows + width)
    x = torch.randn((rows, width), generator=generator, dtype=dtype)
    x[::3, 0] = float('nan')
    x[1::3, 0] = -float('nan')
    def fn(x):
        return x.mean(dim=-1), x.mean(dim=-1, keepdim=True)
    assert_bits(tmp_path, fn, (x,))


def test_half_mean_parallel_writeback_tails(tmp_path):
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(3)
        x = torch.full((65539, 1), -float('nan'), dtype=torch.float16)
        assert_bits(tmp_path, lambda x: x.mean(dim=-1), (x,))
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16, torch.complex64])
def test_std_overloads_preserve_single_opmath_conversion(tmp_path, dtype):
    v = [complex(value, -0.) for value in VALUES] if dtype.is_complex else VALUES
    x = torch.tensor(v, dtype=dtype)[:, None]
    def fn(x):
        return (torch.ops.aten.std.dim(x, [-1], False),
                torch.ops.aten.std.correction(x, [-1], correction=0))
    assert_bits(tmp_path, fn, (x,))


@pytest.mark.parametrize('size', [1, 7, 9, 17])
def test_half_reductions_explicit_float_dtype(tmp_path, size):
    x = torch.tensor([VALUES[i % len(VALUES)] for i in range(size)], dtype=torch.float16)[:, None]
    def fn(x):
        return (torch.sum(x, dim=-1, dtype=torch.float32),
                torch.prod(x, dim=-1, dtype=torch.float32),
                torch.mean(x, dim=-1, dtype=torch.float32))
    assert_bits(tmp_path, fn, (x,))


@pytest.mark.parametrize('shape', [(9, 1), (3, 17)])
@pytest.mark.parametrize('axis', [0, 1])
def test_half_mean_expanded_input_promotion(tmp_path, shape, axis):
    base_shape = list(shape)
    base_shape[axis] = 1
    x = torch.full(base_shape, -float('nan'), dtype=torch.float16).expand(shape)
    assert_bits(tmp_path, lambda x: x.mean(dim=-1), (x,))


@pytest.mark.parametrize('backend', ['fx_importer', 'auto'])
def test_std_reaches_owned_reduction_through_both_capture_paths(backend, monkeypatch):
    import m2m
    import m2m.capture.torch_mlir_bridge as bridge
    monkeypatch.setattr(bridge, '_try_torch_mlir_import', lambda: None)
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.std(x, dim=-1, correction=0)
    x = torch.full((8, 1), float('inf'), dtype=torch.bfloat16)
    result = m2m.convert(Model(), (x,), backend=backend)
    assert result.ok and 'func.call' not in result.mlir_text
    targets = [str(node.target) for node in result.exported_program.graph.nodes if node.op == 'call_function']
    assert targets == ['aten.std.correction']


@pytest.mark.parametrize('dtype', [torch.int8, torch.uint8])
def test_integer_singleton_fibers_all_storage_words(tmp_path, dtype):
    x = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(dtype).reshape(-1, 1)
    def fn(x):
        return (torch.sum(x, dim=-1), torch.prod(x, dim=-1), torch.amax(x, dim=-1),
                torch.amin(x, dim=-1), torch.sum(x, dim=-1, dtype=torch.float32),
                torch.prod(x, dim=-1, dtype=torch.float32))
    assert_bits(tmp_path, fn, (x,))


@pytest.mark.parametrize('source_dtype', DTYPES[:-1])
@pytest.mark.parametrize('output_dtype', DTYPES[:-1])
@pytest.mark.parametrize('scalar', [False, True])
def test_single_element_explicit_dtype_rounding(tmp_path, source_dtype, output_dtype, scalar):
    v = [-0., 0., -2.**-135, 2.**-135, 1 + 2.**-11 - 2.**-40,
         1 + 2.**-11 + 2.**-40, float('nan'), -float('nan'), float('inf'), -float('inf')]
    x = torch.tensor(-2.**-135 if scalar else v, dtype=source_dtype)
    if not scalar:
        x = x[:, None]
    def fn(x):
        return (torch.sum(x, dim=-1, dtype=output_dtype), torch.prod(x, dim=-1, dtype=output_dtype),
                torch.mean(x, dim=-1, dtype=output_dtype))
    assert_bits(tmp_path, fn, (x,))


@pytest.mark.parametrize('dtype', [torch.int8, torch.uint8, torch.int32, torch.int64,
                                   torch.float16, torch.bfloat16, torch.float32, torch.float64,
                                   torch.complex64])
def test_single_element_boolean_output_truthiness(tmp_path, dtype):
    v = ([0, 2, 128, 255] if dtype == torch.uint8 else
         [complex(0., 0.), complex(0., 2.), complex(2., 0.), complex(float('nan'), 0.)]
         if dtype.is_complex else VALUES if dtype.is_floating_point else [0, 1, 2, -2])
    x = torch.tensor(v, dtype=dtype)[:, None]
    def fn(x):
        return (torch.sum(x, dim=-1, dtype=torch.bool), torch.prod(x, dim=-1, dtype=torch.bool))
    assert_bits(tmp_path, fn, (x,))
