"""Shared CPU fast-path eligibility governs auxiliary shapes and BF16 rounding."""

import json

import m2m
import pytest
import torch
from m2m.ir.embedding_bag_cpu import fast_path_eligible
from native_execution import assert_native_matches_eager


def sample_weights(kind, dtype):
    return {
        "none": lambda: None,
        "contiguous": lambda: torch.ones(4, dtype=dtype),
        "strided": lambda: torch.ones(8, dtype=dtype)[::2],
        "expanded": lambda: torch.ones(1, dtype=dtype).expand(4),
    }[kind]()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("sample_layout", ["none", "contiguous", "strided", "expanded"])
@pytest.mark.parametrize("weight_layout", ["contiguous", "columns", "rows", "expanded"])
@pytest.mark.parametrize("padding", [-1, 1])
def test_fast_eligibility_matches_native_auxiliary(dtype, sample_layout, weight_layout, padding):
    w = torch.arange(60, dtype=dtype).reshape(10, 6)
    w = {
        "contiguous": lambda: w[:5],
        "columns": lambda: w[:5, ::2],
        "rows": lambda: w[::2],
        "expanded": lambda: w[:1].expand(5, -1),
    }[weight_layout]()
    i, o = torch.tensor([0, 1, 2, 1]), torch.tensor([0, 2])
    p = sample_weights(sample_layout, dtype)
    output = torch.ops.aten._embedding_bag.default(w, i, o, False, 0, False, p, False, padding)
    assert fast_path_eligible(w, p, padding_idx=padding) == (output[1].numel() == 0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided_indices", [False, True])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_indices_offsets_are_canonicalized_before_fast_dispatch(dtype, strided_indices, requires_grad):
    w = torch.arange(30, dtype=dtype).reshape(5, 6).requires_grad_(requires_grad)
    i, o = torch.tensor([0, 1, 2, 1]), torch.tensor([0, 2])
    if strided_indices:
        storage = torch.zeros(8, dtype=torch.int64)
        storage[::2] = i
        i = storage[::2]
        storage = torch.zeros(4, dtype=torch.int64)
        storage[::2] = o
        o = storage[::2]
    output = torch.ops.aten._embedding_bag.default(w, i, o, False, 0, False, None, False, -1)
    assert fast_path_eligible(w)
    assert output[1].shape == (0,)


@pytest.mark.parametrize("sample_layout", ["contiguous", "strided", "expanded"])
@pytest.mark.parametrize("trace", [False, True])
def test_bfloat16_sample_stride_preserves_all_captured_outputs(sample_layout, trace):
    class Model(torch.nn.Module):
        def forward(self, w, i, o, p):
            return torch.ops.aten._embedding_bag.default(w, i, o, False, 0, False, p, False, -1)

    inputs = (
        torch.arange(30, dtype=torch.bfloat16).reshape(5, 6),
        torch.tensor([0, 1, 2, 1]),
        torch.tensor([0, 2]),
        sample_weights(sample_layout, torch.bfloat16),
    )
    expected = Model()(*inputs)
    result = m2m.convert(Model(), inputs, backend="fx_importer", capture_trace=trace)
    assert result.ok, result.diagnostics
    layout = result.module.attributes.get("m2m.result_layout")
    if layout is not None:
        records = json.loads(layout.data)["results"]
        assert [v["shape"] for v in records] == [list(v.shape) for v in expected]
    else:
        from xdsl.dialects.func import FuncOp

        function = next(op for op in result.module.body.block.ops if isinstance(op, FuncOp) and not op.is_declaration)
        assert [list(v.get_shape()) for v in function.function_type.outputs] == [list(v.shape) for v in expected]


@pytest.mark.parametrize("sample_layout", ["contiguous", "strided", "expanded"])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_bfloat16_midpoint_rounding_follows_native_fast_dispatch(tmp_path, sample_layout, sign):
    w = torch.tensor([[0.10693359375, 0.25], [-0.322265625, -0.1787109375]], dtype=torch.bfloat16) * sign
    i, o = torch.tensor([0, 1, 0, 1]), torch.tensor([0, 2])
    p = sample_weights(sample_layout, torch.bfloat16)

    def fn(w, i, o, p):
        return torch.ops.aten._embedding_bag.default(w, i, o, False, 0, False, p, False, -1)

    assert_native_matches_eager(tmp_path, fn, (w, i, o, p), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_group_norm_mixed_affine_saves_float_statistics(tmp_path, dtype):
    x = torch.linspace(-0.75, 1.25, 32).to(dtype).reshape(2, 4, 2, 2)
    w, b = torch.ones(4), torch.zeros(4)

    def fn(x, w, b):
        return torch.ops.aten.native_group_norm.default(x, w, b, 2, 4, 4, 2, 1e-5)

    assert_native_matches_eager(tmp_path, fn, (x, w, b), rtol=1e-5, atol=1e-6)


def test_float64_group_norm_retains_framework_primitive_lowering(tmp_path):
    x = torch.linspace(-1.0, 1.0, 32, dtype=torch.float64).reshape(2, 4, 2, 2)

    def fn(x):
        return torch.ops.aten.native_group_norm.default(x, None, None, 2, 4, 4, 2, 1e-5)

    assert_native_matches_eager(tmp_path, fn, (x,), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["transpose", "expanded", "rows", "padded"])
def test_rsqrt_tensoriterator_memory_order(tmp_path, dtype, layout):
    x = {
        "transpose": lambda: torch.full((2, 33), 0.001, dtype=dtype).t(),
        "expanded": lambda: torch.full((1,), 0.001, dtype=dtype).expand(64),
        "rows": lambda: torch.full((1, 33), 0.001, dtype=dtype).expand(2, -1),
        "padded": lambda: torch.full((128,), 0.001, dtype=dtype)[::2],
    }[layout]()
    assert_native_matches_eager(tmp_path, torch.rsqrt, (x,), rtol=0, atol=0)


def test_group_norm_channels_last_3d_fused_square_accumulation(tmp_path):
    x = torch.linspace(-5.0, 7.0, 384).reshape(2, 8, 2, 3, 4).contiguous(memory_format=torch.channels_last_3d)

    def fn(x):
        return torch.ops.aten.native_group_norm.default(x, None, None, 2, 8, 24, 4, 1e-5)

    assert_native_matches_eager(tmp_path, fn, (x,), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("channels_last", [False, True])
def test_group_norm_large_offset_small_variance(tmp_path, channels_last):
    x = torch.linspace(1000.0, 1002.0, 32).reshape(2, 4, 2, 2)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)

    def fn(x):
        return torch.ops.aten.native_group_norm.default(x, None, None, 2, 4, 4, 2, 1e-5)

    assert_native_matches_eager(tmp_path, fn, (x,), rtol=1e-5, atol=1e-6)
