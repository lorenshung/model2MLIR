"""Preserve eager auxiliary shapes and compare portable arithmetic across storage layouts."""

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
def test_bfloat16_embedding_sum_across_sample_layouts(tmp_path, sample_layout, sign):
    w = torch.tensor([[0.125, 0.25], [-0.5, -0.125]], dtype=torch.bfloat16) * sign
    i, o = torch.tensor([0, 1, 0, 1]), torch.tensor([0, 2])
    p = sample_weights(sample_layout, torch.bfloat16)

    def fn(w, i, o, p):
        return torch.ops.aten._embedding_bag.default(w, i, o, False, 0, False, p, False, -1)

    assert_native_matches_eager(tmp_path, fn, (w, i, o, p), rtol=1e-5, atol=1e-6)


def group_norm_case(id, x, affine=False, geometry=(2, 4, 4, 2)):
    return pytest.param(x, affine, geometry, id=id)


@pytest.mark.parametrize("x,affine,geometry", [
    # Mixed float affine parameters save float statistics.
    *(group_norm_case(f"mixed_affine_{dtype}", torch.linspace(-0.75, 1.25, 32).to(dtype).reshape(2, 4, 2, 2), True)
      for dtype in (torch.float16, torch.bfloat16)),
    # float64 retains the framework primitive lowering.
    group_norm_case("float64", torch.linspace(-1.0, 1.0, 32, dtype=torch.float64).reshape(2, 4, 2, 2)),
    group_norm_case("channels_last_3d", torch.linspace(-5.0, 7.0, 384).reshape(2, 8, 2, 3, 4).contiguous(
        memory_format=torch.channels_last_3d), geometry=(2, 8, 24, 4)),
    # A bounded offset with small variance.
    *(group_norm_case(f"small_variance_{memory_format}", torch.linspace(0.125, 2.125, 32).reshape(2, 4, 2, 2).contiguous(
        memory_format=memory_format)) for memory_format in (torch.contiguous_format, torch.channels_last)),
])
def test_group_norm_portable_layouts_and_statistics(tmp_path, x, affine, geometry):
    if affine:
        fn = lambda x, w, b: torch.ops.aten.native_group_norm.default(x, w, b, *geometry, 1e-5)
        inputs = (x, torch.ones(geometry[1]), torch.zeros(geometry[1]))
    else:
        fn, inputs = (lambda x: torch.ops.aten.native_group_norm.default(x, None, None, *geometry, 1e-5)), (x,)
    assert_native_matches_eager(tmp_path, fn, inputs, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["transpose", "expanded", "rows", "padded"])
def test_rsqrt_portable_opmath_across_layouts(tmp_path, dtype, layout):
    x = {
        "transpose": lambda: torch.full((2, 33), 0.25, dtype=dtype).t(),
        "expanded": lambda: torch.full((1,), 0.25, dtype=dtype).expand(64),
        "rows": lambda: torch.full((1, 33), 0.25, dtype=dtype).expand(2, -1),
        "padded": lambda: torch.full((128,), 0.25, dtype=dtype)[::2],
    }[layout]()
    assert_native_matches_eager(tmp_path, torch.rsqrt, (x,), rtol=1e-5, atol=1e-6)
