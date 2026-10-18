"""Native CPU storage boundaries, including SIMD tails and boolean overlap."""

import json

import m2m
import numpy as np
import pytest
import torch
from native_execution import run_native, assert_native_matches_eager


@pytest.mark.parametrize("shape,dims", [((2, 4), [0]), ((3, 5), [0]),
                                      ((2, 3, 4), [1]), ((2, 3, 4), [2, 0])])
@pytest.mark.parametrize("layout", ["dense", "padded", "expanded"])
@pytest.mark.parametrize("onesided", [False, True])
def test_fft_native_output_strides_and_dependent_view(shape, dims, layout, onesided):
    base = torch.arange(2 * np.prod(shape), dtype=torch.float32)
    x = base[::2].reshape(shape) if layout == "padded" else base[:np.prod(shape)].reshape(shape)
    if layout == "expanded":
        x = x[:1].expand(shape)
    class Model(torch.nn.Module):
        def forward(self, x):
            y = torch.ops.aten._fft_r2c.default(x, dims, 1, onesided)
            return y, y.transpose(0, 1)
    expected = Model()(x)
    converted = m2m.convert(Model(), (x,), backend="fx_importer", capture_trace=True)
    assert converted.ok, converted.diagnostics
    records = json.loads(converted.module.attributes["m2m.result_layout"].data)["results"]
    for record, value in zip(records, expected, strict=True):
        physical = torch.view_as_real(value)
        assert record["shape"] == list(physical.shape)
        assert record["stride"] == list(physical.stride())
        assert record["storage_offset"] == physical.storage_offset()
    assert records[1]["alias_results"] == [0]


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("padding,stride,dilation", [(0, 1, 1), (1, 1, 1), (1, 2, 2)])
def test_boolean_col2im_overlap_is_or(tmp_path, batched, padding, stride, dilation):
    output, kernel = [5, 5], [2, 2]
    blocks = (5 + 2 * padding - dilation - 1) // stride + 1
    generator = torch.Generator().manual_seed(102)
    x = torch.randint(0, 2, ((2,) if batched else ()) + (8, blocks * blocks), generator=generator).bool()
    def fn(x):
        return torch.ops.aten.col2im.default(x, output, kernel, [dilation] * 2, [padding] * 2, [stride] * 2)
    assert_native_matches_eager(tmp_path, fn, (x,), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("padded", [False, True])
def test_rsqrt_every_storage_code_simd_and_scalar(tmp_path, dtype, padded):
    x = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(dtype)
    if padded:
        storage = torch.empty(x.numel() * 2, dtype=dtype)
        storage[::2] = x
        x = storage[::2]
    assert_native_matches_eager(tmp_path, torch.rsqrt, (x,), rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["dense", "transpose", "row_broadcast", "scalar_broadcast"])
def test_half_rsqrt_simd_tail_per_iterator_row(tmp_path, layout):
    generator = torch.Generator().manual_seed(101)
    x = (torch.rand((2, 33), generator=generator) * 65504).half()
    x[:, -1] = 65504
    x = {"dense": lambda: x, "transpose": lambda: x.t(),
         "row_broadcast": lambda: x[:1].expand(2, -1),
         "scalar_broadcast": lambda: x[0, -1:].expand(67)}[layout]()
    assert_native_matches_eager(tmp_path, torch.rsqrt, (x,), rtol=0, atol=0)


@pytest.mark.parametrize("features", [3, 17])
@pytest.mark.parametrize("bag_size", [2, 7])
def test_bfloat16_embedding_fast_output_ties_and_random_values(tmp_path, features, bag_size):
    generator = torch.Generator().manual_seed(103)
    w = torch.randn(140, features, generator=generator).bfloat16()
    i = torch.randperm(140, generator=generator)
    o = torch.arange(0, 140, bag_size)
    def fn(w, i, o):
        return torch.ops.aten._embedding_bag.default(w, i, o, False, 0, False, None, False, -1)
    assert_native_matches_eager(tmp_path, fn, (w, i, o), rtol=0, atol=0)
