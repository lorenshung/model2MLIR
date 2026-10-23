"""Native real FFT full-spectrum and unchanged half-spectrum controls."""

import json

import m2m
import pytest
import torch
from native_execution import run_native


@pytest.mark.parametrize("dims", [[0], [1], [0, 1], [1, 0]])
@pytest.mark.parametrize("normalization", [0, 1, 2])
@pytest.mark.parametrize("onesided", [False, True])
def test_real_fft_native_modes(tmp_path, dims, normalization, onesided):
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 8 - 0.75
    fn = lambda x: torch.view_as_real(torch.ops.aten._fft_r2c.default(x, dims, normalization, onesided))
    actual = torch.from_numpy(run_native(tmp_path, fn, (x,), capture_trace=True)[0])
    torch.testing.assert_close(actual, fn(x), atol=1e-6, rtol=1e-5, equal_nan=True)


@pytest.mark.parametrize("dims", [[0], [1], [0, 1], [1, 0]])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, "all_negative_zero"])
def test_real_fft_conjugate_nonfinite_and_zero(tmp_path, dims, value):
    x = torch.full((4, 4), -0.0) if value == "all_negative_zero" else torch.zeros(4, 4)
    if value != "all_negative_zero":
        x[1, 1] = value
    fn = lambda x: torch.view_as_real(torch.ops.aten._fft_r2c.default(x, dims, 0, False))
    actual = torch.from_numpy(run_native(tmp_path, fn, (x,), capture_trace=True)[0])
    expected = fn(x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5, equal_nan=True)
    zero = (expected == 0) & (actual == 0)
    assert torch.equal(actual.signbit()[zero], expected.signbit()[zero])


@pytest.mark.parametrize("shape,dims", [((3, 5), [1]), ((3, 5), [0, 1]), ((2, 3, 4), [2, 0])])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_fft_odd_and_batched_dimensions(tmp_path, shape, dims, dtype):
    x = torch.arange(torch.tensor(shape).prod().item(), dtype=dtype).reshape(shape) / 16 - 0.75
    fn = lambda x: torch.view_as_real(torch.ops.aten._fft_r2c.default(x, dims, 1, False))
    actual = torch.from_numpy(run_native(tmp_path, fn, (x,), capture_trace=True)[0])
    torch.testing.assert_close(actual, fn(x), atol=1e-6, rtol=1e-5, equal_nan=True)


@pytest.mark.parametrize("dims", [[0], [1], [0, 1], [1, 0]])
@pytest.mark.parametrize("layout", ["offset", "padded", "expanded"])
def test_real_fft_capture_complex_layout(dims, layout):
    base = torch.arange(40, dtype=torch.float32)
    x = {
        "offset": lambda: base.as_strided((4, 4), (4, 1), 3),
        "padded": lambda: base.as_strided((4, 4), (8, 2), 3),
        "expanded": lambda: base.as_strided((4, 4), (0, 1), 3),
    }[layout]()

    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten._fft_r2c.default(x, dims, 0, False)

    result = m2m.convert(Model(), (x,), backend="fx_importer", capture_trace=True)
    assert result.ok, result.diagnostics
    assert "func.call" not in result.mlir_text
    expected = torch.view_as_real(Model()(x))
    record = json.loads(result.module.attributes["m2m.result_layout"].data)["results"][0]
    assert record["shape"] == list(expected.shape)
    assert record["stride"] == list(expected.stride())
    assert record["storage_offset"] == expected.storage_offset()
    assert record["alias_inputs"] == []


@pytest.mark.parametrize("dims", [[-1], [-2], [-1, -2], [-2, -1]])
def test_public_real_fft_negative_axes(tmp_path, dims):
    # Public FFT normalizes negative axes before the ATen CPU call. Passing
    # negative axes directly to the low-level CPU kernel is not supported.
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 8 - 0.75
    fn = lambda x: torch.view_as_real(torch.fft.rfftn(x, dim=dims, norm="ortho"))
    actual = torch.from_numpy(run_native(tmp_path, fn, (x,), capture_trace=True)[0])
    torch.testing.assert_close(actual, fn(x), atol=1e-6, rtol=1e-5, equal_nan=True)
