"""Signed/unsigned byte storage must never poison or sign-extend uint8 codes."""
import pytest
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401

from tests.native_execution import assert_native_matches_eager


@pytest.mark.parametrize("dtype,low,high,zp", [
    (torch.uint8, 0, 255, 0), (torch.uint8, 1, 254, 129),
    (torch.int8, -128, 127, 0), (torch.int8, -127, 127, -3),
])
@pytest.mark.parametrize("channel", [False, True])
def test_all_byte_codes_qdq_and_conversions(tmp_path, dtype, low, high, zp, channel):
    qd = torch.ops.quantized_decomposed
    codes = torch.arange(low, high + 1).to(dtype)
    x = torch.cat((codes.float() - zp, torch.tensor([-1e6, 1e6]))).reshape(1, -1)
    scales = torch.tensor([1.0])
    zeros = torch.tensor([zp], dtype=torch.int64)

    def fn(x):
        if channel:
            q = qd.quantize_per_channel.default(x, scales, zeros, 0, low, high, dtype)
            dq = qd.dequantize_per_channel.default(q, scales, zeros, 0, low, high, dtype)
        else:
            q = qd.quantize_per_tensor.default(x, 1.0, zp, low, high, dtype)
            dq = qd.dequantize_per_tensor.default(q, 1.0, zp, low, high, dtype)
        return q, dq, q.to(torch.int32), q.to(torch.float64)

    assert_native_matches_eager(tmp_path, fn, (x,), atol=0, rtol=0,
                                convert_options={"fully_standard": True})


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
def test_float_to_byte_wraps_instead_of_poison(tmp_path, dtype):
    x = torch.arange(-255, 256, dtype=torch.float32)
    assert_native_matches_eager(tmp_path, lambda x: x.to(dtype), (x,), atol=0, rtol=0)
