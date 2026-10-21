"""Signed/unsigned byte storage must never poison or sign-extend uint8 codes."""
import numpy as np
import pytest
import torch

from tests.native_execution import assert_native_matches_eager, run_native


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
    x = torch.cat((torch.arange(-255, 256, dtype=torch.float64),
                   torch.tensor([float("nan"), float("inf"), -float("inf"), 1e300, -1e300,
                                 2147483649., -2147483649., 0.5, -0.5], dtype=torch.float64)))
    assert_native_matches_eager(tmp_path, lambda x: x.to(dtype), (x,), atol=0, rtol=0)


@pytest.mark.parametrize("zp_dtype,zps", [(torch.uint8, [129, 255]), (torch.int8, [-127, 3])])
def test_byte_zero_points_retain_their_signedness(tmp_path, zp_dtype, zps, record_property):
    qd = torch.ops.quantized_decomposed
    scales = torch.tensor([1., 0.5])
    zeros = torch.tensor(zps, dtype=zp_dtype)
    x = torch.arange(-256, 257, dtype=torch.float32)[None, :].expand(2, -1).contiguous()
    # Signed zero points below zero still define a mathematical affine map.
    # The standard expander requires a storage-representable zero point, so
    # signed zero points here use signed output storage instead.
    low, high, storage = (-128, 127, torch.int8) if zp_dtype == torch.int8 else (0, 255, torch.uint8)
    def fn(x, scales, zeros):
        q = qd.quantize_per_channel.default(x, scales, zeros, 0, low, high, storage)
        return q, qd.dequantize_per_channel.default(q, scales, zeros, 0, low, high, storage)
    q, dq = run_native(tmp_path, fn, (x, scales, zeros), convert_options={"fully_standard": True})
    expected_q = fn(x, scales, zeros)[0].numpy()
    expected_dq = ((expected_q.astype(np.int64) - zeros.to(torch.int64).numpy()[:, None])
                   * scales.numpy()[:, None]).astype(np.float32)
    np.testing.assert_array_equal(q, expected_q)
    np.testing.assert_array_equal(dq, expected_dq)
    eager_dq = fn(x, scales, zeros)[1].numpy()
    record_property("eager-float divergence", int(np.count_nonzero(eager_dq != expected_dq)))
