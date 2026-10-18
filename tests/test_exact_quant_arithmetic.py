"""Exact Fraction oracle. Eager-float divergence is informational, never a tolerance."""
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401

from m2m.capture import integer_readout as ops
from tests.native_execution import run_native


def oracle_code(value, scale, zp, low, high):
    return min(high, max(low, round(value / Fraction(float(scale))) + zp))


def oracle_float32(value):
    """Bracket an exact Fraction with binary32 neighbors, then choose exact RNE.

    Independent of production's significand rounding. No rational is converted
    to a Python float or decimal string, including at subnormal/overflow ties.
    """
    sign = 0x80000000 if value < 0 else 0
    value = abs(value)

    def decode(bits):
        exponent, mantissa = bits >> 23, bits & 0x7fffff
        return Fraction(mantissa if exponent == 0 else mantissa + (1 << 23)) * Fraction(2) ** (
            -149 if exponent == 0 else exponent - 150)

    low, high = 0, 0x7f800000
    while high - low > 1:
        middle = (low + high) // 2
        if decode(middle) <= value:
            low = middle
        else:
            high = middle
    # The overflow neighbor is the hypothetical finite value 2**128.
    midpoint = (decode(low) + decode(high)) / 2
    bits = high if value > midpoint or (value == midpoint and low & 1) else low
    return np.array(sign | bits, dtype=np.uint32).view(np.float32)[()]


def test_fraction_float32_midpoints_and_double_rounding():
    from m2m.capture.exact_quant import rounded_float
    from m2m.ir.quant_attrs import fp32

    example = Fraction("1.00000005960464477539062500000001")
    assert oracle_float32(example).view(np.uint32) == 0x3f800001
    assert fp32(example) == float(oracle_float32(example))
    # Integer inputs can double-round through binary64 too.
    integer = (1 << 80) + (1 << 56) + 1
    assert oracle_float32(Fraction(integer)).view(np.uint32) == 0x67800001
    assert fp32(integer) == float(oracle_float32(Fraction(integer)))
    # Both tie parities at every normal binade, plus subnormal, normal-boundary
    # and overflow neighbors. Tiny offsets are lost by an intervening binary64.
    neighbors = [0, 1, 0x7ffffe, 0x7fffff, 0x7f7fffff]
    neighbors += [(exponent << 23) + mantissa for exponent in range(1, 255)
                  for mantissa in (0, 1, 0x7fffff)]
    for low_bits in neighbors:
        low = Fraction(float(np.array(low_bits, dtype=np.uint32).view(np.float32)))
        high = (Fraction(2) ** 128 if low_bits == 0x7f7fffff else
                Fraction(float(np.array(low_bits + 1, dtype=np.uint32).view(np.float32))))
        midpoint, epsilon = (low + high) / 2, (high - low) / (1 << 60)
        for offset, expected in [(-epsilon, low_bits), (0, low_bits + (low_bits & 1)),
                                 (epsilon, low_bits + 1)]:
            for sign in (-1, 1):
                value = sign * (midpoint + offset)
                bits = expected | (0x80000000 if sign < 0 else 0)
                assert oracle_float32(value).view(np.uint32) == bits
                for actual in (rounded_float(value), fp32(value)):
                    assert np.float32(actual).view(np.uint32) == bits


SCHEMES = [
    (False, -128, 127, 0), (False, -127, 127, 0),
    (False, -128, 127, 3), (False, -127, 126, -17),
    (True, 0, 255, 0), (True, 1, 255, 128),
    (True, 0, 254, 129), (True, 1, 254, 17),
]


@pytest.mark.parametrize("unsigned,low,high,zp", SCHEMES)
def test_exact_quantize_all_codes_ties_and_neighbors(tmp_path, record_property, unsigned, low, high, zp):
    dtype = torch.uint8 if unsigned else torch.int8
    scale = 0.0371
    # Every output-code boundary plus both adjacent f32 values, saturation.
    centers = torch.tensor([(i + 0.5) * scale for i in range(low - zp - 2, high - zp + 2)])
    x = torch.cat((centers, torch.nextafter(centers, torch.full_like(centers, -float("inf"))),
                   torch.nextafter(centers, torch.full_like(centers, float("inf"))),
                   torch.tensor([-1e30, 1e30, 0.0])))
    q = torch.ops.quantized_decomposed.quantize_per_tensor.default
    fn = lambda x: q(x, scale, zp, low, high, dtype)
    actual, = run_native(tmp_path, fn, (x,), convert_options={"fully_standard": True})
    expected = np.array([oracle_code(Fraction(float(v)), scale, zp, low, high) for v in x], dtype=actual.dtype)
    np.testing.assert_array_equal(actual, expected)
    divergence = int(np.count_nonzero(fn(x).numpy() != expected))
    record_property("eager-float divergence", divergence)
    print(f"eager-float divergence (quantize {dtype}, zp={zp}): {divergence}/{x.numel()}")


@pytest.mark.parametrize("unsigned,low,high,zp", SCHEMES)
@pytest.mark.parametrize("alpha", [1.5, -1.5])
def test_exact_readout_property_schemes_and_channels(tmp_path, record_property, unsigned, low, high, zp, alpha):
    generator = torch.Generator().manual_seed(280)
    acc = torch.cat((torch.arange(-1024, 1025, dtype=torch.int32),
                     torch.randint(-(1 << 31), (1 << 31) - 1, (400,), generator=generator, dtype=torch.int32),
                     torch.tensor([16777215, 16777216, 16777217, -16777217, -(1 << 31), (1 << 31) - 1],
                                  dtype=torch.int32)))
    acc = acc[:, None].expand(-1, 3).contiguous()
    bias = torch.tensor([0.015, -0.025, 0.035], dtype=torch.float32)
    weights = [0.01, 0.03, 0.25]
    sa, beta, so = 0.03, 0.5, 0.01

    def fn(acc, bias):
        return ops.exact_readout(acc, sa, weights, alpha, bias, beta, so, zp, low, high, False, unsigned)

    expected = np.array([
        [oracle_code(Fraction(int(v)) * Fraction(sa) * Fraction(weights[c]) * Fraction(alpha)
                     + Fraction(float(bias[c])) * Fraction(beta), so, zp, low, high)
         for c, v in enumerate(row)] for row in acc
    ], dtype=np.uint8 if unsigned else np.int8)
    assert torch.equal(fn(acc, bias), torch.from_numpy(expected))
    actual, = run_native(tmp_path, fn, (acc, bias), convert_options={"fully_standard": True})
    np.testing.assert_array_equal(actual, expected)
    eager = torch.round(acc.float() * torch.tensor([sa * w * alpha / so for w in weights])
                        + bias * beta / so) + zp
    eager = eager.clamp(low, high).to(torch.uint8 if unsigned else torch.int8).numpy()
    divergence = int(np.count_nonzero(eager != expected))
    record_property("eager-float divergence", divergence)
    print(f"eager-float divergence (readout {unsigned}, zp={zp}): {divergence}/{expected.size}")


def test_rounding_precedes_odd_zero_point_and_bias_is_not_prerounded(tmp_path):
    acc = torch.tensor([[-3, -1, 1, 3, 5], [1, 1, 1, 1, 1]], dtype=torch.int32)
    bias = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    def fn(acc, bias):
        return ops.exact_readout(acc, 0.5, [1.0], 1., bias, 1., 1., 3, -127, 127)
    expected = torch.tensor([[1, 3, 3, 5, 5], [3, 3, 3, 3, 3]], dtype=torch.int8)
    assert torch.equal(fn(acc, bias), expected)
    actual, = run_native(tmp_path, fn, (acc, bias), convert_options={"fully_standard": True})
    np.testing.assert_array_equal(actual, expected.numpy())


@pytest.mark.parametrize("scale", [0.5, 0.03, 2.0**-149, 2.0**100, 2.0**-1074, 2.0**1023])
def test_float_readout_rounds_once_including_subnormals(tmp_path, scale):
    acc = torch.tensor([0, 1, -1, 16777217, -16777217, 2147483647, -2147483648], dtype=torch.int32)
    def fn(acc):
        return ops.acc_readout(acc, [scale])
    expected = np.array([oracle_float32(Fraction(int(v)) * Fraction(scale)) for v in acc])
    actual, = run_native(tmp_path, fn, (acc,), convert_options={"fully_standard": True})
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(fn(acc).numpy().view(np.uint32), expected.view(np.uint32))


def test_requantize_does_not_round_large_accumulator_to_float_first(tmp_path):
    acc = torch.tensor([16777217, -16777217, (1 << 31) - 1, -(1 << 31)], dtype=torch.int32)
    scales = [1.5 / 16777217]
    fn = lambda x: ops.requantize(x, scales, 3, -128, 127)
    expected = np.array([oracle_code(Fraction(int(v)) * Fraction(scales[0]), 1., 3, -128, 127)
                         for v in acc], dtype=np.int8)
    actual, = run_native(tmp_path, fn, (acc,), convert_options={"fully_standard": True})
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(fn(acc).numpy(), expected)


def test_float_readout_avoids_f64_to_f32_double_rounding(tmp_path):
    acc = torch.tensor([3, -3], dtype=torch.int32)
    scale = float.fromhex("0x1.555556aaaaaabp-2")
    fn = lambda x: ops.acc_readout(x, [scale])
    # 3*scale is just ABOVE the f32 midpoint, but f64 multiplication
    # rounds to that midpoint and a subsequent f32 cast rounds down.
    expected = np.array([0x3f800001, 0xbf800001], dtype=np.uint32)
    actual, = run_native(tmp_path, fn, (acc,), convert_options={"fully_standard": True})
    np.testing.assert_array_equal(actual.view(np.uint32), expected)
    np.testing.assert_array_equal(fn(acc).numpy().view(np.uint32), expected)


@pytest.mark.parametrize("unsigned", [False, True])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.float64])
def test_per_channel_quantize_dequantize_exact(tmp_path, unsigned, scale_dtype):
    low, high, zp = (1, 254, 129) if unsigned else (-127, 127, -3)
    dtype = torch.uint8 if unsigned else torch.int8
    scales = torch.tensor([0.03, 0.0371, 0.5], dtype=scale_dtype)
    zps = torch.tensor([zp, 0 if not unsigned else 17, 3], dtype=torch.int64)
    values = torch.arange(-300, 301, dtype=torch.float32) * 0.01855
    x = values[None, :].expand(3, -1).contiguous()
    def fn(x, scales, zps):
        qd = torch.ops.quantized_decomposed
        q = qd.quantize_per_channel.default(x, scales, zps, 0, low, high, dtype)
        return q, qd.dequantize_per_channel.default(q, scales, zps, 0, low, high, dtype)
    q, dq = run_native(tmp_path, fn, (x, scales, zps), convert_options={"fully_standard": True})
    want = np.array([[oracle_code(Fraction(float(v)), float(scales[c]), int(zps[c]), low, high)
                      for v in row] for c, row in enumerate(x)], dtype=q.dtype)
    np.testing.assert_array_equal(q, want)
    want_dq = [[oracle_float32(Fraction(int(code) - int(zps[c])) * Fraction(float(scales[c])))
                for code in row] for c, row in enumerate(want)]
    np.testing.assert_array_equal(dq.view(np.uint32), np.array(want_dq, dtype=np.float32).view(np.uint32))


@pytest.mark.parametrize("dtype,zp", [(torch.int8, 128), (torch.uint8, -1), (torch.int16, 0)])
def test_unsupported_centered_operand_schemes_are_explicitly_refused(dtype, zp):
    from m2m.capture.pt2e_integerize import integerize_pt2e
    low, high = (-128, 127) if dtype == torch.int8 else (0, 255)
    class Model(torch.nn.Module):
        def forward(self, x, w):
            dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
            return torch.mm(dq(x, 0.03, zp, low, high, dtype), dq(w, 0.02, zp, low, high, dtype))
    inputs = (torch.ones(2, 3, dtype=dtype), torch.ones(3, 2, dtype=dtype))
    model = torch.export.export(Model(), inputs).module()
    _, receipt = integerize_pt2e(model, inputs, numerics="integer_readout")
    assert receipt["quantized_contractions_integerized"] == 0
    assert len(receipt["refusals"]) == 1
    assert receipt["quantized_contractions_remaining"] == 1


def test_exact_bias_seed_requires_integral_value_and_overflow_proof(tmp_path):
    from m2m.capture.pt2e_integerize import integerize_pt2e
    from tests.native_execution import assert_native_matches_eager
    class Model(torch.nn.Module):
        def __init__(self, b):
            super().__init__()
            self.register_buffer("bias", torch.tensor([b], dtype=torch.float32))
        def forward(self, x, w):
            dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
            value = torch.addmm(self.bias, dq(x, 0.5, 0, -128, 127, torch.int8),
                                dq(w, 0.5, 0, -128, 127, torch.int8))
            return torch.ops.quantized_decomposed.quantize_per_tensor.default(value, 1., 3, -128, 127, torch.int8)
    inputs = (torch.tensor([[1]], dtype=torch.int8), torch.tensor([[1]], dtype=torch.int8))
    for b, expected_seed in [(0.5, 1), (0.125, 0), (2.0**31, 0)]:
        model = torch.export.export(Model(b), inputs).module()
        integer, receipt = integerize_pt2e(model, inputs, numerics="integer_readout")
        assert receipt["refusals"] == []
        assert receipt["int32_bias_seeds"] == expected_seed
        expected = oracle_code(Fraction(1, 4) + Fraction(b), 1., 3, -128, 127)
        assert integer(*inputs).item() == expected
        from m2m.capture.pt2e_compat import allow_exported_model_train_eval
        allow_exported_model_train_eval(integer)
        probe_dir = tmp_path / str(b)
        probe_dir.mkdir()
        assert_native_matches_eager(probe_dir, integer, inputs, atol=0, rtol=0,
                                    convert_options={"fully_standard": True})


def test_dequantize_avoids_double_rounding(tmp_path):
    scale = float.fromhex("0x1.555556aaaaaabp-2")
    x = torch.tensor([3, -3], dtype=torch.int8)
    fn = lambda x: torch.ops.quantized_decomposed.dequantize_per_tensor.default(
        x, scale, 0, -128, 127, torch.int8)
    actual, = run_native(tmp_path, fn, (x,), convert_options={"fully_standard": True})
    np.testing.assert_array_equal(actual.view(np.uint32), np.array([0x3f800001, 0xbf800001], dtype=np.uint32))


@pytest.mark.parametrize("scale", [2.0**-1074, 0.03, 2.0**1023])
def test_float64_extremes_and_infinities_quantize_without_poison(tmp_path, scale):
    x = torch.tensor([-float("inf"), -1.7976931348623157e308, -2.0**-1074, -0.0,
                      0., 2.0**-1074, 1.7976931348623157e308, float("inf")], dtype=torch.float64)
    fn = lambda x: ops.quantize(x, scale, 129, 1, 254, True)
    expected = np.array([1 if v == -float("inf") else 254 if v == float("inf") else
                         oracle_code(Fraction(float(v)), scale, 129, 1, 254) for v in x], dtype=np.uint8)
    actual, = run_native(tmp_path, fn, (x,), convert_options={"fully_standard": True})
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(fn(x).numpy(), expected)


def test_seeded_operator_refuses_overflow_instead_of_wrapping():
    a = b = torch.ones((1, 1), dtype=torch.int8)
    with pytest.raises(ValueError, match="overflow"):
        ops.int_mm_seeded(a, b, torch.tensor([2147483647], dtype=torch.int32))


def test_backward_input_bound_includes_all_kernel_taps(monkeypatch):
    from types import SimpleNamespace
    from torch.fx import Graph, GraphModule
    from torch.fx.passes.shape_prop import ShapeProp
    from m2m.capture.pt2e_integerize import integerize_pt2e
    import operator
    graph = Graph()
    operands = []
    for name, shape in [("g", (1, 129, 33, 33)), ("x", (1, 1, 65, 65)), ("w", (129, 1, 33, 33))]:
        value = graph.placeholder(name)
        value.meta["tensor_meta"] = SimpleNamespace(shape=shape, dtype=torch.int8)
        dq = graph.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                                 (value, 1., 0, -128, 127, torch.int8))
        dq.meta["tensor_meta"] = SimpleNamespace(shape=shape, dtype=torch.float32)
        operands.append(dq)
    node = graph.call_function(torch.ops.aten.convolution_backward.default,
                              (*operands, [129], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, False, False]))
    graph.output(graph.call_function(operator.getitem, (node, 0)))
    model = GraphModule({}, graph)
    # Shape metadata suffices to prove this refusal. Do not execute the huge
    # floating convolution as a prerequisite to a declaration-wide bound.
    monkeypatch.setattr(ShapeProp, "propagate", lambda *args: None)
    _, receipt = integerize_pt2e(model, (), numerics="integer_readout")
    assert receipt["quantized_contractions_integerized"] == 0
    assert "overflow" in receipt["refusals"][0]["reason"]


@pytest.mark.parametrize("unsigned,low,high,zp", [(True, 1, 254, 0), (False, -127, 127, -128),
                                                (False, -128, 126, 127)])
def test_narrow_range_does_not_require_zero_point_inside_clamp(tmp_path, unsigned, low, high, zp):
    acc = torch.arange(-512, 513, dtype=torch.int32)
    def fn(acc):
        return (ops.exact_readout(acc, 0.5, [1.0], 1., None, 1., 1., zp, low, high, False, unsigned),
                ops.exact_readout(acc, 0.5, [1.0], 1., None, 1., 1., zp, low, high, True, unsigned))
    actual = run_native(tmp_path, fn, (acc,), convert_options={"fully_standard": True})
    for relu, result in zip((False, True), actual):
        expected = np.array([min(high, max(max(low, zp) if relu else low,
                                          round(Fraction(int(v), 2)) + zp)) for v in acc], dtype=result.dtype)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(fn(acc)[int(relu)].numpy(), expected)


def test_zero_beta_ignores_bias_even_with_extreme_scale(tmp_path):
    acc = torch.tensor([1, -1], dtype=torch.int32)
    bias = torch.tensor([float("nan"), float("inf")])
    def fn(acc, bias):
        return ops.exact_readout(acc, 2.0**-200, [1.], 1., bias, 0., 1., 0, -128, 127)
    actual, = run_native(tmp_path, fn, (acc, bias), convert_options={"fully_standard": True})
    np.testing.assert_array_equal(actual, np.zeros(2, dtype=np.int8))
    assert torch.equal(fn(acc, bias), torch.zeros(2, dtype=torch.int8))


def test_integerized_uint8_output_records_unsigned_storage(tmp_path):
    from m2m.capture.pt2e_integerize import integerize_pt2e
    from m2m.capture.pt2e_compat import allow_exported_model_train_eval
    from tests.native_execution import assert_native_matches_eager
    import m2m
    class Model(torch.nn.Module):
        def forward(self, x, w):
            qd = torch.ops.quantized_decomposed
            y = torch.mm(qd.dequantize_per_tensor.default(x, 0.5, 0, -128, 127, torch.int8),
                         qd.dequantize_per_tensor.default(w, 0.25, 0, -128, 127, torch.int8))
            return qd.quantize_per_tensor.default(y, 0.1, 129, 1, 254, torch.uint8)
    inputs = (torch.tensor([[-3, -1], [1, 3]], dtype=torch.int8),
              torch.tensor([[7, 2], [-5, 3]], dtype=torch.int8))
    integer, receipt = integerize_pt2e(torch.export.export(Model(), inputs).module(), inputs,
                                      numerics="integer_readout")
    assert receipt["qparams"][0]["output_dtype"] == "torch.uint8"
    assert receipt["fused_requantizations"] == 1 and receipt["refusals"] == []
    allow_exported_model_train_eval(integer)
    converted = m2m.convert(integer, inputs, backend="fx_importer", fully_standard=True)
    assert 'prov.quant.storage_dtype = "uint8"' in converted.mlir_text
    assert_native_matches_eager(tmp_path, integer, inputs, atol=0, rtol=0,
                                convert_options={"fully_standard": True})


@pytest.mark.parametrize("parameter", ["alpha", "beta"])
def test_large_integer_scalar_is_refused_instead_of_rounded(parameter):
    from m2m.capture.pt2e_integerize import integerize_pt2e
    class Model(torch.nn.Module):
        def forward(self, x, w, bias):
            dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
            return torch.ops.aten.addmm.default(bias, dq(x, 0.5, 0, -128, 127, torch.int8),
                                                dq(w, 0.5, 0, -128, 127, torch.int8),
                                                **{parameter: (1 << 53) + 1})
    inputs = (torch.ones(1, 1, dtype=torch.int8), torch.ones(1, 1, dtype=torch.int8), torch.ones(1))
    model = torch.export.export(Model(), inputs).module()
    from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
    with pytest.raises(ValueError, match="exactly representable as binary64"):
        run_pt2e_integer_reference(model, inputs, numerics="integer_readout")
    _, receipt = integerize_pt2e(model, inputs, numerics="integer_readout")
    assert receipt["quantized_contractions_integerized"] == 0
    assert "exactly representable as binary64" in receipt["refusals"][0]["reason"]
