"""Signed addmm readout against exact Fraction arithmetic, including native MLIR."""

from fractions import Fraction
import math

import numpy as np
import pytest
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401

import m2m
from m2m.capture import integer_readout as ops
from m2m.capture.pt2e_compat import allow_exported_model_train_eval
from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
from m2m.capture.pt2e_integerize import integerize_pt2e
from m2m.ir.quant_attrs import fp32
from tests.native_execution import run_native
from tests.test_exact_quant_arithmetic import SCHEMES, oracle_float32
from tests.test_quant_schema import _check_capture


class Addmm(torch.nn.Module):
    def __init__(self, alpha, scheme, bias_kind, *, per_channel=False, relu=False,
                 activation_scale=0.5, weight_scale=0.25, output_scale=0.5):
        super().__init__()
        self.alpha, self.scheme, self.bias_kind = alpha, scheme, bias_kind
        self.per_channel, self.relu = per_channel, relu
        self.sa, self.sw, self.so = activation_scale, weight_scale, output_scale
        self.register_buffer("scales", torch.tensor([0.25, 0.5, 0.125, 0.03], dtype=torch.float64))
        self.register_buffer("zeros", torch.zeros(4, dtype=torch.int64))
        # alpha=-2, beta=-0.5, sa=0.5, sw=0.25 -> exact seed 2*bias.
        # Positive and negative bias exercise both signs of the signed seed.
        values = [0.5, -1., 1.5, -2.] if bias_kind == "seed" else [0.03125, -0.0625, 0.1, -0.3]
        if bias_kind == "none":
            values = [float("nan"), float("inf"), -float("inf"), 0.]
        self.register_buffer("bias", torch.tensor(values))

    def forward(self, x, w, runtime_bias):
        qd = torch.ops.quantized_decomposed
        x = qd.dequantize_per_tensor.default(x, self.sa, 0, -128, 127, torch.int8)
        w = (qd.dequantize_per_channel.default(w, self.scales, self.zeros, 1, -128, 127, torch.int8)
             if self.per_channel else
             qd.dequantize_per_tensor.default(w, self.sw, 0, -128, 127, torch.int8))
        bias = runtime_bias if self.bias_kind == "runtime" else self.bias
        y = torch.ops.aten.addmm.default(bias, x, w, alpha=self.alpha,
                                       beta=0. if self.bias_kind == "none" else -0.5)
        if self.relu:
            y = torch.relu(y)
        unsigned, low, high, zp = self.scheme
        return qd.quantize_per_tensor.default(y, self.so, zp, low, high,
                                            torch.uint8 if unsigned else torch.int8)


def _oracle(x, w, sa, weights, alpha, bias, beta, so, scheme, relu=False):
    """No compiler/reference helpers and no floating-point products or sums."""
    unsigned, low, high, zp = scheme
    acc = x.to(torch.int64) @ w.to(torch.int64)
    scales = [Fraction(fp32(sa)) * Fraction(alpha) * Fraction(fp32(s)) for s in weights]
    out = []
    for row in acc.tolist():
        values = []
        for c, v in enumerate(row):
            value = v * scales[c % len(scales)]
            if beta != 0:
                value += Fraction(float(bias[c])) * Fraction(beta)
            if relu:
                value = max(Fraction(0), value)
            values.append(min(high, max(low, round(value / Fraction(fp32(so))) + zp)))
        out.append(values)
    return torch.tensor(out, dtype=torch.uint8 if unsigned else torch.int8)


def _check(model, inputs, tmp_path, *, expected_seeds):
    captured = torch.export.export(model, inputs).module()
    x, w, runtime_bias = inputs
    weights = model.scales.tolist() if model.per_channel else [model.sw]
    beta = 0. if model.bias_kind == "none" else -0.5
    bias = runtime_bias if model.bias_kind == "runtime" else model.bias
    expected = _oracle(x, w, model.sa, weights, model.alpha, bias, beta, model.so, model.scheme, model.relu)
    reference = run_pt2e_integer_reference(captured, inputs, expected_contractions=1)
    integer, receipt = integerize_pt2e(captured, inputs)
    assert receipt["refusals"] == [] and receipt["quantized_contractions_integerized"] == 1
    assert receipt["int32_bias_seeds"] == expected_seeds
    assert receipt["fused_requantizations"] == 1 and not receipt["host_only"]
    assert torch.equal(reference.output, expected)
    assert torch.equal(integer(*inputs), expected)
    if expected_seeds:
        seed = next(getattr(integer, n.target) for n in integer.graph.nodes
                    if n.op == "get_attr" and str(n.target).startswith("_m2m_int32_seed"))
        assert torch.equal(seed, torch.tensor([1, -2, 3, -4], dtype=torch.int32))

    allow_exported_model_train_eval(integer)
    converted = m2m.convert(integer, inputs, backend="fx_importer", fully_standard=True)
    _check_capture(converted, set())
    # All data-path operations, even bias decoding and RNE, are integer ops.
    assert not any(op.name in {"arith.mulf", "arith.addf", "arith.divf", "arith.sitofp", "arith.fptosi"}
                   for op in converted.module.walk())
    for op in converted.module.walk():
        attrs = op.attributes
        if "prov.quant.scale_numerator" not in attrs or "prov.quant.alpha" not in attrs:
            continue
        ratios = [Fraction(int(n.data), int(d.data)) for n, d in zip(
            attrs["prov.quant.scale_numerator"], attrs["prov.quant.scale_denominator"], strict=True)]
        want = [Fraction(model.alpha) * Fraction(fp32(model.sa)) * Fraction(fp32(s)) for s in weights]
        if getattr(attrs.get("prov.role"), "data", None) == "readout":
            want = [s / Fraction(fp32(model.so)) for s in want]
        assert ratios == want and all(s < 0 for s in ratios)
        assert getattr(attrs.get("prov.quant.host_only"), "data", "false") != "true"
    actual, = run_native(tmp_path, integer, inputs, convert_options={"fully_standard": True})
    np.testing.assert_array_equal(actual, expected.numpy())


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("bias_kind", ["none", "seed", "separate", "runtime"])
def test_negative_addmm_schemes_bias_and_native(tmp_path, scheme, bias_kind):
    x = torch.arange(-128, 128, dtype=torch.int8).reshape(-1, 1)
    w = torch.tensor([[-3, -1, 1, 3]], dtype=torch.int8)
    bias = torch.tensor([0.1, -0.25, 0.375, -0.5])
    _check(Addmm(-2., scheme, bias_kind), (x, w, bias), tmp_path,
           expected_seeds=int(bias_kind == "seed"))


@pytest.mark.parametrize("scheme", [SCHEMES[2], SCHEMES[6]])
def test_negative_addmm_per_channel_and_relu(tmp_path, scheme):
    generator = torch.Generator().manual_seed(294)
    x = torch.randint(-128, 128, (40, 9), dtype=torch.int8, generator=generator)
    w = torch.randint(-128, 128, (9, 4), dtype=torch.int8, generator=generator)
    _check(Addmm(-1.3, scheme, "runtime", per_channel=True, relu=True,
                 activation_scale=0.03, output_scale=0.0371),
           (x, w, torch.tensor([0.03, -0.5, 0.125, -1.])), tmp_path, expected_seeds=0)


def test_negative_addmm_unsafe_seed_remains_exact_bias(tmp_path):
    model = Addmm(-2., SCHEMES[2], "seed")
    model.bias.copy_(torch.tensor([2.**31, -2.**31, 0.125, -0.125]))
    inputs = (torch.tensor([[-128], [-1], [1], [127]], dtype=torch.int8),
              torch.tensor([[-128, -1, 1, 127]], dtype=torch.int8), torch.zeros(4))
    _check(model, inputs, tmp_path, expected_seeds=0)


def test_negative_addmm_float_result_rounds_once(tmp_path):
    x = torch.arange(-128, 128, dtype=torch.int8).reshape(-1, 1)
    w = torch.tensor([[-127, -1, 1, 127]], dtype=torch.int8)
    bias = torch.tensor([0.1, -0.3, 0.125, -0.5])
    class FloatResult(torch.nn.Module):
        def forward(self, x, w, bias):
            dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
            return torch.addmm(bias, dq(x, 0.03, 0, -128, 127, torch.int8),
                               dq(w, 0.25, 0, -128, 127, torch.int8), alpha=-1.3, beta=-0.5)
    captured = torch.export.export(FloatResult(), (x, w, bias)).module()
    expected = []
    for row in (x.to(torch.int64) @ w.to(torch.int64)).tolist():
        values = []
        for c, v in enumerate(row):
            value = v * Fraction(fp32(0.03)) * Fraction(0.25) * Fraction(-1.3)
            value += Fraction(float(bias[c])) * Fraction(-0.5)
            values.append(oracle_float32(value))
        expected.append(values)
    expected = np.array(expected, dtype=np.float32)
    reference = run_pt2e_integer_reference(captured, (x, w, bias), expected_contractions=1)
    integer, receipt = integerize_pt2e(captured, (x, w, bias))
    assert receipt["refusals"] == [] and receipt["fused_requantizations"] == 0
    np.testing.assert_array_equal(reference.output.numpy().view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(integer(x, w, bias).numpy().view(np.uint32), expected.view(np.uint32))
    allow_exported_model_train_eval(integer)
    actual, = run_native(tmp_path, integer, (x, w, bias), convert_options={"fully_standard": True})
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_negative_addmm_exhaustive_int8_operand_pairs(tmp_path):
    # Every one-term signed int8 product, with negative alpha and no bias.
    x = torch.arange(-128, 128, dtype=torch.int8).reshape(-1, 1)
    w = x.T.contiguous()
    class Exhaustive(torch.nn.Module):
        def forward(self, x, w):
            dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
            y = torch.addmm(torch.zeros(256), dq(x, 0.5, 0, -128, 127, torch.int8),
                            dq(w, 0.25, 0, -128, 127, torch.int8), alpha=-1., beta=0.)
            return torch.ops.quantized_decomposed.quantize_per_tensor.default(y, 0.25, 3, -128, 127, torch.int8)
    captured = torch.export.export(Exhaustive(), (x, w)).module()
    expected = _oracle(x, w, 0.5, [0.25], -1., None, 0., 0.25, SCHEMES[2])
    reference = run_pt2e_integer_reference(captured, (x, w), expected_contractions=1)
    integer, receipt = integerize_pt2e(captured, (x, w))
    assert receipt["refusals"] == []
    assert torch.equal(reference.output, expected) and torch.equal(integer(x, w), expected)
    allow_exported_model_train_eval(integer)
    actual, = run_native(tmp_path, integer, (x, w), convert_options={"fully_standard": True})
    np.testing.assert_array_equal(actual, expected.numpy())


@pytest.mark.parametrize("unsigned,low,high,zp", SCHEMES)
def test_negative_readout_exhaustive_ties_zero_point_clamp_relu(tmp_path, unsigned, low, high, zp):
    # Includes every byte-code half tie and neighboring integer, of both signs,
    # plus the full int32 endpoints (no negation in int32 is permissible).
    acc = torch.cat((torch.arange(-1025, 1026, dtype=torch.int32),
                     torch.tensor([-(1 << 31), (1 << 31) - 1], dtype=torch.int32)))
    def fn(acc):
        return tuple(ops.exact_readout(acc, 0.5, [1.], -1., None, 1., 1., zp, low, high, relu, unsigned)
                     for relu in (False, True))
    actual = run_native(tmp_path, fn, (acc,), convert_options={"fully_standard": True})
    for relu, native, eager in zip((False, True), actual, fn(acc), strict=True):
        expected = np.array([min(high, max(low, round(max(Fraction(0), Fraction(-int(v), 2))
                                                     if relu else Fraction(-int(v), 2)) + zp))
                             for v in acc], dtype=np.uint8 if unsigned else np.int8)
        np.testing.assert_array_equal(native, expected)
        np.testing.assert_array_equal(eager.numpy(), expected)


@pytest.mark.parametrize("alpha", [-2.**-1074, -2.**-1022, -0.1, -1., -1.5,
                                   -float.fromhex("0x1.fffffffffffffp+1023")])
def test_negative_alpha_binary64_extremes_native(tmp_path, alpha):
    # Even products that underflow/overflow fp32/fp64 remain exact rational
    # coefficients until quantization; alpha itself is never canonicalized.
    model = Addmm(alpha, SCHEMES[2], "none", activation_scale=2.**-149,
                  weight_scale=2.**-149, output_scale=2.**-149)
    x = torch.tensor([[-128], [-3], [-1], [0], [1], [3], [127]], dtype=torch.int8)
    w = torch.tensor([[-128, -1, 1, 127]], dtype=torch.int8)
    _check(model, (x, w, torch.zeros(4)), tmp_path, expected_seeds=0)


@pytest.mark.parametrize("alpha", [0., -0., float("inf"), -float("inf"), float("nan"), -((1 << 53) + 1)])
def test_unsupported_alpha_is_refused_cleanly(alpha):
    inputs = (torch.ones(1, 1, dtype=torch.int8), torch.ones(1, 4, dtype=torch.int8), torch.zeros(4))
    captured = torch.export.export(Addmm(-1., SCHEMES[0], "none"), inputs).module()
    node = next(n for n in captured.graph.nodes if n.target == torch.ops.aten.addmm.default)
    node.kwargs = {**node.kwargs, "alpha": alpha}
    captured.recompile()
    _, receipt = integerize_pt2e(captured, inputs)
    assert receipt["quantized_contractions_integerized"] == 0
    assert len(receipt["refusals"]) == 1
    reason = ("alpha 0" if alpha == 0 else "finite literals" if not math.isfinite(alpha)
              else "exactly representable as binary64")
    assert reason in receipt["refusals"][0]["reason"]
    with pytest.raises(ValueError, match="finite|nonzero|exactly representable"):
        run_pt2e_integer_reference(captured, inputs)


@pytest.mark.parametrize("dtype,zp", [(torch.int8, 128), (torch.uint8, -1), (torch.int16, 0)])
def test_negative_alpha_does_not_admit_unsupported_input_schemes(dtype, zp):
    low, high = (-128, 127) if dtype == torch.int8 else (0, 255)
    class Unsupported(torch.nn.Module):
        def forward(self, x, w):
            dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
            return torch.addmm(torch.zeros(1), dq(x, 0.5, zp, low, high, dtype),
                               dq(w, 0.25, zp, low, high, dtype), alpha=-1.)
    inputs = (torch.ones(1, 1, dtype=dtype), torch.ones(1, 1, dtype=dtype))
    captured = torch.export.export(Unsupported(), inputs).module()
    _, receipt = integerize_pt2e(captured, inputs)
    assert receipt["quantized_contractions_integerized"] == 0 and len(receipt["refusals"]) == 1
    with pytest.raises(ValueError, match="zero point|int8/uint8"):
        run_pt2e_integer_reference(captured, inputs)
