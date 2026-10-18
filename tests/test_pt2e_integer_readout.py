"""Exact integer readout and native lowering; eager-float differences are informational."""

from __future__ import annotations

import pytest
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401
from torch import nn

import m2m
from m2m.capture.pt2e_compat import allow_exported_model_train_eval
from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
from m2m.capture.pt2e_integerize import integerize_pt2e
from m2m.coverage import opaque_report
from tests.native_execution import assert_native_matches_eager
from tests.test_pt2e_contraction_coverage import CASES, _captured

A = torch.ops.aten
Q = torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ = torch.ops.quantized_decomposed.dequantize_per_tensor.default
DQC = torch.ops.quantized_decomposed.dequantize_per_channel.default


def qdq(value, scale, low=-128):
    return DQ(Q(value, scale, 0, low, 127, torch.int8), scale, 0, low, 127, torch.int8)


class Requantized(nn.Module):
    """conv (int32 bias, ReLU) -> int8 -> addmm (alpha/beta, bias) -> int8 output."""

    def __init__(self, per_channel=False):
        super().__init__()
        generator = torch.Generator().manual_seed(3)
        self.register_buffer("w1", torch.randint(-100, 101, (6, 2, 3, 3), generator=generator, dtype=torch.int8))
        self.register_buffer("b1", torch.randn(6, generator=generator) * 3)
        self.register_buffer("w2", torch.randint(-100, 101, (24, 5), generator=generator, dtype=torch.int8))
        self.register_buffer("b2", torch.randn(5, generator=generator) * 3)
        self.register_buffer("s1", torch.tensor([0.01, 0.02, 0.015, 0.03, 0.012, 0.02]))
        self.register_buffer("z1", torch.zeros(6, dtype=torch.int64))
        self.per_channel = per_channel

    def forward(self, x):
        w1 = (DQC(self.w1, self.s1, self.z1, 0, -127, 127, torch.int8) if self.per_channel
              else DQ(self.w1, 0.02, 0, -127, 127, torch.int8))
        y = A.convolution.default(qdq(x, 0.05), w1, self.b1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
        y = qdq(A.relu.default(y), 0.04)
        y = y.reshape(1, -1)[:, :24]
        z = A.addmm.default(self.b2, qdq(y, 0.04), DQ(self.w2, 0.03, 0, -127, 127, torch.int8),
                            beta=0.5, alpha=2.0)
        return Q(z, 0.1, 0, -128, 127, torch.int8)


def _model(per_channel=False):
    inputs = (torch.randn(1, 4, 5, 5, generator=torch.Generator().manual_seed(11)) * 2,)
    return torch.export.export(Requantized(per_channel).eval(), inputs).module(), inputs


@pytest.mark.parametrize("per_channel", [False, True], ids=["per_tensor", "per_channel"])
def test_fused_requantization_is_exact(per_channel, record_property):
    model, inputs = _model(per_channel)
    portable = model(*inputs)
    reference = run_pt2e_integer_reference(model, inputs, numerics="integer_readout")
    integer, receipt = integerize_pt2e(model, inputs, numerics="integer_readout")
    assert receipt["numerics"] == "integer_readout" and receipt["refusals"] == []
    assert receipt["fused_requantizations"] == 2  # relu->Q after the conv, Q after the addmm
    assert receipt["int32_bias_seeds"] == 0  # arbitrary float bias must not be pre-rounded
    assert [row["readout"] for row in receipt["qparams"]] == ["fused_requantize"] * 2
    assert [row["bias_application"] for row in receipt["qparams"]] == ["exact_readout_bias"] * 2
    output = integer(*inputs)
    assert output.dtype == torch.int8
    assert torch.equal(output, reference.output)
    record_property("eager-float divergence", int(torch.count_nonzero(output != portable)))


def test_float_result_uses_one_scale_and_seed():
    name = "addmm_alpha_beta"
    model, inputs = _captured(name)
    reference = run_pt2e_integer_reference(model, inputs, numerics="integer_readout")
    integer, receipt = integerize_pt2e(model, inputs, numerics="integer_readout")
    assert receipt["qparams"][0]["readout"] == "single_scale_f32"
    torch.testing.assert_close(integer(*inputs), reference.output, atol=0, rtol=0)
    targets = [str(node.target) for node in integer.graph.nodes if node.op == "call_function"]
    assert "aten._int_mm.default" in targets and "m2m_q.exact_readout.default" in targets
    assert "aten.mul.Tensor" not in targets
    assert not any("int32_seed" in str(node.target) for node in integer.graph.nodes)


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_contraction_case_matches_reference(name):
    model, inputs = _captured(name)
    reference = run_pt2e_integer_reference(model, inputs, numerics="integer_readout")
    integer, receipt = integerize_pt2e(model, inputs, numerics="integer_readout")
    assert receipt["refusals"] == [] and receipt["quantized_contractions_integerized"] == 1
    torch.testing.assert_close(integer(*inputs), reference.output, atol=0, rtol=0, equal_nan=True)


def test_lowered_seed_contraction_and_readout_carry_attributes():
    model, inputs = _model()
    integer, _ = integerize_pt2e(model, inputs, numerics="integer_readout")
    allow_exported_model_train_eval(integer)
    lowered = m2m.convert(integer, inputs, backend="fx_importer")
    assert lowered.ok and opaque_report(lowered.mlir_text) == {}
    text = lowered.mlir_text
    contractions = [line for line in text.splitlines() if "linalg.generic" in line and '"int_matmul"' in line]
    assert len(contractions) == 3
    assert all("prov.quant.scale_numerator" in line and 'prov.quant.arithmetic = "exact"' in line
               for line in contractions)
    readouts = [line for line in text.splitlines() if "linalg.generic" in line and '"requantize"' in line]
    assert len(readouts) == 2
    assert all('prov.quant.rounding = "half_to_even"' in line and 'prov.role = "readout"' in line
               and 'prov.quant.arithmetic = "exact"' in line
               and "prov.quant.quant_min" in line and "prov.quant.quant_max" in line
               and "prov.quant.qmin" not in line and "prov.quant.qmax" not in line
               for line in readouts)
    assert any('prov.quant.relu = "true"' in line for line in readouts)
    assert "prov.quant.output_scale" in text and "prov.quant.rhs_scale" in text
    assert "arith.fptosi" not in text  # exact integer comparisons decide codes


def test_requantized_program_executes_natively_bit_exact(tmp_path):
    model, inputs = _model(per_channel=True)
    integer, _ = integerize_pt2e(model, inputs, numerics="integer_readout")
    allow_exported_model_train_eval(integer)
    assert_native_matches_eager(tmp_path, integer, inputs, rtol=0, atol=0,
                                convert_options={"fully_standard": True})


@pytest.mark.parametrize("numerics", ["float_readout", "integer_readout"])
@pytest.mark.parametrize("name", ["convolution_grouped", "convolution_1d", "convolution_3d", "addmm_runtime_bias"])
def test_integerized_cases_execute_natively(tmp_path, name, numerics):
    model, inputs = _captured(name)
    integer, _ = integerize_pt2e(model, inputs, numerics=numerics)
    allow_exported_model_train_eval(integer)
    assert_native_matches_eager(tmp_path, integer, inputs, rtol=0, atol=0,
                                convert_options={"fully_standard": True})


@pytest.mark.parametrize("zero_point", [0, 3])
def test_quantize_expander_matches_pt2e_natively(tmp_path, zero_point):
    def fn(x):
        q = Q(x, 0.0371, zero_point, -127, 127, torch.int8)
        return q, DQ(q, 0.0371, zero_point, -127, 127, torch.int8)

    x = torch.randn(3, 33, generator=torch.Generator().manual_seed(5)) * 4
    x[0, :4] = torch.tensor([0.0371 * 2.5, -0.0371 * 2.5, 0.0371 * 0.5, 1e9])
    from fractions import Fraction
    import numpy as np
    from m2m.capture.exact_quant import rounded_float
    from tests.native_execution import run_native
    q, dq = run_native(tmp_path, fn, (x,), convert_options={"fully_standard": True})
    expected = np.array([min(127, max(-127, round(Fraction(float(v)) / Fraction(0.0371)) + zero_point))
                         for v in x.reshape(-1)], dtype=np.int8).reshape(x.shape)
    np.testing.assert_array_equal(q, expected)
    expected_dq = np.array([rounded_float(Fraction(int(v) - zero_point) * Fraction(0.0371))
                            for v in expected.reshape(-1)], dtype=np.float32).reshape(x.shape)
    np.testing.assert_array_equal(dq.view(np.uint32), expected_dq.view(np.uint32))


def test_qdq_attributes_record_scale_range_and_rounding():
    def fn(x):
        return qdq(x, 0.0371, low=-127)

    converted = m2m.convert(_Call(fn), (torch.randn(2, 3),), backend="fx_importer")
    text = converted.mlir_text
    from fractions import Fraction
    from m2m.ir.quant_attrs import fp32
    qparams = [op.attributes for op in converted.module.walk() if "prov.quant.scale" in op.attributes]
    assert len(qparams) == 2
    for attrs in qparams:
        assert attrs["prov.quant.scale"].value.data == fp32(0.0371)
        exact = Fraction(int(attrs["prov.quant.scale_numerator"].data[0].data),
                         int(attrs["prov.quant.scale_denominator"].data[0].data))
        assert exact == Fraction(0.0371)
        assert attrs["prov.quant.host_only"].data == "true"
    assert "prov.quant.quant_min = -127" in text and 'prov.quant.rounding = "half_to_even"' in text
    standard = m2m.convert(_Call(fn), (torch.randn(2, 3),), backend="fx_importer", fully_standard=True)
    assert "quant_ext." not in standard.mlir_text and 'prov.quant.arithmetic = "exact"' in standard.mlir_text


class _Call(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 6, 3, padding=1, groups=2)
        self.fc = nn.Linear(6 * 4 * 4, 5)

    def forward(self, x):
        return self.fc(torch.relu(self.conv(x)).flatten(1))


def test_int8_accelerator_profile_through_convert(tmp_path):
    from m2m.capture.torchao_pipeline import QuantizationConfig, static_w8a8_options

    config = QuantizationConfig(scheme="int8_static_act_int8_weight",
                                extra_args={"profile": "int8_accelerator", "activation_quant_min": -127})
    options = static_w8a8_options(config)
    assert options["weight_granularity"] == "per_tensor" and options["integer_numerics"] == "integer_readout"
    torch.manual_seed(0)
    inputs = (torch.randn(2, 4, 4, 4),)
    result = m2m.convert(_Net().eval(), inputs, backend="fx_importer", quantization=config)
    assert result.ok and opaque_report(result.mlir_text) == {}
    receipt = result.integerization_receipt
    assert receipt["numerics"] == "integer_readout" and receipt["refusals"] == []
    assert receipt["fused_requantizations"] == 1  # conv -> relu -> the linear's input quantize
    assert receipt["int32_bias_seeds"] == 0
    assert all(row["activation"]["quant_min"] == -127 for row in receipt["qparams"])
    assert "prov.quant.quant_min = -127" in result.mlir_text


def test_narrow_range_activation_option_is_validated():
    from m2m.capture.torchao_pipeline import QuantizationConfig, static_w8a8_options

    with pytest.raises(ValueError, match="narrow range"):
        static_w8a8_options(QuantizationConfig(scheme="int8_static_act_int8_weight",
                                               extra_args={"activation_quant_min": -100}))
    with pytest.raises(ValueError, match="profile"):
        static_w8a8_options(QuantizationConfig(scheme="int8_static_act_int8_weight",
                                               extra_args={"profile": "gemmini"}))
