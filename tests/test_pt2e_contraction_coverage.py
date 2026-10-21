"""PT2E integerization covers addmm, aten.convolution (1-D/2-D/3-D) and grouped convs.

Each case is a frozen-weight Q/DQ graph. The integerized graph must equal the
independent integer reference exactly and lower to i8 x i8 -> i32 contractions
with no opaque call. Eager-float divergence is informational only.
"""

from __future__ import annotations

import re

import pytest
import torch
from torch import nn

import m2m
from m2m.capture.pt2e_compat import allow_exported_model_train_eval
from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
from m2m.capture.pt2e_integerize import integerize_pt2e
from m2m.coverage import opaque_report
from tests.native_execution import DQ, qdq

A = torch.ops.aten


def weight(module, name):
    return DQ(getattr(module, name), 0.02, 0, -127, 127, torch.int8)


class Case(nn.Module):
    def __init__(self, weight_shape, bias_shape, call):
        super().__init__()
        generator = torch.Generator().manual_seed(sum(weight_shape))
        self.register_buffer("w", torch.randint(-90, 91, weight_shape, generator=generator, dtype=torch.int8))
        if bias_shape is not None:
            self.register_buffer("b", torch.randn(bias_shape, generator=generator))
        self.call = call

    def forward(self, *inputs):
        return self.call(self, *inputs)


def _conv(stride, padding, dilation, groups, *, target="convolution"):
    def call(module, x):
        bias = getattr(module, "b", None)
        dx = qdq(x, 0.05)
        if target == "convolution":
            rank = x.ndim - 2
            return A.convolution.default(dx, weight(module, "w"), bias, stride, padding, dilation, False,
                                         [0] * rank, groups)
        return getattr(A, target).default(dx, weight(module, "w"), bias, stride, padding, dilation, groups)
    return call


CASES = {
    "convolution_2d_bias": ((6, 4, 3, 3), 6, _conv([1, 1], [1, 1], [1, 1], 1), [(1, 4, 6, 6)]),
    "convolution_grouped": ((6, 2, 3, 3), 6, _conv([2, 1], [1, 0], [1, 2], 2), [(2, 4, 7, 7)]),
    "convolution_depthwise": ((4, 1, 3, 3), None, _conv([1, 1], [1, 1], [1, 1], 4), [(1, 4, 5, 5)]),
    "conv2d_grouped": ((6, 2, 3, 3), 6, _conv([1, 1], [1, 1], [1, 1], 2, target="conv2d"), [(1, 4, 5, 5)]),
    "convolution_1d": ((5, 3, 3), 5, _conv([2], [1], [1], 1), [(2, 3, 11)]),
    "conv1d_grouped": ((4, 1, 2), None, _conv([1], [0], [2], 2, target="conv1d"), [(1, 2, 9)]),
    "convolution_3d": ((3, 2, 2, 2, 2), 3, _conv([1, 2, 1], [1, 0, 1], [1, 1, 1], 1), [(1, 2, 4, 5, 4)]),
    "conv3d": ((2, 2, 1, 2, 3), 2, _conv([1, 1, 1], [0, 1, 1], [1, 1, 1], 1, target="conv3d"), [(1, 2, 3, 4, 5)]),
    "addmm_alpha_beta": ((8, 5), 5, lambda m, x: A.addmm.default(m.b, qdq(x, 0.05), weight(m, "w"),
                                                                    beta=0.5, alpha=2.0), [(3, 8)]),
    "addmm_full_bias": ((8, 5), (3, 5), lambda m, x: A.addmm.default(m.b, qdq(x, 0.05), weight(m, "w")), [(3, 8)]),
    "addmm_runtime_bias": ((8, 5), None, lambda m, c, x: A.addmm.default(c, qdq(x, 0.05), weight(m, "w")),
                           [(1, 5), (3, 8)]),
    "addmm_beta0_ignores_nan": ((8, 5), None, lambda m, c, x: A.addmm.default(c * float("nan"), qdq(x, 0.05),
                                                                               weight(m, "w"), beta=0.0),
                                [(3, 5), (3, 8)]),
}


def _captured(name):
    weight_shape, bias_shape, call, shapes = CASES[name]
    generator = torch.Generator().manual_seed(7)
    inputs = tuple(torch.randn(shape, generator=generator) for shape in shapes)
    model = torch.export.export(Case(weight_shape, bias_shape, call).eval(), inputs).module()
    return model, inputs


@pytest.mark.parametrize("name", sorted(CASES))
def test_contraction_integerizes_matches_reference_and_lowers(name, record_property):
    model, inputs = _captured(name)
    portable = model(*inputs)
    reference = run_pt2e_integer_reference(model, inputs, numerics="integer_readout", expected_contractions=1)
    integer, receipt = integerize_pt2e(model, inputs, numerics="integer_readout")
    assert receipt["refusals"] == [] and receipt["quantized_contractions_remaining"] == 0
    assert receipt["quantized_contractions_integerized"] == 1
    source = next(iter(receipt["source_ops"]))
    assert receipt["source_ops"][source] == {"seen": 1, "integerized": 1}
    kind = "matmul" if "addmm" in source else "conv2d"
    assert receipt["quantized_by_kind"][kind] == {"seen": 1, "integerized": 1, "remaining": 0}
    assert receipt["precision_decisions"][0]["source_op"] == source
    groups = receipt["qparams"][0]["groups"]
    assert receipt["integer_mm_emitted"] == groups
    output = integer(*inputs)
    torch.testing.assert_close(output, reference.output, atol=0, rtol=0, equal_nan=True)
    record_property("eager-float divergence", int(torch.count_nonzero(output != portable)))

    allow_exported_model_train_eval(integer)
    lowered = m2m.convert(integer, inputs, backend="fx_importer")
    assert lowered.ok and opaque_report(lowered.mlir_text) == {}
    contractions = [line for line in lowered.mlir_text.splitlines()
                    if "linalg.generic" in line and 'prov.op = "int_matmul"' in line]
    assert len(contractions) == groups
    assert all(re.search(r"ins\(%\d+, %\d+ : tensor<\d+x\d+xi8>, tensor<\d+x\d+xi8>\) outs\(%\d+ : tensor<\d+x\d+xi32>\)",
                         line) for line in contractions)


def test_transposed_convolution_is_refused_as_unsupported():
    class Transposed(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("w", torch.ones((4, 3, 3, 3), dtype=torch.int8))

        def forward(self, x):
            return A.convolution.default(qdq(x, 0.05), weight(self, "w"), None, [2, 2], [1, 1], [1, 1], True,
                                         [1, 1], 1)

    inputs = (torch.randn(1, 4, 5, 5),)
    model = torch.export.export(Transposed().eval(), inputs).module()
    _, receipt = integerize_pt2e(model, inputs)
    assert receipt["quantized_by_kind"]["unsupported"]["seen"] == 1
    assert "transposed" in receipt["refusals"][0]["reason"]


def test_too_many_groups_are_refused():
    model, inputs = None, (torch.randn(1, 300, 3, 3),)

    class Wide(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("w", torch.ones((300, 1, 1, 1), dtype=torch.int8))

        def forward(self, x):
            return A.conv2d.default(qdq(x, 0.05), weight(self, "w"), None, [1, 1], [0, 0], [1, 1], 300)

    model = torch.export.export(Wide().eval(), inputs).module()
    _, receipt = integerize_pt2e(model, inputs)
    assert receipt["quantized_contractions_integerized"] == 0
    assert ">256" in receipt["refusals"][0]["reason"]


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 6, 3, padding=1, groups=2)
        self.fc = nn.Linear(6 * 4 * 4, 5)

    def forward(self, x):
        return self.fc(torch.relu(self.conv(x)).flatten(1))


def test_convert_integerizes_static_w8a8_and_records_the_receipt():
    from m2m.capture.torchao_pipeline import QuantizationConfig

    torch.manual_seed(0)
    inputs = (torch.randn(2, 4, 4, 4),)
    config = QuantizationConfig(scheme="int8_static_act_int8_weight",
                                extra_args={"integerize": True, "weight_granularity": "per_tensor"})
    result = m2m.convert(_Net().eval(), inputs, backend="fx_importer", quantization=config)
    assert result.ok and opaque_report(result.mlir_text) == {}
    receipt = result.integerization_receipt
    assert receipt["quantized_contractions_integerized"] == 2 and receipt["refusals"] == []
    assert receipt["integer_mm_emitted"] == 3  # two conv groups and one linear
    assert "prov.integerization" in result.mlir_text
    contractions = [line for line in result.mlir_text.splitlines()
                    if "linalg.generic" in line and 'prov.op = "int_matmul"' in line]
    assert len(contractions) == 3 and all("xi8>, tensor<" in line for line in contractions)
    assert "linalg.matmul" not in result.mlir_text  # no float contraction remains


def test_convert_integerization_fails_closed_and_refuses_preapplied():
    from m2m.capture.torchao_pipeline import QuantizationConfig

    class Big(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(140000, 2)

        def forward(self, x):
            return self.fc(x)

    config = QuantizationConfig(scheme="int8_static_act_int8_weight", extra_args={"integerize": True},
                                calibration_samples=1)
    with pytest.raises(ValueError, match="integerization is incomplete"):
        m2m.convert(Big().eval(), (torch.randn(1, 140000),), backend="fx_importer", quantization=config)
    with pytest.raises(ValueError, match="preapplied"):
        m2m.convert(_Net().eval(), (torch.randn(1, 4, 4, 4),), backend="fx_importer", quantization=config,
                    quantization_preapplied=True)


def test_all_contractions_option_quantizes_addmm_and_bmm():
    from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization

    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(6, 5))
            self.b = nn.Parameter(torch.randn(5))

        def forward(self, x, y):
            z = torch.addmm(self.b, x, self.w)
            return torch.bmm(z.unsqueeze(0), y)

    inputs = (torch.randn(3, 6), torch.randn(1, 5, 4))
    for contractions, expected in (("conv_linear", None), ("all", 2)):
        config = QuantizationConfig(scheme="int8_static_act_int8_weight",
                                    extra_args={"contractions": contractions})
        if expected is None:
            with pytest.raises(ValueError, match="no supported"):
                apply_quantization(Mixed().eval(), config, example_inputs=inputs)
            continue
        quantized = apply_quantization(Mixed().eval(), config, example_inputs=inputs)
        assert quantized._m2m_quantization_stats["annotated_contractions"] == expected
        integer, receipt = integerize_pt2e(quantized, inputs)
        assert receipt["quantized_contractions_integerized"] == expected and receipt["refusals"] == []
        assert set(receipt["source_ops"]) == {"aten.addmm.default", "aten.bmm.default"}
