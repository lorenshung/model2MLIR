"""PT2E integerization of 2-D convolution_backward: i8 x i8 -> i32 GEMMs and an int32 sum.

grad_input (backward GEMM columns accumulated tap by tap), grad_weight (gradient x
im2col) and grad_bias (int32 sum) must equal the independent integer reference
bit for bit, stay within float rounding of portable PT2E, and lower to
``int_matmul`` contractions with no gather loop.
"""

from __future__ import annotations

import pytest
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401

import m2m
from m2m.capture.pt2e_compat import allow_exported_model_train_eval
from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
from m2m.capture.pt2e_integerize import integerize_pt2e
from m2m.coverage import opaque_report
from tests.native_execution import assert_native_matches_eager

A = torch.ops.aten
Q = torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ = torch.ops.quantized_decomposed.dequantize_per_tensor.default
GATHER = ("scf.for", "scf.while", "tensor.extract ", "linalg.index")


def qdq(value, scale):
    return DQ(Q(value, scale, 0, -128, 127, torch.int8), scale, 0, -128, 127, torch.int8)


class Backward(torch.nn.Module):
    def __init__(self, geometry, mask):
        super().__init__()
        self.geometry, self.mask = geometry, mask

    def forward(self, go, x, w):
        stride, padding, dilation, groups = self.geometry
        out = A.convolution_backward.default(qdq(go, 0.03), qdq(x, 0.05), qdq(w, 0.02), [w.shape[0]], stride,
                                             padding, dilation, False, [0, 0], groups, self.mask)
        return tuple(out[i] for i in range(3) if self.mask[i])


CASES = {
    "s1p1_TTT": (([1, 1], [1, 1], [1, 1], 1), [True, True, True], (1, 3, 6, 6), (4, 3, 3, 3)),
    "strided_dilated_grouped_TTF": (([2, 1], [1, 0], [1, 2], 2), [True, True, False], (2, 4, 7, 6), (6, 2, 3, 3)),
    "depthwise_TFT": (([1, 1], [1, 1], [1, 1], 4), [True, False, True], (1, 4, 5, 5), (4, 1, 3, 3)),
    "weight_only_FTF": (([2, 2], [0, 0], [1, 1], 1), [False, True, False], (2, 3, 8, 8), (5, 3, 2, 2)),
}


def _case(name):
    geometry, mask, x_shape, w_shape = CASES[name]
    generator = torch.Generator().manual_seed(len(name))
    x, w = torch.randn(x_shape, generator=generator), torch.randn(w_shape, generator=generator)
    forward = A.convolution.default(x, w, None, *geometry[:3], False, [0, 0], geometry[3])
    inputs = (torch.randn(forward.shape, generator=generator), x, w)
    return Backward(geometry, mask), inputs


@pytest.mark.parametrize("numerics", ["float_readout", "integer_readout"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_convolution_backward_integerizes_exactly(name, numerics):
    module, inputs = _case(name)
    model = torch.export.export(module, inputs).module()
    portable = model(*inputs)
    reference = run_pt2e_integer_reference(model, inputs, numerics=numerics)
    integer, receipt = integerize_pt2e(model, inputs, numerics=numerics)
    assert receipt["refusals"] == [] and receipt["quantized_contractions_integerized"] == 1
    assert receipt["source_ops"] == {"aten.convolution_backward.default": {"seen": 1, "integerized": 1}}
    output = integer(*inputs)
    for got, want, portable_value in zip(output, reference.output, portable):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
        torch.testing.assert_close(got, portable_value, atol=1e-5, rtol=1e-5)
    allow_exported_model_train_eval(integer)
    lowered = m2m.convert(integer, inputs, backend="fx_importer")
    assert lowered.ok and opaque_report(lowered.mlir_text) == {}
    assert not [token for token in GATHER if token in lowered.mlir_text]
    contractions = [line for line in lowered.mlir_text.splitlines()
                    if "linalg.generic" in line and '"int_matmul"' in line]
    assert len(contractions) == receipt["integer_mm_emitted"]
    assert all("xi8>, tensor<" in line and "xi32>)" in line for line in contractions)


def test_integerized_convolution_backward_executes_natively(tmp_path):
    module, inputs = _case("strided_dilated_grouped_TTF")
    integer, _ = integerize_pt2e(torch.export.export(module, inputs).module(), inputs, numerics="integer_readout")
    allow_exported_model_train_eval(integer)
    assert_native_matches_eager(tmp_path, integer, inputs, rtol=0, atol=0, convert_options={"fully_standard": True})


def test_per_channel_weights_are_refused():
    DQC = torch.ops.quantized_decomposed.dequantize_per_channel.default

    class PerChannel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("w", torch.ones(4, 3, 3, 3, dtype=torch.int8))
            self.register_buffer("s", torch.full((4,), 0.02))
            self.register_buffer("z", torch.zeros(4, dtype=torch.int64))

        def forward(self, go, x):
            w = DQC(self.w, self.s, self.z, 0, -127, 127, torch.int8)
            return A.convolution_backward.default(qdq(go, 0.03), qdq(x, 0.05), w, None, [1, 1], [1, 1], [1, 1],
                                                  False, [0, 0], 1, [True, False, False])[0]

    inputs = (torch.randn(1, 4, 5, 5), torch.randn(1, 3, 5, 5))
    _, receipt = integerize_pt2e(torch.export.export(PerChannel(), inputs).module(), inputs)
    assert receipt["quantized_contractions_integerized"] == 0 and "per-tensor" in receipt["refusals"][0]["reason"]


def test_int8_accelerator_profile_quantizes_and_integerizes_convolution_backward():
    from m2m.capture.torchao_pipeline import QuantizationConfig

    class Plain(torch.nn.Module):
        def forward(self, go, x, w):
            out = A.convolution_backward.default(go, x, w, [4], [1, 1], [1, 1], [1, 1], False, [0, 0], 1,
                                                 [True, True, True])
            return out[0], out[1], out[2]

    generator = torch.Generator().manual_seed(4)
    inputs = (torch.randn(1, 4, 6, 6, generator=generator), torch.randn(1, 3, 6, 6, generator=generator),
              torch.randn(4, 3, 3, 3, generator=generator))
    config = QuantizationConfig(scheme="int8_static_act_int8_weight", extra_args={"profile": "int8_accelerator"})
    result = m2m.convert(Plain().eval(), inputs, backend="fx_importer", quantization=config)
    assert result.ok and opaque_report(result.mlir_text) == {}
    receipt = result.integerization_receipt
    assert receipt["source_ops"] == {"aten.convolution_backward.default": {"seen": 1, "integerized": 1}}
    assert receipt["integer_mm_emitted"] == 2
    assert not [token for token in GATHER if token in result.mlir_text]


CAPSULE = ("/scratch2/loren/gemmini-coverage/PyTorchFullCoverageKernelSet/kernels/"
           "aten__convolution_backward__default/capsule.pytorch.py")


def _canonical_case():
    """The corpus case: grad [1,2,3,3], input [1,1,5,5], weight [2,1,3,3], all masks."""
    import importlib.util
    import os

    if os.path.exists(CAPSULE):
        spec = importlib.util.spec_from_file_location("canonical_convolution_backward", CAPSULE)
        capsule = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(capsule)
        return capsule.get_model_and_inputs()

    class Model(torch.nn.Module):  # the same case, inlined when the corpus is absent
        def forward(self, go, x, w):
            return A.convolution_backward.default(go, x, w, [2], [1, 1], [0, 0], [1, 1], False, [0, 0], 1,
                                                  [True, True, True])

    x = (torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5) - 9) / 12
    w = torch.linspace(-0.75, 1.25, 18).reshape(2, 1, 3, 3)
    return Model(), (torch.ones(1, 2, 3, 3), x, w)


@pytest.mark.parametrize("traced_selection", [False, True], ids=["getitem", "trace_instrumented_getitem"])
@pytest.mark.parametrize("numerics", ["float_readout", "integer_readout"])
def test_canonical_case_integerizes_with_traced_tuple_selection(tmp_path, numerics, traced_selection):
    """Regression: a capture-traced graph selects the three result slots through
    trace.tuple_selection_trace_program's instrumented getitem, which integerization
    refused as "a consumer other than its result slots"."""
    import operator

    from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization
    from m2m.capture.trace import tuple_selection_trace_program

    model, inputs = _canonical_case()
    config = QuantizationConfig(scheme="int8_static_act_int8_weight",
                                extra_args={"contractions": "all", "weight_granularity": "per_tensor"})
    quantized = apply_quantization(model.eval(), config, example_inputs=inputs)
    if traced_selection:
        instrumented = tuple_selection_trace_program(torch.export.export(model.eval(), inputs))
        selection = next(node.target for node in instrumented.graph_module.graph.nodes
                         if getattr(node.target, "_m2m_tuple_selection", False))
        for node in quantized.graph.nodes:
            if node.target is operator.getitem:
                node.target = selection
        quantized.recompile()
    portable = quantized(*inputs)
    reference = run_pt2e_integer_reference(quantized, inputs, numerics=numerics)
    integer, receipt = integerize_pt2e(quantized, inputs, numerics=numerics)
    assert receipt["refusals"] == [] and receipt["quantized_contractions_integerized"] == 1
    assert receipt["integer_mm_emitted"] == 2
    output = integer(*inputs)
    assert len(output) == 3
    for got, want, portable_value in zip(output, reference.output, portable):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
        torch.testing.assert_close(got, portable_value, atol=1e-5, rtol=1e-5)
    allow_exported_model_train_eval(integer)
    lowered = m2m.convert(integer, inputs, backend="fx_importer")
    assert lowered.ok and opaque_report(lowered.mlir_text) == {}
    assert not [token for token in GATHER if token in lowered.mlir_text]
    assert_native_matches_eager(tmp_path, integer, inputs, rtol=0, atol=0, convert_options={"fully_standard": True})


@pytest.mark.parametrize("mask", __import__("itertools").product((False, True), repeat=3))
@pytest.mark.parametrize("numerics", ["float_readout", "integer_readout"])
def test_every_output_mask_recompiles(mask, numerics):
    # Trace manually so even the all-false mask retains the dead source node.
    # Bias-only produces no GEMM, which used to skip recompile() entirely.
    import operator
    from torch.fx import Graph, GraphModule
    graph = Graph()
    go, x, w = [graph.placeholder(name) for name in ("go", "x", "w")]
    operands = []
    for value, scale in zip((go, x, w), (0.03, 0.05, 0.02)):
        q = graph.call_function(Q, (value, scale, 0, -128, 127, torch.int8))
        operands.append(graph.call_function(DQ, (q, scale, 0, -128, 127, torch.int8)))
    backward = graph.call_function(A.convolution_backward.default,
        (*operands, [2], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, list(mask)))
    graph.output(tuple(graph.call_function(operator.getitem, (backward, i)) for i in range(3) if mask[i]))
    model = GraphModule({}, graph)
    inputs = (torch.randn(1, 2, 2, 2), torch.randn(1, 1, 3, 3), torch.randn(2, 1, 2, 2))
    reference = run_pt2e_integer_reference(model, inputs, numerics=numerics)
    integer, receipt = integerize_pt2e(model, inputs, numerics=numerics)
    assert receipt["quantized_contractions_integerized"] == 1
    assert not any(n.target == A.convolution_backward.default for n in integer.graph.nodes)
    assert "convolution_backward" not in integer.code
    output = integer(*inputs)
    assert len(output) == sum(mask)
    for got, want in zip(output, reference.output):
        assert torch.equal(got, want)
