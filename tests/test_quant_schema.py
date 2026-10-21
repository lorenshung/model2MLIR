"""Consumer contract tests using real captures, not handwritten MLIR bodies."""

from fractions import Fraction
import json
from pathlib import Path

import pytest
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401
from xdsl.dialects.builtin import ArrayAttr, Float32Type, Float64Type, FloatAttr, IntegerAttr, StringAttr

import m2m
from m2m.capture.pt2e_compat import allow_exported_model_train_eval
from m2m.capture.pt2e_integerize import integerize_pt2e
from m2m.ir import quant_schema
from m2m.ir.quant_attrs import fp32, scale_attributes


def test_fraction_scale_metadata_rounds_directly_to_binary32():
    value = Fraction("1.00000005960464477539062500000001")
    for signed in (value, -value):
        attrs = scale_attributes([signed])
        assert attrs["prov.quant.scale"].value.data == (1 if signed > 0 else -1) * (1 + 2.0**-23)
        assert int(attrs["prov.quant.scale_numerator"].data[0].data) == signed.numerator
        assert int(attrs["prov.quant.scale_denominator"].data[0].data) == signed.denominator
        assert attrs["prov.quant.host_only"].data == "true"
    # Huge rational coefficients omit optional fp32 metadata and stay host-only.
    attrs = scale_attributes([Fraction(2) ** 2048])
    assert "prov.quant.scale" not in attrs
    assert attrs["prov.quant.host_only"].data == "true"


def test_json_is_canonical_version_1():
    text = Path(quant_schema.__file__).with_suffix(".json").read_text()
    assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"
    assert quant_schema.SCHEMA_VERSION == 1


def _check_type(value, type_):
    if type_.startswith("array<") and isinstance(value, ArrayAttr):
        assert len(value) > 0
        for item in value:
            _check_type(item, type_[6:-1])
    elif " | " in type_ and not type_.startswith("string("):
        if isinstance(value, ArrayAttr):
            _check_type(value, type_.split(" | ")[1])
        else:
            _check_type(value, type_.split(" | ")[0])
    elif type_ in ("f32", "f64"):
        assert isinstance(value, FloatAttr)
        assert isinstance(value.type, Float32Type if type_ == "f32" else Float64Type)
    elif type_ == "i64":
        assert isinstance(value, IntegerAttr) and value.type.width.data == 64
    elif "decimal_integer_string" in type_:
        assert isinstance(value, StringAttr) and str(int(value.data)) == value.data
        if type_.startswith("positive"):
            assert int(value.data) > 0
    else:
        assert isinstance(value, StringAttr)
        if type_.startswith("string("):
            assert value.data in ("true", "false")


def _check_capture(result, seen):
    assert result.ok, result.diagnostics
    schema = quant_schema.schema()["attributes"]
    for op in result.module.walk():
        attrs = {k: v for k, v in op.attributes.items() if k.startswith("prov.quant.")}
        for key, value in attrs.items():
            assert key in schema, (op.name, key)
            seen.add(key)
            _check_type(value, schema[key]["type"])
        if "prov.quant.scale_numerator" in attrs:
            ns, ds = (attrs[f"prov.quant.scale_{suffix}"] for suffix in ("numerator", "denominator"))
            assert len(ns) == len(ds)
            for n, d in zip(ns, ds):
                ratio = Fraction(int(n.data), int(d.data))
                assert ratio.numerator == int(n.data) and ratio.denominator == int(d.data)
        if not attrs:
            continue
        source = getattr(op.attributes.get("prov.aten"), "data", "")
        role = getattr(op.attributes.get("prov.role"), "data", "")
        if "dequantize_per_" in source:
            kind = "dequantize"
        elif "quantize_per_" in source or source in ("m2m_q.quantize.default", "m2m_q.quantize_channel.default"):
            kind = "quantize"
        elif role == "readout":
            kind = "integer_readout"
        else:
            kind = "contraction"
        if (kind in ("contraction", "integer_readout") and op.name == "linalg.generic"
                and "prov.quant.scale_numerator" in attrs):
            channels = op.results[0].type.get_shape()[-1]
            assert len(attrs["prov.quant.scale_numerator"]) in (1, channels)
            if "prov.quant.rhs_scale" in attrs and isinstance(attrs["prov.quant.rhs_scale"], ArrayAttr):
                assert len(attrs["prov.quant.rhs_scale"]) in (1, channels)
        for key in attrs:
            assert kind in schema[key]["op_kinds"], (kind, key, source)
        for key, spec in schema.items():
            if kind in spec["required_on"]:
                assert key in attrs, (kind, key, source)


def test_every_emitted_key_and_type_in_real_captures():
    from tests.test_pt2e_integer_readout import _model
    from tests.test_pt2e_conv_backward import _case
    seen = set()
    for numerics in ("integer_readout", "float_readout"):
        for channel in (False, True):
            model, inputs = _model(channel)
            integerize_pt2e(model, inputs, numerics=numerics)
            allow_exported_model_train_eval(model)
            for standard in (False, True):
                _check_capture(m2m.convert(model, inputs, backend="fx_importer", fully_standard=standard), seen)
        model, inputs = _case("depthwise_TFT")
        model = torch.export.export(model, inputs).module()
        integerize_pt2e(model, inputs, numerics=numerics)
        allow_exported_model_train_eval(model)
        _check_capture(m2m.convert(model, inputs, backend="fx_importer"), seen)

    class QDQ(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.dtype = dtype
            self.register_buffer("scales", torch.tensor([0.03, 0.125], dtype=torch.float64))
            self.register_buffer("zeros", torch.tensor([1, 3], dtype=dtype))

        def forward(self, x):
            qd = torch.ops.quantized_decomposed
            low, high = (1, 254) if self.dtype == torch.uint8 else (-127, 127)
            a = qd.quantize_per_tensor.default(x, 0.03, 3, low, high, self.dtype)
            b = qd.dequantize_per_tensor.default(a, 0.03, 3, low, high, self.dtype)
            c = qd.quantize_per_channel.default(x, self.scales, self.zeros, 0, low, high, self.dtype)
            d = qd.dequantize_per_channel.default(c, self.scales, self.zeros, 0, low, high, self.dtype)
            return a, b, c, d

    for dtype in (torch.int8, torch.uint8):
        for standard in (False, True):
            _check_capture(m2m.convert(QDQ(dtype), (torch.randn(2, 5),), backend="fx_importer",
                                       fully_standard=standard), seen)
    # Legacy public helpers obey the same contract, in addition to real PT2E captures.
    from m2m.capture import integer_readout as ops
    class Legacy(torch.nn.Module):
        def forward(self, x):
            return ops.requantize(x, [0.5], 3, -127, 127), ops.acc_readout(x, [0.5])
    _check_capture(m2m.convert(Legacy(), (torch.ones(2, 3, dtype=torch.int32),), backend="fx_importer"), seen)
    # Every listed key is exercised, including legacy diagnostics and backward metadata.
    assert seen == set(quant_schema.schema()["attributes"])


def test_unknown_custom_quant_key_is_rejected():
    from m2m.ir.import_fx import _stamp_custom_metadata
    from xdsl.dialects.builtin import ModuleOp
    with pytest.raises(ValueError, match="unknown prov.quant keys"):
        _stamp_custom_metadata(ModuleOp([]), {"m2m_quant": {"qmin": -128}})


@pytest.mark.parametrize("override", [
    {"integer_numerics": "float_readout"}, {"integerize": False}, {"integerize_strict": False},
])
def test_accelerator_profile_refuses_approximate_or_partial_integerization(override):
    from m2m.capture.torchao_pipeline import QuantizationConfig, static_w8a8_options
    with pytest.raises(ValueError, match="int8_accelerator requires"):
        static_w8a8_options(QuantizationConfig(scheme="int8_static_act_int8_weight",
                                              extra_args={"profile": "int8_accelerator", **override}))


def test_capture_and_reference_defaults_are_exact_and_legacy_is_host_only():
    from m2m.api import integerize_static_w8a8
    from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
    from tests.test_pt2e_contraction_coverage import _captured
    from m2m.capture.torchao_pipeline import QuantizationConfig, static_w8a8_options
    assert static_w8a8_options(QuantizationConfig(scheme="int8_static_act_int8_weight"))["integer_numerics"] == "integer_readout"
    for integerize in (integerize_pt2e, integerize_static_w8a8):
        model, inputs = _captured("addmm_alpha_beta")
        reference = run_pt2e_integer_reference(model, inputs)
        integer, receipt = integerize(model, inputs)
        assert receipt["numerics"] == "integer_readout" and not receipt["host_only"]
        assert torch.equal(integer(*inputs), reference.output)
        assert any(str(n.target) == "m2m_q.exact_readout.default" for n in integer.graph.nodes)
    model, inputs = _captured("addmm_alpha_beta")
    integer, receipt = integerize_pt2e(model, inputs, numerics="float_readout")
    assert receipt["host_only"] and receipt["arithmetic"] == "eager_float_reassociated"
    allow_exported_model_train_eval(integer)
    result = m2m.convert(integer, inputs, backend="fx_importer")
    contractions = [op for op in result.module.walk() if "prov.quant.numerics" in op.attributes]
    assert contractions
    assert all(op.attributes["prov.quant.host_only"].data == "true" for op in contractions)


def test_canonical_scale_change_is_recorded_and_reference_is_bit_exact(tmp_path):
    from tests.test_pt2e_integer_readout import _model
    from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
    from tests.native_execution import run_native
    import numpy as np
    model, inputs = _model(per_channel=True)
    reference = run_pt2e_integer_reference(model, inputs)
    integer, receipt = integerize_pt2e(model, inputs)
    changes = receipt["scale_canonicalization"]
    assert changes["dtype"] == "float32" and changes["semantics_changed"]
    assert all(fp32(v) == v for row in changes["changes"]
               for v in (row["canonical"] if isinstance(row["canonical"], list) else [row["canonical"]]))
    assert torch.equal(integer(*inputs), reference.output)
    allow_exported_model_train_eval(integer)
    result = m2m.convert(integer, inputs, backend="fx_importer", fully_standard=True)
    for op in result.module.walk():
        assert getattr(op.attributes.get("prov.quant.host_only"), "data", "false") != "true"
    native, = run_native(tmp_path, integer, inputs, convert_options={"fully_standard": True})
    np.testing.assert_array_equal(native, reference.output.numpy())


def test_frozen_channel_canonicalization_preserves_unrelated_uses_and_is_idempotent():
    from m2m.capture.quant_scales import canonicalize_pt2e_scales

    class Channel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scales", torch.tensor([0.03, 0.0371], dtype=torch.float64))
            self.register_buffer("zeros", torch.zeros(2, dtype=torch.int64))

        def forward(self, x):
            qd = torch.ops.quantized_decomposed
            q = qd.quantize_per_channel.default(x, self.scales, self.zeros, 0, -127, 127, torch.int8)
            return qd.dequantize_per_channel.default(q, self.scales, self.zeros, 0, -127, 127, torch.int8), self.scales

    inputs = (torch.randn(2, 9),)
    model = torch.export.export(Channel(), inputs).module()
    original = model.scales.clone()
    changes = canonicalize_pt2e_scales(model)
    assert len(changes) == 2
    assert canonicalize_pt2e_scales(model) == changes
    assert torch.equal(model(*inputs)[1], original)
    targets = [node.args[1].target for node in model.graph.nodes
               if "quantize_per_channel" in str(node.target)]
    assert len(set(targets)) == 1
    assert torch.equal(getattr(model, targets[0]), original.float())
    allow_exported_model_train_eval(model)
    result = m2m.convert(model, inputs, backend="fx_importer")
    assert result.ok, result.diagnostics
    for op in result.module.walk():
        if "prov.quant.scale_numerator" in op.attributes:
            ratios = [Fraction(int(n.data), int(d.data)) for n, d in zip(
                op.attributes["prov.quant.scale_numerator"], op.attributes["prov.quant.scale_denominator"])]
            assert ratios == [Fraction(float(v)) for v in original.float()]
            assert "prov.quant.host_only" not in op.attributes


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_nonrepresentable_fp32_scales_are_refused(scale):
    from m2m.capture.quant_scales import canonicalize_pt2e_scales
    class Q(torch.nn.Module):
        def forward(self, x):
            return torch.ops.quantized_decomposed.quantize_per_tensor.default(x, scale, 0, -128, 127, torch.int8)
    model = torch.export.export(Q(), (torch.zeros(1),)).module()
    with pytest.raises(ValueError, match="finite positive fp32"):
        canonicalize_pt2e_scales(model)
