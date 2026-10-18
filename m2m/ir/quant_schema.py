"""Canonical, versioned contract for ``prov.quant.*`` attributes.

This stdlib-only module owns the schema; regenerate the vendorable JSON with
``python -m m2m.ir.quant_schema``. Changes to names, types, requiredness or
semantics require a schema version change. Consumers must reject unknown keys.
"""

from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = 1
_QD = ["quantize", "dequantize"]
_RC = ["integer_readout", "contraction"]
_ALL = [*_QD, *_RC]


def _key(type_, meaning, op_kinds, *, required_on=(), required_when=None):
    return {"type": type_, "meaning": meaning, "op_kinds": op_kinds,
            "required": bool(required_on), "required_on": list(required_on),
            "required_when": required_when}


ATTRIBUTES = {
    "quant_min": _key("i64", "Inclusive lower storage-code clamp bound; never an accumulator bound.",
                      _ALL, required_on=_QD, required_when="integer_readout with integer result"),
    "quant_max": _key("i64", "Inclusive upper storage-code clamp bound; permits narrow ranges.",
                      _ALL, required_on=_QD, required_when="integer_readout with integer result"),
    "scale": _key("f32 | array<f32>", "Informational nearest fp32 representation of the effective scale. Never use for exact computation; optional when rational attributes exist.", _ALL),
    "scale_numerator": _key("array<decimal_integer_string>", "Reduced exact rational numerators, paired with scale_denominator. Q/DQ: original scale; contraction: alpha * lhs_scale * rhs_scale; readout: that product / output_scale for integer output, or that product for float output. One entry broadcasts; multiple entries follow the output channel (readout/contraction last axis; Q/DQ axis). Products and ratios are formed exactly from fp32-canonical factors, without rounding the product to fp32.",
                            _ALL, required_when="statically known exact scale; both rational arrays must appear together"),
    "scale_denominator": _key("array<positive_decimal_integer_string>", "Positive reduced exact rational denominators; same length and ordering as scale_numerator. Dynamic Q/DQ scales remain SSA operands and cannot be admitted using absent static provenance.",
                              _ALL, required_when="scale_numerator is present"),
    "zero_point": _key("i64", "Storage zero code, added AFTER round-half-to-even; Q/DQ tensor zero points remain SSA operands when not literal.", _ALL,
                       required_when="integer_readout with integer result"),
    "zero_point_dtype": _key("string", "Source zero-point tensor dtype, e.g. int8, uint8, int32, int64; controls extension from signless storage.", _QD),
    "storage_dtype": _key("string", "Public quantized storage dtype (int8 or uint8); i8 alone does not express signedness.", _ALL,
                          required_on=_QD, required_when="integer_readout with integer result"),
    "axis": _key("i64", "Q/DQ channel axis (negative axes use tensor rank).", _QD,
                 required_when="channel Q/DQ"),
    "granularity": _key("string", "tensor or channel; a single static scale broadcasts.", _ALL, required_on=_QD),
    "lhs_scale": _key("f64", "Source lhs scale factor for diagnostics; exact consumers use the rational pair. Accelerator factors must be fp32-representable.", _RC),
    "rhs_scale": _key("f64 | array<f64>", "Source rhs scale factor(s), in output-channel order; diagnostic, never a rounded substitute for the rational pair.", _RC),
    "output_scale": _key("f64", "Output Q scale factor, diagnostic; it is already divided into the readout rational pair. Do not divide by it a second time.", _RC),
    "alpha": _key("f64", "Exact binary scalar multiplier of the contraction, included in the rational pair.", _RC),
    "beta": _key("f64", "Exact binary multiplier of the explicit bias operand. The scale pair describes the accumulator coefficient only: an unfused bias operand is additional exact math, not an integer seed.", ["integer_readout"]),
    "arithmetic": _key("string", "exact means rational/integer math until the required final rounding; eager_float_reassociated makes no exactness claim.", _ALL,
                       required_on=["integer_readout", "contraction"]),
    "rounding": _key("string", "half_to_even (RNE), before adding zero point and clamping; float results round once to their IEEE result type.", _ALL,
                     required_on=["quantize"], required_when="integer_readout with integer result"),
    "numerics": _key("string", "integer_readout (exact) or float_readout (legacy approximate and host-only). This is not device-placement permission.", _RC,
                     required_on=_RC),
    "saturate": _key('string("true" | "false")', "Whether final integer codes clamp to quant_min/max; never permits intermediate saturation.", ["integer_readout"],
                     required_when="integer_readout with integer result"),
    "relu": _key('string("true" | "false")', "Exact real ReLU before final output quantization.", ["integer_readout"],
                 required_when="integer_readout with integer result"),
    "accumulator": _key("string", "i32, the full-width exact integer accumulator. Device readout defaults to full i32, with exact requantization performed separately.", _RC, required_on=_RC),
    "acc_scale": _key("f64 | array<f64>", "Legacy rounded product, informational only; only emitted with float_readout and host_only=true.", ["contraction"]),
    "gradient": _key("string", "Backward contraction output: input or weight.", ["contraction"]),
    "host_only": _key('string("true" | "false")', "true forbids accelerator placement: legacy approximate numerics or non-fp32 source scales. Absence/false is not admission or an accumulator-overflow proof.", _ALL,
                     required_when="float_readout, or a source scale is not fp32-representable"),
}


def schema() -> dict:
    """Return a detached JSON-compatible schema, using fully qualified keys."""
    return json.loads(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "namespace": "prov.quant.",
        "unknown_keys": "reject",
        "op_kinds": {
            "quantize": "Q operation, including expanded standard-dialect bodies",
            "dequantize": "DQ operation, including expanded standard-dialect bodies",
            "contraction": "Integer matmul/bmm, including seeded and backward contractions",
            "integer_readout": "Exact accumulator readout, including legacy m2m_q readout operators",
        },
        "attributes": {f"prov.quant.{key}": value for key, value in sorted(ATTRIBUTES.items())},
    }))


def schema_json() -> str:
    return json.dumps(schema(), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    Path(__file__).with_suffix(".json").write_text(schema_json())
