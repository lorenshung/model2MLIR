# Quantization contract, schema version 1

The canonical schema is `m2m/ir/quant_schema.json`, included in installed
distributions; `m2m.ir.quant_schema` loads it. Version 1 covers every emitted
`prov.quant.*` key, including legacy diagnostics and backward contractions.
Changes to a key's type, meaning, or requiredness require a version change.
Consumers must reject unknown keys and unsupported versions. The JSON carries
the MLIR attribute encoding, applicable op kinds, unconditional requirements,
and conditional requirements for each key. Boolean flags currently use string
attributes `"true"` and `"false"`.

Use `scale_numerator` / `scale_denominator` for exact arithmetic. Both are arrays
of decimal integer strings, allowing arbitrary precision. Numerators and positive
denominators are reduced, paired, and ordered by output channel. One element
broadcasts. Q/DQ uses its source scale; contractions use
`alpha * lhs_scale * rhs_scale`; integer-result readouts use that product divided
by output scale. A consumer must not divide a readout ratio by `output_scale`
again. Products and ratios are exact combinations of canonical factors: they
are not rounded to fp32. The optional fp32 `scale` is informational.

Negative `addmm` alpha uses a negative `scale_numerator` with a positive
denominator, as permitted by version 1. The int32 accumulator is unchanged,
including at `-2^31`; no integer negation or intermediate rounding is needed.
An exact bias seed is `beta * bias / (alpha * lhs_scale * rhs_scale)`, including
alpha's sign. Nonintegral or unsafe seeds remain separate exact bias operands.

Grouped convolution GEMMs carry only their own output-channel scale slice;
the readout following their concatenation carries the full channel array.

For readouts with a separate bias operand, the pair describes only the
accumulator coefficient. The bias and its exact binary `beta` factor must also
be evaluated exactly before the final rounding. A consumer that only supports
an integer seed must refuse such a readout unless it proves the bias integral
and the seeded reduction safe. Neither provenance nor `arithmetic="exact"`
grants placement permission or replaces the target's overflow/admission proof.

Exact PT2E integerization and its reference canonicalize frozen source Q/DQ
scales to fp32 once, before rewriting/execution. The receipt's
`scale_canonicalization` records the old and canonical values and whether graph
semantics changed. Unrelated uses of frozen scale tensors retain the original
buffer. Dynamic scales remain SSA operands and cannot establish static device
admission. General host Q/DQ and explicit readout utilities can preserve
binary64 scales; their noncanonical provenance carries `host_only="true"`.

`integerize_pt2e`, `integerize_static_w8a8`, the PT2E integer reference, and static
W8A8 options default to `integer_readout`. Explicit `float_readout` is legacy
approximate host execution: its receipt has `host_only=true`, and contraction
IR carries `prov.quant.host_only="true"`. The `int8_accelerator` profile refuses
this mode and disabling strict integerization. Device execution must keep full
int32 readout by default and perform any required requantization exactly, once,
with RNE followed by zero-point addition and clamping. The frontend does not
authorize a hardware fp32 readout.

PT2E operands may use signed or unsigned byte storage, valid byte subranges,
and asymmetric per-tensor or frozen per-output-channel zero points. The
integerizer widens both operands to int32 before subtracting their zero points
when either operand requires centering. Convolution pads the centered tensor
with integer zero. These sites use exact int32 matrix products; symmetric
signed-int8 sites retain their existing int8 matrix products and exact bias
seeds. A storage-wide bound `K * max_abs(qx-zx) * max_abs(qw-zw)` must fit int32,
independently of calibration values. Backward contractions check each gradient
bound, including overlapping input-gradient contributions and the bias sum.
Dequantization's declared subrange is pattern metadata rather than a runtime
clamp: the reference also accepts storage words outside that subrange.

The receipt retains source storage dtype, zero points and channel axis. The
zero-point corrections are explicit integer SSA operands; contraction/readout
scale arrays follow output-channel order (the schema's implicit last axis).
No new provenance keys or schema version are needed. Consumers must legalize
the emitted int32 contractions exactly or refuse them; an int8-only target
cannot narrow centered values, which may have magnitude 255, to int8.

## Merlin and gemmini-mlir counterparts

Vendor the JSON from the pinned model2mlir commit and test byte-for-byte drift.
Accept its full key set and reject unknown keys; do not require informational
`scale` when rational attributes exist. Test admission on real model2mlir
captures (including seeded, per-channel, backward, and asymmetric Q/DQ), rather
than handwritten or legacy-only readouts. Honor `storage_dtype` and
`zero_point_dtype` independently of MLIR signless integer element types.

Merlin's capture worker and `merlin.capture.integerization.integerize_static_pt2e`
default to `integer_readout`. Explicit recipe `float_readout` may only select a host path: device admission must reject
`host_only=true` receipts or `prov.quant.host_only="true"` IR, and must never
reinterpret approximate `acc_scale` as an exact coefficient. Keep the exact
reference on the same canonicalized graph, and retain the scale-change ledger
in capture receipts.

Merlin's host requantization API must accept full int32 accumulator tiles and
the exact numerator/denominator arrays, include any separate exact bias,
then apply RNE, add zero point, and clamp to `quant_min` / `quant_max`.
gemmini-mlir must preserve full-width int32 descriptors and this rational
contract through its ABI. An optional fp32 device readout requires a recorded
proof that its product introduces no rounding for every reachable accumulator.
