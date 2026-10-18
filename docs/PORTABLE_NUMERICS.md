# Portable arithmetic and bounded CPU observations

The default FX importer emits standard MLIR arithmetic. A finite tolerance does
not promise agreement at overflow or severe cancellation boundaries with an
arbitrary host BLAS/oneDNN schedule. An accelerator's explicit integer policy
has its own reference; a host fallback still uses the original floating oracle.

## Exact quantized arithmetic

For `integerize_pt2e(..., numerics="integer_readout")`, the oracle is exact
rational arithmetic from integer codes and original stored binary scales and
zero points. Evaluate the full scale product and bias exactly, then round half
to even, add the output zero point, and clamp to the declared range. An odd
zero point does not change tie parity. Float output receives one final IEEE
rounding, including subnormals, overflow and signed underflow.

`m2m_q.exact_readout` preserves activation/weight scales, alpha/beta and bias
until that final rounding. A bias preload requires an exactly integral scaled
bias and a conservative proof that the seed plus the full reduction fits i32.
Bias is never pre-rounded or saturated. Backward input-gradient bounds include
all kernel taps. Every output-mask rewrite recompiles the FX module.

Standard Q/DQ expansion compares decoded IEEE mantissas and exponents using
integers. Scalar f64 and stored per-channel f32/f64 scales retain their exact
values. Byte storage is signless i8; unsigned input uses zero extension, signed
input uses sign extension, and quantized codes are narrowed only after an
integer range decision. No float-to-byte conversion can produce poison.
Infinities quantize to the endpoints. NaN and invalid scales/zero points are
checked instead of converted.

Canonical range attributes are `prov.quant.quant_min` and
`prov.quant.quant_max`. Exact readout carries original factor attributes and
exact decimal-string rational `prov.quant.scale_numerator` /
`prov.quant.scale_denominator` arrays. A rounded combined scale cannot replace
these parameters. Exact expansion is marked `prov.quant.arithmetic = "exact"`.

Contraction rewriting currently admits symmetric signed-int8 operands with
per-tensor activations and frozen per-output-channel or per-tensor weights.
Nonzero operand zero points, uint8 contraction storage, unsupported channel
axes and non-f32 bias/dequantize precision remain explicit refusals. Integer
alpha/beta values that cannot be represented exactly by the binary64 readout
schema are also refused. Standalone
byte Q/DQ and exact readout output support signed/unsigned, asymmetric,
narrow-range and per-channel parameters. Readout widths above 4096 integer bits
are refused to bound compiler artifacts. Device placement still needs a
target-owned proof of these semantics; frontend integerization is not that
proof.

`numerics="float_readout"` retains its legacy reassociated floating evaluation
and makes no exact-arithmetic claim. Eager float Q/DQ divergence is informational.
`tests/test_exact_quant_arithmetic.py` compares exact codes/bits without
tolerances and records eager-float divergence separately.

## Saved layer-normalization statistics

`native_layer_norm_backward` consumes the supplied mean and rstd tensors. They
must never be recomputed from the activation. In the installed PyTorch source,
[the CPU backward kernel](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cpu/layer_norm_kernel.cpp#L186)
reads each statistic with a linear row index from its data pointer. Tensor
strides do not redirect these reads. By contrast, input and affine tensors are
made contiguous by the public CPU entry point.

The importer can retain a noncontiguous statistic's raw reads only when its
actual backing allocation is supplied. `contiguous_storage_tensor` checks the
physical span. A portable document containing only logical values, shape,
stride, and offset cannot specify holes or bytes beyond its logical span.
Merlin's source-only `InputStorageUnavailable` refusal correctly excludes these
inputs. Allocator leftovers and a fresh contiguous copy are not substitutes
for missing source storage. Even a sample whose existing NaNs hide the effect
has no authority to read those bytes.

`test_layer_norm_backward_hidden_statistic_cells_are_observable` supplies
initialized backing storage with identical logical tensors and changes only
hidden cells. Both padded and zero-stride statistics change eager input and
weight gradients. This is an input contract boundary, not a reason to rewrite
or recompute statistics.

## Contraction cancellation and overflow

Let `M` be float32's largest finite value. A weight-gradient dot product in the
bounded suite has products `[M, 0, 0, -1, 0, -M, 1, 0, 1]`. Exact real arithmetic
gives 1; the portable ascending float32 recurrence gives 2. On the tested
PyTorch/MKL build, fresh eager gives 0, while `MKL_CBWR=COMPATIBLE` gives 1.
This backend switch changes no tensor, dtype, stride, or operator argument.
Reproducing that schedule is outside portable operator lowering.

The addmm overflow witness contains products `[-M, -M, M, 0]`, alpha 0.5,
and self 1. Accumulate then scale overflows to negative infinity; distributing
alpha before accumulation yields a finite negative half-maximum. Neither the
algebraic formula nor a wider accumulator defines the host BLAS schedule.
The current default and compatible MKL paths exhibit those different results.

Forward convolutions have analogous distinctions: float32 im2col uses SGEMM,
while bfloat16 can select oneDNN. Fused products can cancel an overflowing
intermediate; a separately rounded product can produce infinity followed by
NaN. Bias initialization and the reduction tree also affect overflow.
[PyTorch's convolution implementation](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/ConvolutionMM2d.cpp#L223)
selects GEMM calls, not a portable scalar evaluation order.

The bounded suite's six remaining numerical mismatches are documented as
outside the portable numerical contract. Keep their failures visible; do not
change oracle values, tolerances, reduction trees to imitate a CPU library, or
quantization policy to manufacture agreement. The case-by-case adjudication
and simulator receipts live in Merlin's
`experiments/reference-data/phase0/core-aten-last19-adjudication.yaml` and the
requested `out/artifacts/verification/core-aten/last19/` output directory.
