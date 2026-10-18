# Portable arithmetic and bounded CPU observations

The FX importer emits standard MLIR arithmetic. An accelerator's explicit
integer policy has its own exact reference; a host fallback uses the original
eager floating oracle, including source CPU rounding and layout semantics.

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

## Source CPU arithmetic policies

Host fallbacks retain the eager CPU oracle. The importer represents native
rounding and traversal explicitly with `arith`, `math`, `scf`, `tensor` and
`linalg` operations. The target may be a different CPU: these operations carry
the source algorithm, including its intermediate rounding. Sample tensor
values never select an algorithm or become precomputed output constants.

The CPU moment implementation follows
[`moments_utils.h`](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cpu/moments_utils.h):
sixteen-vector Welford chunks, a binary cascade, scalar tails, and ordered
lane merges. FMA boundaries are explicit. Half/BF16 vectors retain two float
halves. Empty lanes still merge; their zero products can propagate a NaN.
LayerNorm preserves native affine FMAs. GroupNorm distinguishes NCHW row
moments from channels-last column sums, variance clamping and double epsilon.
Large channels-last reductions currently require one source CPU thread.

BatchNorm preserves its
[collect-stats and affine kernels](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cpu/batch_norm_kernel.cpp)
and the scalar type boundaries in
[`Normalization.cpp`](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/Normalization.cpp).
Dense float statistics use double accumulation; channels-last statistics use
float buffers and fused square accumulation. Reduced-width dense statistics
have SIMD partials and scalar tails. BF16 statistic buffers and momentum
round at their native storage boundaries; Half retains float buffers.
Functionalized running-state results remain explicit SSA outputs.

The slow-convolution backend is selected using ATen's metadata-only backend
selector. On the tested AMD/AVX2 MKL source CPU, small float SGEMM uses separate
SSE multiply/add operations. Its NN path initializes the accumulator from
bias, scales the left row-major operand by alpha before multiplication, and
splits each eight-term tile only in the alignment prefix. Its TN path uses
four-lane sums for blocks of four outputs and two four-lane accumulators for
trailing outputs, with alignment peeling and ordered horizontal sums. Input
and output layout, including the offset of a contiguous gradient, determine
those boundaries. The reduced-width NN fallback follows
[`BlasKernel.cpp`](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cpu/BlasKernel.cpp)'s
four independent float sums, scalar tail and final bias addition.
These policies are checked against independent random native outputs and
cancellation/overflow witnesses. They do not replace integerized contractions.
Other BLAS vendors, ISA paths, MKL instruction overrides, and larger SGEMM
kernels retain their existing lowering; this policy is not a claim of native
bit agreement for those different implementations.

CPU mean uses the native four-stream/cascade sum traversal after input dtype
promotion. CPU half/BF16 rsqrt distinguishes complete SIMD blocks from scalar
tails: scalar sqrt rounds to the storage dtype before reciprocal division.
The contiguous BF16 embedding-bag sum fast path preserves FBGEMM's output
conversion, including its halfway rounding. Boolean col2im accumulates overlaps
with OR. CPU MKL R2C FFT result metadata preserves native batch-axis packing
and transformed-axis strides, and replays dependent view metadata.

The regression tests compare exact stored bits (and NaN masks), including
random inputs, cancellation, nonfinite values, SIMD boundaries and non-dense
layouts. The exhaustive rsqrt tests cover all 65,536 Half and BF16 encodings.
These host policies do not change the exact rational quantization oracle,
requantization rules, or exported quantization schema.
