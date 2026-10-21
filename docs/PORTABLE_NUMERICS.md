# Portable arithmetic and bounded CPU observations

The FX importer emits standard MLIR arithmetic. An accelerator's explicit
integer policy has its own exact reference; a host fallback uses the original
eager floating oracle, including source CPU rounding and layout semantics.

## Exact quantized arithmetic

[QUANT_SCHEMA.md](QUANT_SCHEMA.md) is the contract for exact scales, bias, readout
rounding, operand schemes and the legacy `float_readout` mode. For
`numerics="integer_readout"` the oracle is exact rational arithmetic from
integer codes and the original stored binary scales and zero points. An odd
zero point does not change tie parity. Float output receives one final IEEE
rounding, including subnormals, overflow and signed underflow. Bias is never
pre-rounded or saturated, backward input-gradient bounds include all kernel
taps, and every output-mask rewrite recompiles the FX module.

Standard Q/DQ expansion compares decoded IEEE mantissas and exponents using
integers. Scalar f64 and stored per-channel f32/f64 scales retain their exact
values. Byte storage is signless i8; unsigned input uses zero extension, signed
input uses sign extension, and quantized codes are narrowed only after an
integer range decision. No float-to-byte conversion can produce poison.
Infinities quantize to the endpoints. NaN and invalid scales/zero points are
checked instead of converted.

Contraction rewriting refuses unsupported channel axes, non-f32
bias/dequantize precision, and integer alpha/beta values that the binary64
readout schema cannot represent exactly. Readout widths above 4096 integer bits
are refused to bound compiler artifacts. Frontend integerization is not a
device placement proof. `tests/test_exact_quant_arithmetic.py` compares exact
codes/bits without tolerances and records eager-float divergence separately.

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
These inputs require an explicit backing-storage initialization contract or
a clean refusal. A defined zero-fill makes the missing cells specified input;
allocator leftovers and a fresh contiguous copy are not substitutes for
missing source storage. Even a sample whose existing NaNs hide the effect has
no authority to read unspecified bytes.

`test_layer_norm_backward_hidden_statistic_cells_are_observable` supplies
initialized backing storage with identical logical tensors and changes only
hidden cells. Both padded and zero-stride statistics change eager input and
weight gradients. This is an input contract boundary, not a reason to rewrite
or recompute statistics.

The weight gradient follows the CPU affine form: `a = rstd`, `b = -a*mean`,
then `dgamma = fma(dy, fma(a, x, b), dgamma)`. Replacing the inner affine
expression with `(x-mean)*rstd` changes finite rounding and can turn native
NaNs into infinities. The emitted ordered row loop models the single-thread
CPU boundary used by the evaluator. It retains float opmath for half/BF16,
rounds the intermediate gradient buffer to the input dtype after every row
(including mixed float parameters), and adds that buffer to the final opmath
positive-zero seed before converting to the parameter dtype.

## Source CPU arithmetic policies

Host fallbacks retain the eager CPU oracle. The importer represents native
rounding and traversal explicitly with `arith`, `math`, `scf`, `tensor` and
`linalg` operations. The target may be a different CPU: these operations carry
the source algorithm, including its intermediate rounding. Sample tensor
values never select an algorithm or become precomputed output constants.

CPU add/sub follows
[`ufunc/add.h`](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/ufunc/add.h)
and `BinaryOps.cpp`: subtraction negates the original Scalar alpha before its
dtype conversion and enters the add kernel. Integer zero alpha therefore
differs from floating positive zero alpha. Real SIMD lanes use FMA; Half and
BF16 scalar tails first round the coefficient product to storage precision.
Integral Scalars retain their signed/unsigned 64-bit conversion; subtraction
negates that word and interprets the result as signed, including wraparound at
the signed boundary. Finite alpha overflow is refused as in `Scalar::to`, while
the wrapped Scalar operand uses TensorIterator's unchecked dtype conversion.
Real scalar-broadcast tails retain the compiler's initial FMA peel, blocks of
four float or two double additions using a separately rounded product, and
the remaining scalar FMAs.
The TensorIterator inner runs determine these boundaries. BF16 SIMD NaNs use
the CPU conversion's `0xffff` word, while scalar conversions use `0x7fc0`.

Complex add/sub retains both zero cross terms of complex multiplication, even
for real alpha and alpha one. The AVX2 horizontal-subtraction product,
four-element contiguous tails, broadcast peels and blocks, two-element strided
tails, and remaining C10 scalar products are represented separately. NaN
selection follows the registers of each path, including its scalar unrolls.
Quieting and conversion bits
are explicit, including the source CPU's negative invalid NaN. Targets need
not share the source CPU's NaN canonicalization behavior. Regression tests
compare complete storage words, including NaN signs and signed zeros.
The add/sub storage policy currently requires the verified AVX2 CPU source
kernels; other source CPU variants and GPU kernels are cleanly refused.
For runs larger than TensorIterator's 32,768-element parallel grain, the captured
OpenMP thread count determines the source iterator's contiguous ranges. The
SIMD and scalar policies restart at each range's intersections with coalesced
rows, including partial first and last rows. Target execution need not use the
same thread count.
Half RHS broadcasts also retain the distinct NaN operand priority in the last
eight lanes of each 32-element AVX2 block.
Typed-source observers must support the emitted standard operations: bitcasts,
integer bit operations, unsigned integer comparisons, unordered floating
comparisons and correctly rounded `math.fma`. An observer lacking those
semantics must retain its refusal; compiling the original host source remains
valid. In particular, Merlin's `core_aten_w8a8_reference.py` currently refuses
`arith.bitcast` before it can establish the source/eager storage boundary.

The CPU moment implementation follows
[`moments_utils.h`](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cpu/moments_utils.h):
sixteen-vector Welford chunks, a binary cascade, scalar tails, and ordered
lane merges. FMA boundaries are explicit. Half/BF16 vectors retain two float
halves. Empty lanes still merge; their zero products can propagate a NaN.
LayerNorm preserves native affine FMAs. GroupNorm distinguishes NCHW row
moments from channels-last column sums, variance clamping and double epsilon.
Large channels-last reductions currently require one source CPU thread.

Scalar reductions execute their seed, even with no physical reduction axes.
A rank-zero or size-one sum adds positive zero; complex product multiplies
by `(1+0j)`, including both zero cross terms. Min/max CPU reductions emit a
positive canonical quiet NaN. Numeric reduction results explicitly quiet
signaling NaNs, so LLVM cannot turn multiplication by one into a bitwise copy.
CPU mean follows the native sum-then-divide algorithm in
[`ReduceOps.cpp`](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/ReduceOps.cpp),
with the four-stream/cascade sum traversal after input dtype promotion.
Half input promotion retains the dense copy's F16C vector conversions and
software scalar tails; BF16 mean distinguishes scalar and tensor writeback.
Complex mean retains the fused scaled-division numerators and their NaN
operand priorities. Standard deviation keeps sqrt in opmath before its final
storage conversion, rather than inheriting generic intermediate casts.

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

CPU half/BF16 rsqrt distinguishes complete SIMD blocks from scalar
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
