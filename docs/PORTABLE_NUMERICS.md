# Portable arithmetic and bounded CPU observations

The default FX importer emits standard MLIR arithmetic. A finite tolerance does
not promise agreement at overflow or severe cancellation boundaries with an
arbitrary host BLAS/oneDNN schedule. An accelerator's explicit integer policy
has its own reference; a host fallback still uses the original floating oracle.

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
