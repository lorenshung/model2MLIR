# ATen CPU RNG lowering

The FX importer initializes a mutable `ml_program.global @global_seed` with
ATen's CPU MT19937 engine state: 624 uint32 words represented in i64 and a
cursor. `rng_seed` defaults to zero. Initialization uses the seed's low 32 bits
and the standard 1812433253 recurrence, exactly as `ATen/core/MT19937RNGEngine.h`.
These constants are generator state, never sampled outputs. The in-place
624-word twist, uint32 tempering, cursor advancement and distribution transforms
execute in the lowered program. There is no Python/Torch RNG call at runtime.
The separate `@global_normal_cache` stores a double sample and its validity.

This contract follows the installed ATen CPU headers (PyTorch 2.13.0+cu130):
`MT19937RNGEngine.h`, `DistributionsHelper.h`, `TransformationHelper.h`, and
`native/cpu/DistributionTemplates.h`. It targets the CPU eager stream, regardless
of which downstream machine executes the generated program. It does not model
CUDA Philox or the optional torch-mlir backend's RNG.

- `rand`: row-major serial consumption. Float32 uses one uint32 draw's low
  24 bits scaled by 2^-24. Float64 uses two draws, first in the high 32 bits,
  selects the low 53 bits and scales by 2^-53. Half/bfloat16 use float32 opmath
  and then cast. Empty tensors consume no words.
- `randn`: the scalar ATen path (contiguous tensors with fewer than 16 elements)
  uses double uniforms: `theta = 2*pi*u1`, `r = sqrt(-2*log1p(-u2))`.
  It returns `r*cos(theta)` and caches `r*sin(theta)`. A cached sample consumes
  no words; a new pair consumes four words. Conversion to the output dtype
  happens after the double transform. The cache persists across calls and other
  distributions. Contiguous tensors with at least 16 real elements use the
  16-wide Box-Muller fill. Float/double first draw `numel` uniforms, transform
  complete blocks, then redraw and overwrite the last 16 elements when the
  length is not a multiple of 16. Half/bfloat16 draw one float buffer per block
  and overlapping tail instead. This path leaves the scalar normal cache
  unchanged. Standard math operations reproduce the vector distribution within
  the suite's `torch_close` tolerance; capability-specific SIMD approximations
  are not a bitwise contract.
- Complex64 `rand` consumes interleaved real/imaginary float draws. Complex64
  `randn` runs the normal path on twice the logical element count with each
  component's standard deviation `1/sqrt(2)`, including the vector threshold
  and shared scalar cache.
- `randperm`: ATen's small-n forward Fisher-Yates algorithm. Initialize an
  iota; for each `i` from 0 through `n-2`, draw one uint32 and swap `i` with
  `i + draw % (n-i)`. No random-key sort or rejection sampler is involved.
  The large-n ATen path (n >= 2^32/20) remains unsupported.
- Training `native_dropout`: one double uniform per element, keep when
  `u < 1-p`, then multiply the input by its mask and inverse keep probability.
  The p=0 and p=1 endpoints still consume draws. Inference consumes none.
  Preserve both tensor results, including the boolean mask and signed zeros.
  Complex64 uses one mask per logical element for both components and consumes
  exactly the same Bernoulli stream as its real counterpart.
- `rand_like`/`randn_like` follow `rand`/`randn`. RNG operations inside
  imported loop bodies share the enclosing module's globals; the declaration
  is discovered from the emitted IR, including nested regions.

Repeated invocations advance the same engine and normal cache. A fresh module
reproduces the seeded stream. Multiple stochastic operations share both globals
in program order. Targets must lower mutable globals and serial `scf` loops;
parsing the module alone is not execution evidence.

Regression tests interpret the emitted IR against freshly seeded eager Torch,
including mixed distributions, odd normal sizes/cache reuse, high-bit seeds,
empty draws, dropout endpoints and draws crossing the 624-word twist boundary.
End-to-end Core ATen verification uses the existing `torch_close` policy
(rtol=1e-5, atol=1e-6, equal_nan), including complete call-boundary semantics.
