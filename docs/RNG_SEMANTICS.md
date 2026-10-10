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
  distributions. Larger tensors use a distinct, CPU-capability-dependent
  vector path in eager ATen; this lowering explicitly leaves them unsupported,
  rather than silently producing a different stream.
- `randperm`: ATen's small-n forward Fisher-Yates algorithm. Initialize an
  iota; for each `i` from 0 through `n-2`, draw one uint32 and swap `i` with
  `i + draw % (n-i)`. No random-key sort or rejection sampler is involved.
  The large-n ATen path (n >= 2^32/20) remains unsupported.
- Training `native_dropout`: one double uniform per element, keep when
  `u < 1-p`, then multiply the input by its mask and inverse keep probability.
  The p=0 and p=1 endpoints still consume draws. Inference consumes none.
  Preserve both tensor results, including the boolean mask and signed zeros.

Repeated invocations advance the same engine and normal cache. A fresh module
reproduces the seeded stream. Multiple stochastic operations share both globals
in program order. Targets must lower mutable globals and serial `scf` loops;
parsing the module alone is not execution evidence.

Regression tests interpret the emitted IR against freshly seeded eager Torch,
including mixed distributions, odd normal sizes/cache reuse, high-bit seeds,
empty draws, dropout endpoints and draws crossing the 624-word twist boundary.
End-to-end Core ATen verification uses the existing `torch_close` policy
(rtol=1e-5, atol=1e-6, equal_nan), including complete call-boundary semantics.
