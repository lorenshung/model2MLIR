# Stateful RNG lowering contract

`model2MLIR` lowers `aten.rand.default`, `aten.randn.default`,
`aten.randperm.default`, and training `aten.native_dropout.default` without
capturing or embedding sampled outputs. These operations are fundamentally
stateful, so their linalg-on-tensors modules declare one mutable
`ml_program.global @global_seed : tensor<i64>`.

## Contract

- `m2m.convert(..., backend="fx_importer", rng_seed=N)` sets the owned
  importer's initial 64-bit state. The default is zero. (The optional
  torch-mlir backend owns its seed policy separately.)
- Empty draws consume no state. Nonempty uniform draws retain 11/8/24/53
  high bits for f16/bf16/f32/f64 respectively, then multiply by the exact
  power-of-two reciprocal. The maximum value is representable and below one.
- One uniform tensor draw performs
  `state = state * 6364136223846793005 + 1442695040888963407 (mod 2^64)`, stores
  that state, and derives each element from a deterministic integer hash of the
  new state and the element's row-major linear index.
- `rand` consumes one draw. `randn` consumes two draws and applies Box-Muller.
  Training dropout with `0 < p < 1` consumes one draw. Inference dropout and
  the exact `p=0`/`p=1` cases consume no draw. `randperm` consumes one draw and
  sorts an iota by generated 64-bit keys; equal keys retain iota order.
- RNG operations nested inside imported loop bodies share the enclosing module
  global; the declaration is discovered from emitted IR, including nested regions.
- Re-running a newly instantiated module with the same seed is reproducible.
  Repeated calls to one instantiated module advance its global state and do not
  repeat the first sample.
- `native_dropout` retains its two-result ABI: the scaled value tensor and the
  boolean keep mask.

The arithmetic/hash form intentionally matches torch-mlir's current
linalg-on-tensors RNG lowering. The `random` provenance family is separate
because stateful random generation cannot truthfully be classified as an
elementwise, fill, or reduction operation.

## What this does not claim

This contract is not bit-for-bit PyTorch CPU RNG compatibility. PyTorch CPU
uses MT19937, while CUDA uses counter-based Philox; PyTorch's own
`torch/_decomp/decompositions_for_rng.py` explicitly refuses CPU RNG
functionalization for that reason. Backend-independent compiler validation
must therefore compare deterministic runs of this documented contract or test
distribution/invariants, not compare one stochastic result against an
uncoordinated eager CPU call.

The emitted `ml_program` global/load/store operations are upstream MLIR ops,
but a target must bufferize and lower them. Parsing them is not sufficient.
Merlin's current textual host pipeline was checked against all four generated
modules: it bufferized the state and produced host shared objects, with the
seed visible as a mutable LLVM global. Host execution then demonstrated
repeated-call advancement, fresh-module same-seed reproducibility, permutation
validity, and dropout value/mask agreement. This is frontend and host-lowering
evidence, not Gemmini hardware evidence. A target without that lowering needs
one of:

1. a lowering for mutable `ml_program` globals through bufferization to its
   runtime/LLVM memory model; or
2. a wrapper ABI that threads `(seed, offset)` into and out of the compiled
   function and persists it between invocations.

The second option makes state explicit and is preferable for accelerators that
do not support writable globals. Either implementation must preserve ordering
between load/store pairs and must validate repeated-call advancement plus
same-seed reproducibility.

The host check used Merlin itself, not a standalone `mlir-opt` shortcut:

```text
python -m merlin.cli lower <module.mlir> --out <fresh-dir> --textual --target host
```

For seed 17, `rand(2, 3)` produced a different second result from its first,
while loading a fresh copy of the shared object reproduced the first result
exactly. The same advancement/reseed checks passed for `randn`, `randperm`, and
dropout. The observed `randperm(6)` contained exactly `{0,1,2,3,4,5}`; dropout's
mask agreed elementwise with `output == input/(1-p)` versus zero.

## Capsule recommendation

One grouped `stateful_random` capsule is sufficient to expose all four source
overloads to a regeneration pass. It should accept a small `2x3` float input
and return `rand([2,3])`, `randn([2,3])`, `randperm(6)`, plus both results of
`native_dropout(input, 0.25, True)`. Keep all outputs live so capture provenance
retains every overload and the dropout mask ABI.

That capsule needs a stochastic oracle mode: initialize the compiled RNG state,
run twice to test advancement, reinitialize to test reproducibility, and check
range/distribution/permutation/dropout invariants. Its output must not be
compared directly with an independently advanced eager CPU generator.

## Bounded `randperm`

The owned FX lowering accepts statically sized `randperm` results through 256
elements. It emits a compare-exchange network, which keeps the implementation
portable but grows quadratically. Larger or dynamic permutations remain
explicitly unsupported until a loop-form Fisher-Yates lowering and a bounded
integer sampling primitive are added; they are not silently counted as
covered.
