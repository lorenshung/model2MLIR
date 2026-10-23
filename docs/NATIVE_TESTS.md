# Native regression tests

CI should run the full suite with an upstream LLVM/MLIR installation:

```bash
export MODEL2MLIR_MLIR_TRANSLATE=/path/to/llvm-install/bin/mlir-translate
export MODEL2MLIR_CLANG=/path/to/llvm-install/bin/clang
export PATH=/path/to/llvm-install/bin:$PATH
export PYTHONPATH="$PWD"
python -m pytest tests
```

`mlir-opt` must be beside `mlir-translate`, and the install must include
`lib/libmlir_c_runner_utils`. With either tool variable configured, session
preflight fails on missing, non-executable, or broken tools. Compilation,
translation, linking, loading, and numerical mismatches fail tests. An optional
local run without configured variables may skip native tests if tools are absent.

`tests/native_execution.py` snapshots mutable `ml_program` tensor globals into
`memref` globals for upstream CPU execution; [RNG_SEMANTICS.md](RNG_SEMANTICS.md)
lists what the RNG regressions cover.

Half native LayerNorm uses float Welford moments, including lane and zero-count
merges at overflow boundaries, and preserves the mean/rstd result dtypes.
Mixed float affine parameters retain float statistics. Its float affine
evaluation follows the [ATen CPU kernel](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/layer_norm_kernel.cpp).

Empty/empty_strided outputs have undefined contents. Tests compile and execute
their allocation, then compare only shape and dtype, following the bounded
suite's metadata policy. Lowerings emit `tensor.empty`, without zero fills or
sample-derived values. Public stride and alias metadata remain in
`m2m.result_layout`; tensor value types describe logical elements.

Static `resize_` lowerings read the runtime backing allocation, including holes
and the source offset. Callers must supply the captured storage capacity and
layout, not a contiguous copy of the logical elements. A resize preserves the
old storage prefix and leaves newly allocated elements uninitialized. Mutation
results carry the updated input descriptor through the existing result ABI.
The native helper copies complete input storage so numeric resize tests can
compare initialized holes and offsets without changing their eager semantics.
Intermediate tensor values do not declare a complete backing allocation;
storage reads through those values remain visibly refused.

Torch 2.13 lacks complex item fake metadata and rejects some complex dropout
fake kernels. The stateless ATen FX fallback propagates only metadata and keeps
the original operators on graph edges. Complex item can feed scalar_tensor;
checked conversion to real asserts that its imaginary component is zero at
runtime. Unsupported scalar control flow, attributes, and data mutations remain
refused by that fallback.
