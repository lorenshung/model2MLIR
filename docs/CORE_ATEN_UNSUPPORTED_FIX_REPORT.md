# Core ATen unsupported-op fix report

## Scope and revision

- model2MLIR base: `59e22668b0e9526d75e944b5505bf196cee698d5`
- structural-lowering commit: `ca82e02c43324b129cfb62a6d544d1289fdeb251`
- final semantic/ABI commit: `fac15d7030062a21cb9a51ef84be2745f7cdbd17`
- branch: `core-aten-unsupported-fixes`
- worktree: `/scratch/jack/model2MLIR-core-aten-fixes`
- input corpus: Merlin's complete pinned 193-overload PyTorch `2.13.0+cu130`
  Core ATen corpus, denominator digest
  `2195c829a9f600e91247ee168374180c5b34367a10d5a6243723d0f75f82af65`
- capture interpreter: `/scratch/jack/model2MLIR/.venv/bin/python`
  (`torch==2.13.0+cu130`, `xdsl==0.69.0`)

This branch does not weaken the zero-opaque gate and does not synthesize fake
provenance.  It fixes capture preparation and adds standard-dialect xDSL
lowerings or a small, explicit set of pinned PyTorch decompositions.

## Current result

`51/95` formerly capture-failed cases now convert to valid, zero-opaque MLIR.
`44/95` remain unsupported.  Together with the original 98 captures, this
raises the structurally capturable corpus from `98/193` to `149/193`.

The full 193-case recapture produced:

- 149 clean zero-opaque captures;
- 44 capture failures;
- 124 captures retaining the requested exact-overload provenance;
- 25 clean captures whose decomposition no longer retains the requested exact
  overload in provenance.

"Zero opaque" is a model2MLIR structural result: emitted IR verifies and its
result shape/dtype agree with eager PyTorch.  It is not a numerical Merlin or
hardware pass.  The 51 newly capturable cases have not yet been executed through
the final compiler and therefore are explicitly `not_run` in the ledger.

## Semantic and result-ABI corrections

The original 98 clean captures included seven cases which were structurally
clean but not semantically complete.  This branch also fixes those cases:

- `aten.add.Scalar`, `aten.add.Tensor`, `aten.sub.Scalar`, and
  `aten.sub.Tensor` now multiply the second argument by non-default `alpha`.
- `aten.addmm.default` now implements `beta * self + alpha * (mat1 @ mat2)` and
  zero-initializes the accumulating `linalg.matmul` output.
- `aten.native_layer_norm.default` returns output, mean, and reciprocal standard
  deviation with the exact eager three-result ABI.
- `aten._native_batch_norm_legit_no_training.default` returns the normalized
  output plus the two required empty saved-stat tensors.

Fresh captures at `fac15d7` contain all required constants and result types.
Independent final-executable Spike runs pass all output leaves for six of the
seven cases: the four add/sub overloads, `addmm`, and all three layer-norm
results.  Batch norm reaches a separate Merlin lowering failure described
below.  This is scalar-ISA simulation evidence, not FPGA hardware evidence.

## Fixes implemented

- Range constraints now serialize SymPy infinity, PyTorch `int_oo`, and Python
  infinity as the existing unbounded `None` sentinel.  This prevents the API
  from silently discarding a valid decomposed export.
- Direct unary math: `acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`, `cosh`,
  `erf`, `expm1`, `log10`, `log1p`, `log2`, `sinh`, `tan`, and `trunc`.
- Direct binary/pointwise families: `atan2`, integer bitwise operations,
  numeric logical truth conversion, `isnan`, `sign`, `elu`, and rounded
  `div` (`floor`/`trunc`).
- `clamp.Tensor` now consumes SSA tensor bounds and selects integer or floating
  min/max operations by dtype.
- Static `constant_pad_nd` accepts nonzero scalar fills.
- Direct reductions/layout/indexing: `prod`, `flip`, `gather`, and the canonical
  Boolean-mask `masked_scatter` path.
- Five explicitly audited upstream PyTorch decompositions are enabled:
  `_log_softmax`, `hardtanh`, `index_select`, `isinf`, and `leaky_relu`.
- The range and integer-clamp fixes also unlock the default decompositions for
  reflection/replication padding, selected scatter, fill, and the layer/group
  normalization backward cases in this corpus.

## Remaining 44 cases

```text
aten._adaptive_avg_pool2d.default
aten._adaptive_avg_pool2d_backward.default
aten._adaptive_avg_pool3d.default
aten._cdist_forward.default
aten._embedding_bag.default
aten._local_scalar_dense.default
aten._native_batch_norm_legit.default
aten._native_batch_norm_legit.no_stats
aten._pdist_forward.default
aten.adaptive_avg_pool1d.default
aten.as_strided.default
aten.avg_pool1d.default
aten.avg_pool2d.default
aten.avg_pool2d_backward.default
aten.avg_pool3d.default
aten.col2im.default
aten.convolution_backward.default
aten.diagonal.default
aten.embedding_dense_backward.default
aten.empty_strided.default
aten.fmod.Scalar
aten.fmod.Tensor
aten.grid_sampler_2d.default
aten.index_put.default
aten.max_pool2d_with_indices.default
aten.max_pool2d_with_indices_backward.default
aten.max_pool3d_with_indices.default
aten.native_dropout.default
aten.native_group_norm.default
aten.nonzero.default
aten.rand.default
aten.randn.default
aten.randperm.default
aten.remainder.Scalar
aten.remainder.Tensor
aten.resize_.default
aten.scatter.src
aten.scatter.value
aten.scatter_add.default
aten.scatter_reduce.two
aten.sort.default
aten.topk.default
aten.var.correction
aten.var.dim
```

The remaining categories are 14 pooling/resampling/convolution-backward cases,
8 indexing/scatter/order/dynamic-result cases, 7
normalization/distance/embedding-backward cases, 5
storage/layout/mutation/scalar-extraction cases, 4 random/stateful cases, 4
floating remainder cases, and 2 variance reductions.

The highest-leverage remaining mechanisms are a shared scatter/accumulate
lowering, pooling windows, curated multi-result normalization decompositions,
and a correct dynamic-result ABI.  The hard semantic contracts are:

- `nonzero` has a data-dependent result extent.  Valid MLIR uses `?`, not the
  current invalid `-1`; Merlin's static arena planner will then need a dynamic
  allocation/shape contract.
- `sort` and `topk` must return both values and indices and honor tie/NaN rules.
- `as_strided`, `empty_strided`, and `resize_` expose storage, stride, alias, or
  mutation semantics not represented by ordinary tensor SSA values.
- random creation and training dropout require an explicit generator/state ABI
  for reproducible cross-backend testing.
- convolution and pooling backward need real accumulation kernels and complete
  multi-result handling.
- floating `fmod`/`remainder` have no direct standard MLIR arithmetic op; the
  tempting divide-round-multiply formula is not IEEE-equivalent for all finite
  magnitudes, so this branch deliberately leaves them unsupported.

## Verification

Commands were run with the worktree on `PYTHONPATH`, because the shared virtual
environment's editable install points at the clean checkout.

```text
python -m pytest -q tests/test_ops.py
101 passed

python -m pytest -q tests/test_core_aten_gap_ops.py
51 passed

python -m pytest -q tests/test_torch_export_range_constraints.py
5 passed

git diff --check
clean
```

The combined focused and existing suite (`test_core_aten_gap_ops.py`,
`test_torch_export_range_constraints.py`, and `test_ops.py`) passes in full.
The complete 193-case structural rescan produced `149 zero_opaque / 44
remaining` using the pinned interpreter above.

Machine-readable before/after evidence is in
`artifacts/core-aten-before-after-ledger.json`.  It records every overload's
capture status, ABI, exact failure reason, numerical status, and evidence path.
Its aggregate numerical states are intentionally conservative: 89 pass, 9
execution failures, 51 newly capturable but not run, and 44 capture unavailable.
Of the 89 passes, 83 are carried forward only where normalized captured MLIR is
semantically identical to the baseline, and 6 are fresh `fac15d7` executions.

## Merlin-side issues discovered during this audit

No Merlin files were changed.

- The new standard math operations lower through `convert-math-to-libm` to
  RISC-V newlib symbols, and Merlin already links `-lm`.  However, Merlin's
  `arena_bind` pure-callee allowlist omits `acosh{,f}`, `asinh{,f}`, and
  `atanh{,f}`.  This is an arena-classification/optimization gap; it has not
  been demonstrated as a link failure.
- A valid dynamic `nonzero` result would be rejected by Merlin's static arena
  planner (`ArenaPlanError`) until a dynamic-buffer contract is added.
- `aten._native_batch_norm_legit_no_training.default` now captures the correct
  `(tensor<2x4x2x2xf32>, tensor<0xf32>, tensor<0xf32>)` ABI, but Merlin's
  upstream lowering subprocess terminates with `SIGSEGV` (`si_addr=NULL`) and
  no diagnostic while lowering the zero-length result buffers.  The exact
  process trace is
  `artifacts/semantic-seven-fac15d7/bn-strace.txt`.  No Merlin source was
  changed.
