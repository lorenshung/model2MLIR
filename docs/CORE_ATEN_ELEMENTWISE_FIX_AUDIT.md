# Core ATen elementwise gap audit

This audit covers the 38 elementwise-like Core ATen overloads that were opaque in the
193-overload test corpus captured with PyTorch 2.13.0+cu130.  It is deliberately an
implementation guide, not a claim of support: an overload only becomes supported after its
lowering is captured with zero opaque calls and its result is compared with eager PyTorch.

The checkout used for the audit is model2MLIR `59e22668b0e9526d75e944b5505bf196cee698d5`
with xDSL 0.69.0.  The existing pointwise construction machinery is sufficient:
`_make_math_unary`, `_unary_elementwise`, `_binary_elementwise`, `_pointwise`, `_splat_scalar`,
`_fx_arg`, and the raw keyword arguments in `meta["_fx_kwargs"]`.

## Available scalar operations in xDSL 0.69.0

The following constructors are present in the pinned environment and accept scalar SSA values:

| MLIR operation | xDSL constructor |
| --- | --- |
| `math.acos` | `xdsl.dialects.math.AcosOp(x)` |
| `math.acosh` | `AcoshOp(x)` |
| `math.asin` | `AsinOp(x)` |
| `math.asinh` | `AsinhOp(x)` |
| `math.atan` | `AtanOp(x)` |
| `math.atan2` | `Atan2Op(y, x)` |
| `math.atanh` | `AtanhOp(x)` |
| `math.cosh` | `CoshOp(x)` |
| `math.erf` | `ErfOp(x)` |
| `math.expm1` | `ExpM1Op(x)` |
| `math.log10` | `Log10Op(x)` |
| `math.log1p` | `Log1pOp(x)` |
| `math.log2` | `Log2Op(x)` |
| `math.sinh` | `SinhOp(x)` |
| `math.tan` | `TanOp(x)` |
| `math.trunc` | `TruncOp(x)` |
| integer bitwise | `arith.AndIOp`, `OrIOp`, `XOrIOp` |
| integer division/remainder | `arith.DivSIOp`, `FloorDivSIOp`, `RemSIOp` |
| comparisons | `arith.CmpfOp(a, b, predicate)`, `CmpiOp(a, b, predicate)` |
| choice | `arith.SelectOp(condition, true_value, false_value)` |
| floating min/max | `arith.MinimumfOp`, `MaximumfOp` |
| signed integer min/max | `arith.MinSIOp`, `MaxSIOp` |

There is no floating remainder operation in xDSL's `arith` dialect and no fmod/remainder
operation in its `math` dialect.  Formula expansions are usable for the bounded corpus but are
not bit-for-bit substitutes for a primitive `fmodf` over the entire floating-point domain.

## Low-risk direct lowerings (23 overloads)

Fifteen unary math overloads can be registered directly through the existing pointwise builders:

```python
"aten.acos.default":  _make_math_unary("AcosOp", "acos"),
"aten.acosh.default": _make_math_unary("AcoshOp", "acosh"),
"aten.asin.default":  _make_math_unary("AsinOp", "asin"),
"aten.asinh.default": _make_math_unary("AsinhOp", "asinh"),
"aten.atan.default":  _make_math_unary("AtanOp", "atan"),
"aten.atanh.default": _make_math_unary("AtanhOp", "atanh"),
"aten.cosh.default":  _make_math_unary("CoshOp", "cosh"),
"aten.erf.default":   _make_math_unary("ErfOp", "erf"),
"aten.expm1.default": _make_math_unary("ExpM1Op", "expm1"),
"aten.log10.default": _make_math_unary("Log10Op", "log10"),
"aten.log1p.default": _make_math_unary("Log1pOp", "log1p"),
"aten.log2.default":  _make_math_unary("Log2Op", "log2"),
"aten.sinh.default":  _make_math_unary("SinhOp", "sinh"),
"aten.tan.default":   _make_math_unary("TanOp", "tan"),
"aten.trunc.default": _make_math_unary("TruncOp", "trunc"),
```

`aten.atan2.default` is the binary analogue: use `_binary_elementwise` with a scalar builder
that constructs `math.Atan2Op(args[0], args[1])`.  PyTorch's order is `(self, other)`, matching
the MLIR operation's `(y, x)` order.

Five bitwise overloads use the same broadcast/scalar-splat path as add/mul, selecting
`arith.AndIOp`, `OrIOp`, or `XOrIOp`:

- `aten.bitwise_and.Scalar`
- `aten.bitwise_or.Scalar`
- `aten.bitwise_or.Tensor`
- `aten.bitwise_xor.Scalar`
- `aten.bitwise_xor.Tensor`

The scalar helper must accept a Python `bool` in addition to `int`; `_binary_elementwise`
currently excludes booleans when recognizing a scalar argument.  The result element type must be
integer (including i1), and unsupported float inputs must fail closed to opaque.

`aten.hardtanh.default` is scalar clamp.  Read `min_val` and `max_val` from positional arguments
1 and 2, construct typed constants, then select `MaximumfOp`/`MinimumfOp` for floats and
`MaxSIOp`/`MinSIOp` for signed integers.  `aten.leaky_relu.default` is
`select(x >= 0, x, negative_slope * x)`.  Both fit in one `_pointwise` generic.

## Moderate lowerings requiring dtype/edge-case care (9 overloads)

### Logical operations

`aten.logical_and.default`, `logical_or.default`, and `logical_xor.default` always produce i1.
For each operand, first convert it to truth with `CmpiOp(x, 0, "ne")` for integers or
`CmpfOp(x, 0, "une")` for floats.  The unordered-not-equal predicate is important: NaN is true
under PyTorch truth conversion.  Combine the i1 values with `AndIOp`, `OrIOp`, or `XOrIOp`.
The current bool-only corpus would pass without conversion, but that is not general support.

### Classification and sign

- `aten.isnan.default`: for a floating input, `CmpfOp(x, x, "uno")`; for integer/bool input,
  return an i1 false constant.
- `aten.isinf.default`: for a floating input, compare `math.absf(x)` equal to a typed positive
  infinity constant; for integer/bool input return false.  xDSL prints `FloatAttr(inf, f32)` as
  the portable hexadecimal `0x7f800000`, avoiding a textual `+inf` spelling.
- `aten.sign.default`: use ordered `x > 0` and `x < 0` comparisons and nested selects among
  typed `1`, `-1`, and `0`.  Ordered comparisons intentionally map NaN to zero, matching current
  PyTorch behavior.  Preserve signed zero through the zero branch if bitwise signed-zero behavior
  is tested; returning a fresh +0 is only numerically, not bitwise, equivalent.

### Activations

`aten.elu.default(input, alpha, scale, input_scale)` is
`select(x > 0, scale*x, scale*alpha*expm1(input_scale*x))`.  All three scalar parameters must be
read from `_fx_args`, not silently assumed to be 1.  Use `math.ExpM1Op` in the negative branch.

### Division with rounding mode

`aten.div.Scalar_mode` and `aten.div.Tensor_mode` must inspect
`meta["_fx_kwargs"]["rounding_mode"]` (or positional argument 2):

- signed integer `"floor"`: `arith.FloorDivSIOp`;
- signed integer `"trunc"`: `arith.DivSIOp`;
- float `"floor"`: `math.FloorOp(arith.DivfOp(...))`;
- float `"trunc"`: `math.TruncOp(arith.DivfOp(...))`.

The corpus uses positive i64 inputs and `rounding_mode="floor"`, but tests must include mixed
signs because floor and truncation agree on the current positive-only sample.

## Lowerings with explicit limitations (6 overloads)

### `aten.clamp.Tensor`

The existing `decompose_clamp` only handles scalar bounds: it calls `float(lo)`/`float(hi)` on
raw FX arguments.  Tensor bounds arrive as SSA operands and require a three-input pointwise
generic with pairwise max then min, plus normal broadcast maps.  Optional `None` bounds change
the operand positions, so the implementation must derive which operands correspond to min/max
from raw FX args rather than assume exactly three operands.

### Floating remainder and fmod

The bounded suite's float32 cases can be expressed as:

- `remainder(x, y) = x - floor(x / y) * y`
- `fmod(x, y) = x - trunc(x / y) * y`

Integer remainder should instead use floor division plus multiply/subtract for PyTorch remainder,
and `arith.RemSIOp` for fmod.  These formulae need scalar-splat and tensor broadcast variants for:

- `aten.remainder.Scalar`
- `aten.remainder.Tensor`
- `aten.fmod.Scalar`
- `aten.fmod.Tensor`

For publication-grade all-value semantics, the float formula is insufficient: separate rounded
division and multiplication can disagree with a primitive fmod/remainder for extreme ratios,
subnormals, infinities, zero divisors, and signed zero.  A fully sound implementation needs either
a standard-dialect primitive with a guaranteed lowering or an explicitly modeled scalar libm
call.  Until that path exists, document these as bounded-domain implementations and test the
special values separately.

### `aten.atan2.out`

The numerical result can reuse `aten.atan2.default`, but that is not complete `out=` semantics.
The FX importer only resolves positional node arguments into operands; the `out` tensor remains
in `_fx_kwargs` and is neither passed to the generated operation nor written.  Returning a new
tensor may match result values while failing mutation and alias guarantees.  This overload needs
an explicit destination-passing/write representation (and an ABI test observing the original
`out` input), not merely a second table entry pointing at the functional emitter.

## Focused validation required

For each direct math operation, validate zero opaque calls and eager agreement on ordinary values,
domain boundaries, NaN, and infinities.  Add these interaction tests in addition to the existing
one-case-per-overload corpus:

1. atan2 quadrants, signed zeros, infinities, and the `out=` mutation/alias witness;
2. bitwise scalar/tensor with i1, i32, i64, negative integers, and broadcasting;
3. logical operations on bool, integer, float zero/nonzero, NaN, and broadcasting;
4. sign on negative/positive values, both signed zeros, infinities, and NaN;
5. ELU with non-default alpha/scale/input_scale, and leaky-ReLU with a non-default slope;
6. floor/trunc division and both remainder families over every sign quadrant;
7. clamp.Tensor with lower-only, upper-only, both bounds, and broadcast bounds;
8. fmod/remainder special values, with any known divergence reported rather than tolerance-hidden.

Merlin's pipeline contains `convert-math-to-libm` followed by `convert-math-to-llvm`, so emitting a
valid standard-dialect math operation is only the model2MLIR half of support.  The new capsules
must still be compiled and executed unchanged through Merlin.  Any missing libm symbol, illegal
math conversion, unsupported vector form, or numerical mismatch is a Merlin-side result and must
be reported separately rather than worked around in model2MLIR.
