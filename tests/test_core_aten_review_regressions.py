"""Execute emitted IR for edge cases missed by structural coverage tests."""

from contextlib import nullcontext

import numpy as np
import pytest
import torch
from xdsl.dialects import arith, tensor
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, ModuleOp, TensorType, f32
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.builtin import xtype_for_el_type
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.linalg import LinalgFunctions
from xdsl.interpreters.math import MathFunctions
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.tensor import TensorFunctions
from xdsl.interpreters.utils.ptr import TypedPtr
from xdsl.ir import Block, Region

from m2m.ir.decompositions import DECOMPOSITION_TABLE
from m2m.ir.decompositions import _opaque_decomp
from test_native_batch_norm_training import _BatchNormInterpreterExtras, _as_shaped_array


@register_impls
class _ReviewInterpreterExtras(InterpreterFunctions):
    @impl(arith.SelectOp)
    def select(self, interpreter, op, args):
        return (args[1] if args[0] else args[2],)

    @impl(arith.DivfOp)
    def divide(self, interpreter, op, args):
        # Python division raises on zero; MLIR floating division follows IEEE.
        with np.errstate(divide="ignore", invalid="ignore"):
            return (float(np.divide(args[0], args[1])),)

    @impl(arith.ConstantOp)
    def constant(self, interpreter, op, args):
        if isinstance(op.value, DenseIntOrFPElementsAttr):
            typ = op.result.type
            return (ShapedArray(TypedPtr.new(list(op.value.get_values()),
                        xtype=xtype_for_el_type(typ.element_type, interpreter.index_bitwidth)),
                        list(typ.get_shape())),)
        return (op.value.value.data,)

    @impl(tensor.CollapseShapeOp)
    def reshape(self, interpreter, op, args):
        return (ShapedArray(args[0].data_ptr, list(op.results[0].type.get_shape())),)

    @impl(tensor.ExpandShapeOp)
    def expand(self, interpreter, op, args):
        return (ShapedArray(args[0].data_ptr, list(op.results[0].type.get_shape())),)

    @impl(tensor.FromElementsOp)
    def from_elements(self, interpreter, op, args):
        typ = op.results[0].type
        return (ShapedArray(TypedPtr.new(args,
                    xtype=xtype_for_el_type(typ.element_type, interpreter.index_bitwidth)),
                    list(typ.get_shape())),)


def _execute(target, inputs, fx_args, expected, kwargs=None, ssa_positions=None):
    """Interpret the actual decomposition, without importing or rewriting its IR."""
    types = [TensorType(f32, list(value.shape)) for value in inputs]
    block = Block(arg_types=types)
    meta = {"val": expected, "_aten_target": target, "_fx_args": fx_args,
            "_fx_kwargs": kwargs or {}}
    if ssa_positions is not None:
        meta["_fx_ssa_args"] = {pos: block.args[i] for pos, i in ssa_positions.items()}
    result = DECOMPOSITION_TABLE[target](list(block.args), meta, "review")
    results = result.results or [result.result]
    assert all(value is not None for value in results)
    block.add_ops([*result.ops, ReturnOp(*results)])
    module = ModuleOp([FuncOp("review", (types, [value.type for value in results]), Region(block))])
    module.verify()
    interpreter = Interpreter(module)
    for impls in (ArithFunctions(), FuncFunctions(), LinalgFunctions(), MathFunctions(),
                  TensorFunctions(), _BatchNormInterpreterExtras()):
        interpreter.register_implementations(impls)
    interpreter.register_implementations(_ReviewInterpreterExtras(), override=True)
    actual = interpreter.call_op("review", tuple(_as_shaped_array(x, interpreter) for x in inputs))
    tensors = [torch.tensor(value.data).reshape(value.shape) for value in actual]
    return tensors


def test_addmm_zero_beta_ignores_nonfinite_bias():
    bias = torch.tensor([[float("nan"), float("inf")]])
    lhs, rhs = torch.ones(1, 2), torch.ones(2, 2)
    expected = torch.addmm(bias, lhs, rhs, beta=0, alpha=0.5)
    actual, = _execute("aten.addmm.default", (bias, lhs, rhs), (None, None, None),
                      expected, {"beta": 0, "alpha": 0.5})
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("correction", [0, 2, 3, -1])
@pytest.mark.parametrize("values", [[1., 3.], [2., 2.], []])
def test_variance_degrees_of_freedom_and_empty(values, correction):
    x = torch.tensor(values)
    with pytest.warns(UserWarning) if len(values) <= correction else nullcontext():
        expected = torch.var(x, correction=correction)
    actual, = _execute("aten.var.correction", (x,), (None, None), expected,
                      {"correction": correction})
    torch.testing.assert_close(actual, expected, equal_nan=True)


def test_sign_nan_and_signed_zero_match_eager():
    x = torch.tensor([float("nan"), -0., 0., float("inf"), float("-inf"), -2., 3.])
    expected = torch.sign(x)
    actual, = _execute("aten.sign.default", (x,), (None,), expected)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))


@pytest.mark.parametrize("descending", [False, True])
def test_sort_places_nans_and_preserves_indices(descending):
    x = torch.tensor([1., float("nan"), -1., float("nan"), float("inf"), float("-inf")])
    expected = torch.sort(x, descending=descending, stable=True)
    actual = _execute("aten.sort.default", (x,), (None, -1, descending), expected)
    for observed, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, oracle, equal_nan=True)


@pytest.mark.parametrize("weight_present,bias_present", [(False, True), (True, False), (False, False), (True, True)])
def test_inference_batch_norm_optional_affine_slots(weight_present, bias_present):
    x = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    weight, bias = torch.tensor([2., 3., 4.]), torch.tensor([-2., 1., 3.])
    mean, var = torch.tensor([0.5, -1., 2.]), torch.tensor([2., 4., 3.])
    target = "aten._native_batch_norm_legit_no_training.default"
    w, b = weight if weight_present else None, bias if bias_present else None
    expected = torch.ops.aten._native_batch_norm_legit_no_training.default(x, w, b, mean, var, 0.1, 1e-5)
    inputs = [x]; positions = {0: 0}
    for position, value in ((1, w), (2, b), (3, mean), (4, var)):
        if value is not None:
            positions[position] = len(inputs); inputs.append(value)
    actual = _execute(target, inputs, (None, object() if w is not None else None,
                      object() if b is not None else None, object(), object(), 0.1, 1e-5),
                      expected, ssa_positions=positions)
    for observed, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, oracle)


def test_native_layer_norm_bias_only_preserves_affine_and_stats():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    bias = torch.tensor([-2., 1., 3.])
    expected = torch.ops.aten.native_layer_norm.default(x, [3], None, bias, 1e-5)
    actual = _execute("aten.native_layer_norm.default", (x, bias),
                      (object(), [3], None, object(), 1e-5), expected,
                      ssa_positions={0: 0, 3: 1})
    for observed, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, oracle)


def test_full_resolves_runtime_scalar_after_literal_size_argument():
    block = Block(arg_types=[f32])
    meta = {"val": torch.ones(2, 3), "_fx_args": ([2, 3], object()),
            "_fx_ssa_args": {1: block.args[0]}}
    result = DECOMPOSITION_TABLE["aten.full.default"](list(block.args), meta, "review")
    assert not any(op.name == "func.call" for op in result.ops)
    assert isinstance(result.result.owner, tensor.SplatOp)
    assert result.result.owner.operands[0] is block.args[0]
    block.add_ops([*result.ops, ReturnOp(result.result)])
    module = ModuleOp([FuncOp("review", ([f32], [result.result.type]), Region(block))])
    module.verify()
    interpreter = Interpreter(module)
    interpreter.register_implementations(FuncFunctions())
    interpreter.register_implementations(_BatchNormInterpreterExtras())
    for scalar in [-2.5, 0., float("inf")]:
        actual, = interpreter.call_op("review", (scalar,))
        torch.testing.assert_close(torch.tensor(actual.data).reshape(actual.shape),
                                   torch.full((2, 3), scalar))


def test_raw_max_pool_preserves_both_results():
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    expected = torch.ops.aten.max_pool2d_with_indices.default(x, [2, 2])
    block = Block(arg_types=[TensorType(f32, list(x.shape))])
    meta = {"val": expected, "_aten_target": "aten.max_pool2d_with_indices.default",
            "_fx_args": (object(), [2, 2])}
    result = DECOMPOSITION_TABLE[meta["_aten_target"]](list(block.args), meta, "review")
    assert not any(op.name == "func.call" for op in result.ops)
    assert len(result.results) == 2
    for actual, oracle in zip(result.results, expected, strict=True):
        assert list(actual.type.get_shape()) == list(oracle.shape)


@pytest.mark.parametrize("operand_count", [0, 1, 2])
def test_opaque_fallback_uses_result_dtype_without_assuming_two_operands(operand_count):
    block = Block(arg_types=[TensorType(f32, [2])] * operand_count)
    result = _opaque_decomp("review_unsupported", list(block.args),
                            {"val": torch.zeros(2, dtype=torch.bool)}, "review")
    assert result.result.type.element_type.bitwidth == 1


@pytest.mark.parametrize("kind", ["linear", "conv2d"])
def test_pt2e_integerization_preserves_keyword_bias_and_geometry(kind):
    import torch.ao.quantization.fx._decomposed  # Register portable Q/DQ ops.
    from m2m.capture.pt2e_integerize import integerize_pt2e

    # Build the portable FX form explicitly: symbolic_trace evaluates Q/DQ
    # on a buffer eagerly and would remove the weight dequantize we must match.
    graph = torch.fx.Graph()
    x, weight, bias = [graph.placeholder(name) for name in ("x", "weight", "bias")]
    dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    lhs = graph.call_function(dq, (x, 0.125, 0, -128, 127, torch.int8))
    rhs = graph.call_function(dq, (weight, 0.25, 0, -128, 127, torch.int8))
    target = torch.ops.aten.linear.default if kind == "linear" else torch.ops.aten.conv2d.default
    kwargs = {"bias": bias}
    if kind == "conv2d":
        kwargs.update(stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1)
    graph.output(graph.call_function(target, (lhs, rhs), kwargs))
    model = torch.fx.GraphModule(torch.nn.Module(), graph)
    shape = (2, 4) if kind == "linear" else (1, 4, 4, 4)
    weight_shape = (3, 4) if kind == "linear" else (3, 4, 2, 2)
    inputs = (torch.ones(shape, dtype=torch.int8), torch.ones(weight_shape, dtype=torch.int8),
              torch.tensor([1., 2., 3.]))
    expected = model(*inputs)
    rewritten, receipt = integerize_pt2e(model, inputs)
    assert receipt[f"{kind}_integerized"] == 1, receipt
    torch.testing.assert_close(rewritten(*inputs), expected)


def test_dropout_drop_all_preserves_nan_inf_and_signed_zero():
    x = torch.tensor([float("nan"), float("inf"), float("-inf"), -2., 0., -0., 3.])
    expected = torch.ops.aten.native_dropout.default(x, 1.0, True)
    # Training endpoints consume the CPU generator stream too.
    from test_core_aten_rng_ops import as_array, interpreter_for
    interpreter, _ = interpreter_for(lambda x: torch.ops.aten.native_dropout.default(x, 1.0, True), (x,))
    values = interpreter.call_op("forward", (as_array(x, interpreter),))
    actual = [torch.tensor(value.data, dtype=oracle.dtype).reshape(value.shape)
              for value, oracle in zip(values, expected, strict=True)]
    for observed, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, oracle, equal_nan=True)
    assert torch.equal(torch.signbit(actual[0][3:]), torch.signbit(expected[0][3:]))
