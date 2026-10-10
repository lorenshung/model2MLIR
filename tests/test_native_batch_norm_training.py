"""Training batch-normalization capture, ABI, and numerical semantics."""

from __future__ import annotations

import math

import torch
from xdsl.dialects import arith, math as math_dialect, tensor
from xdsl.dialects.builtin import f32
from xdsl.dialects.linalg.ops import ReduceOp
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.builtin import xtype_for_el_type
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.linalg import LinalgFunctions
from xdsl.interpreters.math import MathFunctions
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.tensor import TensorFunctions
from xdsl.interpreters.utils.ptr import TypedPtr

import m2m
from m2m.capture.torch_export import capture_frontend_artifact


class _TrainingBatchNorm(torch.nn.Module):
    def __init__(self, affine: bool = True) -> None:
        super().__init__()
        self.affine = affine

    def forward(self, x, weight, bias, running_mean, running_var):
        return torch.ops.aten._native_batch_norm_legit.default(
            x,
            weight if self.affine else None,
            bias if self.affine else None,
            running_mean,
            running_var,
            True,
            0.1,
            1e-5,
        )


def _inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260924)
    return (
        torch.randn(2, 4, 2, 2, generator=generator),
        torch.randn(4, generator=generator),
        torch.randn(4, generator=generator),
        torch.randn(4, generator=generator),
        torch.rand(4, generator=generator) + 0.5,
    )


def _clones(values):
    return tuple(value.clone() for value in values)


def test_training_batch_norm_capture_preserves_mutation_signature():
    artifact = capture_frontend_artifact(_TrainingBatchNorm(), _clones(_inputs()))
    exported = artifact.exported_program
    assert exported is not None

    targets = [str(node.target) for node in exported.graph.nodes if node.op == "call_function"]
    assert "aten._native_batch_norm_legit_functional.default" in targets

    output_specs = list(exported.graph_signature.output_specs)
    assert [str(spec.kind).rsplit(".", 1)[-1] for spec in output_specs[:2]] == [
        "USER_INPUT_MUTATION",
        "USER_INPUT_MUTATION",
    ]
    assert [spec.target for spec in output_specs[:2]] == ["running_mean", "running_var"]
    assert [str(spec.kind).rsplit(".", 1)[-1] for spec in output_specs[2:]] == [
        "USER_OUTPUT",
        "USER_OUTPUT",
        "USER_OUTPUT",
    ]


def test_training_batch_norm_lowers_all_five_results_without_opaque_calls():
    result = m2m.convert(
        _TrainingBatchNorm(), _clones(_inputs()), backend="fx_importer"
    )
    assert result.ok
    assert result.module is not None
    result.module.verify()
    assert "func.call" not in result.mlir_text
    assert 'prov.aten = "aten._native_batch_norm_legit_functional.default"' in result.mlir_text
    assert (
        "-> (tensor<4xf32>, tensor<4xf32>, tensor<2x4x2x2xf32>, "
        "tensor<4xf32>, tensor<4xf32>)"
    ) in result.mlir_text


def test_training_batch_norm_supports_absent_affine_parameters():
    result = m2m.convert(
        _TrainingBatchNorm(affine=False), _clones(_inputs()), backend="fx_importer"
    )
    assert result.ok
    assert result.module is not None
    result.module.verify()
    assert "func.call" not in result.mlir_text
    assert 'prov.aten = "aten._native_batch_norm_legit_functional.default"' in result.mlir_text


@register_impls
class _BatchNormInterpreterExtras(InterpreterFunctions):
    """Small missing pieces in xDSL 0.69's standard-dialect interpreter."""

    @impl(arith.DivfOp)
    def run_divf(self, interpreter, op, args):
        return (args[0] / args[1],)

    @impl(math_dialect.RsqrtOp)
    def run_rsqrt(self, interpreter, op, args):
        return (1.0 / math.sqrt(args[0]),)

    @impl(tensor.SplatOp)
    def run_splat(self, interpreter, op, args):
        result_type = op.result.type
        shape = list(result_type.get_shape())
        element_type = xtype_for_el_type(
            result_type.element_type, interpreter.index_bitwidth
        )
        return (
            ShapedArray(
                TypedPtr.new([args[0]] * math.prod(shape), xtype=element_type),
                shape,
            ),
        )

    @impl(ReduceOp)
    def run_reduce(self, interpreter, op, args):
        source, initial = args
        output = initial.copy()
        reduction_dims = set(op.dimensions.iter_values())
        for source_index in source.indices():
            output_index = tuple(
                value
                for dim, value in enumerate(source_index)
                if dim not in reduction_dims
            )
            (value,) = interpreter.run_ssacfg_region(
                op.regions[0],
                (source.load(source_index), output.load(output_index)),
                "batch_norm_reduce",
            )
            output.store(output_index, value)
        return (output,)


def _as_shaped_array(value: torch.Tensor, interpreter: Interpreter):
    element_type = xtype_for_el_type(f32, interpreter.index_bitwidth)
    return ShapedArray(
        TypedPtr.new(value.flatten().tolist(), xtype=element_type),
        list(value.shape),
    )


def test_training_batch_norm_numerics_and_mutations_match_eager():
    original = _inputs()
    eager_inputs = _clones(original)
    eager_outputs = _TrainingBatchNorm()(*eager_inputs)

    # Export executes the stateful operator while tracing, so give conversion its own
    # copies and retain pristine inputs for executing the generated pure function.
    result = m2m.convert(
        _TrainingBatchNorm(), _clones(original), backend="fx_importer"
    )
    assert result.module is not None
    interpreter = Interpreter(result.module)
    for implementations in (
        ArithFunctions(),
        FuncFunctions(),
        LinalgFunctions(),
        MathFunctions(),
        TensorFunctions(),
        _BatchNormInterpreterExtras(),
    ):
        interpreter.register_implementations(implementations)

    actual = interpreter.call_op(
        "forward",
        tuple(_as_shaped_array(value, interpreter) for value in original),
    )
    expected = (eager_inputs[3], eager_inputs[4], *eager_outputs)
    assert len(actual) == len(expected) == 5
    for actual_value, expected_value in zip(actual, expected, strict=True):
        actual_tensor = torch.tensor(actual_value.data).reshape(actual_value.shape)
        torch.testing.assert_close(actual_tensor, expected_value, rtol=2e-5, atol=2e-6)
