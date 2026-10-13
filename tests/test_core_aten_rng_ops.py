"""Execute the emitted CPU RNG IR against independently seeded eager ATen."""

import math as pymath

import m2m
import numpy as np
import pytest
import torch
from test_core_aten_review_regressions import _ReviewInterpreterExtras
from test_native_batch_norm_training import _BatchNormInterpreterExtras
from xdsl.dialects import arith, math, ml_program
from xdsl.dialects.builtin import UnregisteredOp
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    impl,
    impl_terminator,
    register_impls,
)
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.builtin import xtype_for_el_type
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.linalg import LinalgFunctions
from xdsl.interpreters.math import MathFunctions
from xdsl.interpreters.scf import ScfFunctions
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.tensor import TensorFunctions
from xdsl.interpreters.utils.ptr import TypedPtr


@register_impls
class RNGFunctions(InterpreterFunctions):
    def __init__(self, module, interpreter):
        self.globals = {}
        for op in module.body.block.ops:
            if isinstance(op, ml_program.GlobalOp):
                typ = op.type
                self.globals[op.sym_name.data] = ShapedArray(
                    TypedPtr.new(
                        list(op.value.get_values()),
                        xtype=xtype_for_el_type(
                            typ.element_type, interpreter.index_bitwidth
                        ),
                    ),
                    list(typ.get_shape()),
                )

    def global_access(self, interpreter, op, args):
        name = op.properties["global"].root_reference.data
        if op.op_name.data == "ml_program.global_load":
            return (self.globals[name].copy(),)
        assert op.op_name.data == "ml_program.global_store"
        self.globals[name] = args[0].copy()
        return ()

    @impl(arith.ShRUIOp)
    def shift(self, interpreter, op, args):
        return ((args[0] & ((1 << 64) - 1)) >> args[1],)

    @impl(arith.RemUIOp)
    def remainder(self, interpreter, op, args):
        return ((args[0] & ((1 << 64) - 1)) % args[1],)

    @impl(arith.DivUIOp)
    def divide(self, interpreter, op, args):
        return ((args[0] & ((1 << 64) - 1)) // args[1],)

    @impl(arith.UIToFPOp)
    def to_float(self, interpreter, op, args):
        return (float(args[0] & ((1 << 64) - 1)),)

    @impl(arith.TruncFOp)
    def truncate(self, interpreter, op, args):
        return (float(np.float32(args[0])),)

    @impl(math.Log1pOp)
    def log1p(self, interpreter, op, args):
        return (pymath.log1p(args[0]),)

    @impl(math.SinOp)
    def sin(self, interpreter, op, args):
        return (pymath.sin(args[0]),)

    @impl(math.CosOp)
    def cos(self, interpreter, op, args):
        return (pymath.cos(args[0]),)


class Module(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def interpreter_for(fn, inputs=(), seed=0):
    result = m2m.convert(Module(fn), inputs, backend="fx_importer", rng_seed=seed)
    assert result.ok
    assert result.exported_program is not None
    assert "func.call" not in result.mlir_text
    result.module.verify()
    interpreter = Interpreter(result.module)
    for functions in (
        ArithFunctions(),
        FuncFunctions(),
        ScfFunctions(),
        TensorFunctions(),
        LinalgFunctions(),
        MathFunctions(),
        _BatchNormInterpreterExtras(),
    ):
        interpreter.register_implementations(functions)
    interpreter.register_implementations(_ReviewInterpreterExtras(), override=True)
    rng = RNGFunctions(result.module, interpreter)
    interpreter.register_implementations(rng, override=True)
    for cls in {
        type(op) for op in result.module.walk() if isinstance(op, UnregisteredOp)
    }:

        def access(self, interpreter, op, args):
            return None, rng.global_access(interpreter, op, args)

        functions = register_impls(
            type(
                "GlobalAccess",
                (InterpreterFunctions,),
                {
                    "access": impl_terminator(cls)(access),
                },
            )
        )
        interpreter.register_implementations(functions())
    return interpreter, rng


def as_array(value, interpreter):
    from m2m.ir.import_fx import _torch_dtype_to_xdsl

    typ = _torch_dtype_to_xdsl(value.dtype)
    return ShapedArray(
        TypedPtr.new(
            value.flatten().tolist(),
            xtype=xtype_for_el_type(typ, interpreter.index_bitwidth),
        ),
        list(value.shape),
    )


@pytest.mark.parametrize("seed", [0, 17, (1 << 63) + 23])
def test_mixed_stream_matches_eager_across_repeated_calls(seed):
    def fn(x):
        return (
            torch.rand(2, 3),
            torch.randn(5),
            torch.randn(3),
            torch.randperm(7),
            *torch.ops.aten.native_dropout(x, 0.25, True),
            torch.rand(4, dtype=torch.float64),
        )

    x = torch.linspace(-2, 2, 6).reshape(2, 3)
    interpreter, rng = interpreter_for(fn, (x,), seed)
    torch.manual_seed(seed)
    for _ in range(2):
        expected = fn(x)
        actual = interpreter.call_op("forward", (as_array(x, interpreter),))
        for value, eager in zip(actual, expected, strict=True):
            got = torch.tensor(value.data, dtype=eager.dtype).reshape(value.shape)
            torch.testing.assert_close(got, eager, rtol=1e-5, atol=1e-6)
        # Engine consumption is checked as well as the visible samples.
        eager_cursor = int.from_bytes(
            bytes(torch.get_rng_state()[16:20].tolist()), "little"
        )
        assert rng.globals["global_seed"].data[-1] == eager_cursor


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_uniform_crosses_twist_boundary_and_empty_draw(dtype):
    fn = lambda: (torch.rand(0, dtype=dtype), torch.rand(635, dtype=dtype))
    interpreter, _ = interpreter_for(fn, seed=42)
    torch.manual_seed(42)
    expected = fn()
    actual = interpreter.call_op("forward", ())
    for value, eager in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            torch.tensor(value.data, dtype=dtype), eager, rtol=0, atol=0
        )


def test_inference_dropout_preserves_stream():
    fn = lambda x: (*torch.ops.aten.native_dropout(x, 0.25, False), torch.rand(3))
    x = torch.ones(3)
    interpreter, _ = interpreter_for(fn, (x,))
    torch.manual_seed(0)
    actual = interpreter.call_op("forward", (as_array(x, interpreter),))
    for value, eager in zip(actual, fn(x), strict=True):
        torch.testing.assert_close(torch.tensor(value.data, dtype=eager.dtype), eager)


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_training_dropout_endpoints_still_consume_bernoulli_draws(p):
    fn = lambda x: (*torch.ops.aten.native_dropout(x, p, True), torch.rand(3))
    x = torch.ones(3)
    interpreter, _ = interpreter_for(fn, (x,))
    torch.manual_seed(0)
    actual = interpreter.call_op("forward", (as_array(x, interpreter),))
    for value, eager in zip(actual, fn(x), strict=True):
        torch.testing.assert_close(torch.tensor(value.data, dtype=eager.dtype), eager)


def test_conversion_exposes_the_imported_program_without_precapture(monkeypatch):
    from m2m.ir.import_fx import FXImporter

    captured = []
    original = FXImporter.import_graph

    def record(self, exported):
        captured.append(exported)
        return original(self, exported)

    monkeypatch.setattr(FXImporter, "import_graph", record)
    result = m2m.convert(
        Module(lambda: torch.rand(3)), (), backend="fx_importer", decompose=False
    )
    assert result.ok
    assert result.exported_program is captured[0]


def test_rng_inside_loop_body_shares_one_module_global():
    class Loop(torch.nn.Module):
        def forward(self, x):
            def cond(i, value):
                return i < 2

            def body(i, value):
                return i + 1, value + torch.rand_like(value)

            return torch.while_loop(cond, body, (torch.tensor(0), x))[1]

    result = m2m.convert(Loop(), (torch.zeros(2),), backend="fx_importer")
    assert result.ok
    assert "func.call" not in result.mlir_text
    assert result.mlir_text.count("ml_program.global private mutable @global_seed") == 1
    assert "ml_program.global_load" in result.mlir_text


@pytest.mark.parametrize('with_normal', [False, True])
def test_traced_rng_globals_have_only_exact_live_user_lineage(with_normal):
    class Model(torch.nn.Module):
        def forward(self, x):
            uniform = torch.rand_like(x)
            following = torch.randn_like(x) if with_normal else torch.rand_like(x)
            return uniform, following

    result = m2m.convert(Model(), (torch.zeros(3),), backend='fx_importer', capture_trace=True)
    assert result.ok, result.diagnostics
    assert result.capture_trace['status'] == 'complete', result.capture_trace['blockers']
    declarations = {op.sym_name.data: op for op in result.module.body.block.ops
                    if isinstance(op, ml_program.GlobalOp)}
    assert set(declarations) == ({'global_seed', 'global_normal_cache'} if with_normal else {'global_seed'})
    def ids(op, key):
        return {value.data for value in op.attributes.get(key, [])}
    for name, declaration in declarations.items():
        users = [op for op in result.module.walk() if isinstance(op, UnregisteredOp)
                 and op.op_name.data in {'ml_program.global_load', 'ml_program.global_store'}
                 and op.properties['global'].root_reference.data == name]
        for key in ['prov.source_node_ids', 'prov.origin_node_ids']:
            expected = set().union(*(ids(user, key) for user in users))
            assert expected and ids(declaration, key) == expected
        assert declaration.attributes['prov.trace_role'].data == 'support'
    assert ('@global_normal_cache' in result.module.attributes['prov.rng_state'].data) == with_normal
