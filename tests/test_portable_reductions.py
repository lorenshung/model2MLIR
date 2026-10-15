"""Structured arithmetic must not encode the compiler host's reduction kernels."""

import platform

import pytest
import torch
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

import m2m
from m2m.ir.decompositions import build_layer_norm_body


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("operation", ["layer_norm", "mean", "rsqrt", "group_norm", "embedding_bag"])
def test_ir_is_independent_of_host_cpu(monkeypatch, dtype, operation):
    class Model(torch.nn.Module):
        def forward(self, x):
            if operation == "layer_norm":
                return torch.ops.aten.native_layer_norm.default(x, [33], None, None, 1e-5)
            if operation == "mean":
                return x.mean()
            if operation == "rsqrt":
                return torch.rsqrt(x)
            if operation == "group_norm":
                return torch.ops.aten.native_group_norm.default(x, None, None, 2, 4, 33, 2, 1e-5)
            return torch.ops.aten._embedding_bag.default(
                x, torch.tensor([0, 1]), torch.tensor([0]), False, 0, False, None, False, -1
            )

    shape = (2, 4, 33) if operation == "group_norm" else (2, 33)
    inputs = (torch.linspace(.1, 2., torch.tensor(shape).prod().item()).reshape(shape).to(dtype),)
    exported = torch.export.export(Model(), inputs).module()
    from m2m.capture.pt2e_compat import allow_exported_model_train_eval
    allow_exported_model_train_eval(exported)
    texts = []
    for machine, capability in [("x86_64", "AVX2"), ("x86_64", "AVX512"), ("aarch64", "DEFAULT")]:
        monkeypatch.setattr(platform, "machine", lambda: machine)
        monkeypatch.setattr(torch.backends.cpu, "get_cpu_capability", lambda: capability)
        result = m2m.convert(exported, inputs, backend="fx_importer")
        assert result.ok
        result.module.verify()
        names = [op.name for op in result.module.walk()]
        assert "func.call" not in names
        if operation in ("layer_norm", "group_norm"):
            assert names.count("linalg.reduce") == 2
            assert "f64" not in result.mlir_text
        elif operation == "mean":
            assert names.count("linalg.reduce") == 1
        elif operation == "rsqrt":
            assert names.count("math.rsqrt") == 1
            assert "arith.select" not in names
            assert "linalg.index" not in names
            if dtype != torch.float32:
                assert "arith.extf" in names and "arith.truncf" in names
                root = next(op for op in result.module.walk() if op.name == "math.rsqrt")
                assert root.results[0].type == builtin.f32
        elif operation == "embedding_bag":
            assert "arith.bitcast" not in names
            if dtype != torch.float32:
                assert "arith.truncf" in names
        texts.append(result.mlir_text)
    assert texts[0] == texts[1] == texts[2]


@pytest.mark.parametrize("width", [33, 4096, 16384])
def test_native_f32_layer_norm_has_two_small_structured_reductions(width):
    typ = builtin.TensorType(builtin.f32, [2, width])
    block = Block(arg_types=[typ])
    ops, output, mean, rstd = build_layer_norm_body(
        block.args[0], None, None, eps=1e-5, k=1, return_stats=True
    )
    block.add_ops(ops)
    block.add_op(func.ReturnOp(output, mean, rstd))
    module = builtin.ModuleOp([func.FuncOp(
        "forward", ([typ], [output.type, mean.type, rstd.type]), region=Region(block)
    )])
    module.verify()
    operations = list(module.walk())
    assert sum(op.name == "linalg.reduce" for op in operations) == 2
    assert not any(op.name == "scf.for" for op in operations)
    assert len(operations) < 100
    assert len(str(module).splitlines()) < 120
