"""Portable IR retains the captured source CPU's rounding and reduction order."""

import platform

import pytest
import torch
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

import m2m
from m2m.ir.decompositions import build_layer_norm_body


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('operation', ['layer_norm', 'mean', 'rsqrt', 'group_norm', 'embedding_bag'])
def test_source_cpu_ir_is_independent_of_platform_name(monkeypatch, dtype, operation):
    class Model(torch.nn.Module):
        def forward(self, x):
            if operation == 'layer_norm':
                return torch.ops.aten.native_layer_norm.default(x, [33], None, None, 1e-5)
            if operation == 'mean':
                return x.mean()
            if operation == 'rsqrt':
                return torch.rsqrt(x)
            if operation == 'group_norm':
                return torch.ops.aten.native_group_norm.default(x, None, None, 2, 4, 33, 2, 1e-5)
            return torch.ops.aten._embedding_bag.default(
                x, torch.tensor([0, 1]), torch.tensor([0]), False, 0, False, None, False, -1)

    shape = (2, 4, 33) if operation == 'group_norm' else (2, 33)
    inputs = (torch.linspace(.1, 2., torch.tensor(shape).prod().item()).reshape(shape).to(dtype),)
    exported = torch.export.export(Model(), inputs).module()
    from m2m.capture.pt2e_compat import allow_exported_model_train_eval
    allow_exported_model_train_eval(exported)
    texts = []
    # Changing the source CPU capability changes its Welford lanes and scalar
    # tails. Keep that source policy fixed while checking platform independence.
    for machine in ['x86_64', 'aarch64', 'riscv64']:
        monkeypatch.setattr(platform, 'machine', lambda: machine)
        result = m2m.convert(exported, inputs, backend='fx_importer')
        assert result.ok
        result.module.verify()
        names = [op.name for op in result.module.walk()]
        assert 'func.call' not in names
        if operation in ('layer_norm', 'group_norm'):
            assert 'scf.for' in names and 'math.fma' in names
            assert 'math.sqrt' in names
        elif operation == 'mean':
            assert ('linalg.reduce' if dtype == torch.float32 else 'scf.for') in names
        elif operation == 'rsqrt':
            if dtype == torch.float32:
                assert 'math.rsqrt' in names
            else:
                assert 'math.sqrt' in names and 'arith.divf' in names
                assert 'arith.extf' in names and 'arith.truncf' in names
        elif operation == 'embedding_bag':
            if dtype == torch.bfloat16:
                # The native contiguous fast path has a distinct tie conversion.
                assert 'arith.bitcast' in names
            elif dtype == torch.float16:
                assert 'arith.truncf' in names
        texts.append(result.mlir_text)
    assert texts[0] == texts[1] == texts[2]


@pytest.mark.parametrize('width', [33, 4096, 16384])
def test_native_f32_layer_norm_uses_ordered_welford_loops(width):
    typ = builtin.TensorType(builtin.f32, [2, width])
    block = Block(arg_types=[typ])
    ops, output, mean, rstd = build_layer_norm_body(
        block.args[0], None, None, eps=1e-5, k=1, return_stats=True)
    block.add_ops(ops)
    block.add_op(func.ReturnOp(output, mean, rstd))
    module = builtin.ModuleOp([func.FuncOp(
        'forward', ([typ], [output.type, mean.type, rstd.type]), region=Region(block))])
    module.verify()
    operations = list(module.walk())
    assert any(op.name == 'scf.for' for op in operations)
    assert any(op.name == 'math.fma' for op in operations)
    assert not any(op.name == 'linalg.reduce' for op in operations)
    loads = [op for op in operations if op.name == 'tensor.extract']
    assert loads
    for load in loads:
        parent = load.parent_op()
        while parent is not None and parent.name != 'scf.for':
            parent = parent.parent_op()
        assert parent is not None, 'Welford values must be loaded within the ordered loops'
