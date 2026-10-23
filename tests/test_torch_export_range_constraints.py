from types import SimpleNamespace

import pytest
import sympy
import torch
from torch.utils._sympy.numbers import int_oo

from m2m.capture.torch_export import (
    RangeConstraint,
    _prepare_exported_program,
    _serialize_range_constraints,
)


def _export_with_constraints(constraints):
    return SimpleNamespace(range_constraints=constraints)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (-sympy.oo, sympy.oo),
        (-int_oo, int_oo),
        (float("-inf"), float("inf")),
    ],
)
def test_serialize_range_constraints_maps_infinite_bounds_to_none(lower, upper):
    exported = _export_with_constraints(
        {sympy.Symbol("unbounded"): SimpleNamespace(lower=lower, upper=upper)}
    )

    assert _serialize_range_constraints(exported) == (
        RangeConstraint(symbol="unbounded", minimum=None, maximum=None),
    )


def test_serialize_range_constraints_preserves_finite_bounds():
    exported = _export_with_constraints(
        {
            sympy.Symbol("bounded"): SimpleNamespace(
                lower=sympy.Integer(-7), upper=sympy.Integer(11)
            )
        }
    )

    assert _serialize_range_constraints(exported) == (
        RangeConstraint(symbol="bounded", minimum=-7, maximum=11),
    )


def test_cosh_decomposition_serializes_unbounded_shape_constraints():
    class Cosh(torch.nn.Module):
        def forward(self, value):
            return torch.ops.aten.cosh.default(value)

    # A view input makes export retain storage-shape symbols.  Decomposing
    # cosh gives those symbols an ``int_oo`` upper bound in the pinned PyTorch.
    view_input = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    original = torch.export.export(Cosh(), (view_input,))
    prepared, _ = _prepare_exported_program(
        original,
        run_default_decompositions=False,
        export_decomposition_table={
            torch.ops.aten.cosh.default: torch._decomp.decomposition_table[
                torch.ops.aten.cosh.default
            ]
        },
    )

    constraints = _serialize_range_constraints(prepared)

    assert constraints
    assert all(constraint.maximum is None for constraint in constraints)
    assert all(constraint.minimum == 2 for constraint in constraints)


@pytest.mark.parametrize('capture_trace', [False, True])
def test_decomposing_conversion_does_not_change_later_raw_conversion(capture_trace):
    import m2m
    from m2m.ir.decompositions import DECOMPOSITION_TABLE

    class Hardtanh(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.hardtanh.default(x, -1.5, 2.)

    inputs = (torch.tensor([-3., 0., 4.]),)
    table_before = dict(DECOMPOSITION_TABLE)
    before = m2m.convert(Hardtanh(), inputs, backend='fx_importer', decompose=False)
    decomposed = m2m.convert(Hardtanh(), inputs, backend='fx_importer', capture_trace=capture_trace)
    after = m2m.convert(Hardtanh(), inputs, backend='fx_importer', decompose=False)
    assert before.ok and decomposed.ok and after.ok
    assert 'func.call' not in decomposed.mlir_text
    assert 'prov.aten = "aten.hardtanh.default"' in decomposed.mlir_text
    assert DECOMPOSITION_TABLE == table_before
    assert before.mlir_text == after.mlir_text


def test_core_lowerings_belong_to_capture_artifact():
    from m2m.capture.torch_export import capture_frontend_artifact
    from m2m.ir.decompositions import DECOMPOSITION_TABLE

    class Hardtanh(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.hardtanh.default(x)

    before = dict(DECOMPOSITION_TABLE)
    artifact = capture_frontend_artifact(Hardtanh(), (torch.ones(2),), preserve_supported_ops=True)
    assert 'aten.hardtanh.default' in artifact.importer_decompositions
    assert 'aten.hardtanh.default' in artifact.strict_import_options()['dynamic_decompositions']
    assert DECOMPOSITION_TABLE == before


@pytest.mark.parametrize('already_registered', [False, True])
@pytest.mark.parametrize('retry_fails', [False, True])
def test_metadata_export_retry_restores_module_registration(monkeypatch, already_registered, retry_fails):
    import sys
    import types
    from m2m.capture.torch_export import capture_model

    module_name = '_m2m_test_metadata_loader'
    namespace = {'__name__': module_name, 'torch': torch}
    exec('class Resize(torch.nn.Module):\n'
         '    def forward(self, x):\n'
         '        return torch.ops.aten.resize_.default(x, [3, 2])\n', namespace)
    model = namespace['Resize']()
    registered = types.ModuleType(module_name)
    if already_registered:
        monkeypatch.setitem(sys.modules, module_name, registered)
    else:
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    attempts = []
    expected = SimpleNamespace(graph=SimpleNamespace(nodes=[]))

    def export(model, inputs, **kwargs):
        attempts.append(list(inputs[0].shape))
        if len(attempts) == 1:
            inputs[0].resize_(3, 2)
            raise AssertionError('sources must not be empty for symbol s1')
        assert kwargs['strict'] is True
        if already_registered:
            assert sys.modules[module_name] is registered
        else:
            assert sys.modules[module_name].Resize is type(model)
            assert sys.modules[module_name].torch is torch
        if retry_fails:
            raise RuntimeError('strict retry failed')
        return expected

    monkeypatch.setattr(torch.export, 'export', export)
    x = torch.arange(6.).reshape(2, 3)
    if retry_fails:
        with pytest.raises(RuntimeError, match='strict retry failed'):
            capture_model(model, (x,))
    else:
        assert capture_model(model, (x,)) is expected
    assert attempts == [[2, 3], [2, 3]]
    assert x.shape == (2, 3)
    assert sys.modules.get(module_name) is (registered if already_registered else None)


def test_unrelated_export_error_does_not_inject_module_or_retry(monkeypatch):
    import sys
    from m2m.capture.torch_export import capture_model

    class Model(torch.nn.Module):
        def forward(self, x):
            return x

    module_name = '_m2m_unrelated_export_error'
    Model.__module__ = module_name
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    attempts = []

    def export(*args, **kwargs):
        attempts.append(kwargs)
        raise RuntimeError('unrelated export failure')

    monkeypatch.setattr(torch.export, 'export', export)
    with pytest.raises(RuntimeError, match='unrelated export failure'):
        capture_model(Model(), (torch.ones(2),))
    assert len(attempts) == 1
    assert module_name not in sys.modules
