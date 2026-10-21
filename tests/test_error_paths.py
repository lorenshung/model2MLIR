"""Expected failures take their fallback; any other exception surfaces."""
import pytest
import torch
from torch._subclasses.fake_tensor import DataDependentOutputException

import m2m


class Rows(torch.nn.Module):
    def forward(self, x):
        rows, _ = x.shape  # FX cannot iterate a Proxy (TraceError)
        return torch.ops.aten.scalar_tensor.default(rows)


class Length(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten.scalar_tensor.default(len(x))  # FX refuses len() (RuntimeError)


class Numel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten.scalar_tensor.default(torch.ops.aten.sym_numel.default(x))


@pytest.mark.parametrize("model", [Rows(), Length()])
def test_untraceable_source_keeps_the_exported_program(model):
    from m2m.capture.torch_export import _retain_static_queries
    x = torch.ones(2, 3)
    exported = torch.export.export(model, (x,))
    assert _retain_static_queries(model, (x,), exported) is exported


@pytest.mark.parametrize("error,kept", [
    (DataDependentOutputException(torch.ops.aten.nonzero.default), True), (ValueError("recovery bug"), False)])
def test_only_typed_fake_propagation_failures_keep_the_exported_program(monkeypatch, error, kept):
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp
    from m2m.capture.torch_export import _retain_static_queries
    def propagate(*_):
        raise error
    monkeypatch.setattr(FakeTensorProp, "propagate", propagate)
    x = torch.ones(2, 3)
    exported = torch.export.export(Numel(), (x,))
    if kept:
        assert _retain_static_queries(Numel(), (x,), exported) is exported
    else:
        with pytest.raises(ValueError, match="recovery bug"):
            _retain_static_queries(Numel(), (x,), exported)
