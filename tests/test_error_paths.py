"""Sources FX cannot trace or propagate keep the exported program."""
import pytest
import torch
from torch._subclasses.fake_tensor import DataDependentOutputException

st = torch.ops.aten.scalar_tensor.default


class Branch(torch.nn.Module):
    def forward(self, x):
        return st(1 if x.shape[0] > 1 else 0)  # FX cannot branch on a Proxy (TraceError)


class Length(torch.nn.Module):
    def forward(self, x):
        return st(len(x))  # FX refuses len() (RuntimeError)


class Float(torch.nn.Module):
    def forward(self, x):
        return st(float(x.size(0)))  # float() of a Proxy (TypeError)


class Range(torch.nn.Module):
    def forward(self, x):
        return st(sum(range(x.shape[0])))  # range() of a Proxy (TypeError)


class Numel(torch.nn.Module):
    def forward(self, x):
        return st(torch.ops.aten.sym_numel.default(x))


@pytest.mark.parametrize("model", [Branch(), Length(), Float(), Range()])
def test_untraceable_source_keeps_the_exported_program(model):
    from m2m.capture.torch_export import _retain_static_queries
    x = torch.ones(2, 3)
    exported = torch.export.export(model, (x,))
    assert _retain_static_queries(model, (x,), exported) is exported


@pytest.mark.parametrize("error", [DataDependentOutputException(torch.ops.aten.nonzero.default), ValueError("shape")])
def test_fake_propagation_failure_keeps_the_exported_program(monkeypatch, error):
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp
    from m2m.capture.torch_export import _retain_static_queries
    def propagate(*_):
        raise error
    monkeypatch.setattr(FakeTensorProp, "propagate", propagate)
    x = torch.ones(2, 3)
    exported = torch.export.export(Numel(), (x,))
    assert _retain_static_queries(Numel(), (x,), exported) is exported
