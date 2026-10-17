"""SmolVLA's pinned pretrained loader preserves LeRobot config choice semantics."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


pytest.importorskip("lerobot")


def _loader():
    path = Path(__file__).resolve().parents[1] / "workloads" / "smolvla" / "loader.py"
    spec = importlib.util.spec_from_file_location("smolvla_loader_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pretrained_config_uses_registered_dispatcher_and_exact_revision(monkeypatch):
    loader = _loader()
    expected = loader.SmolVLAConfig()
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(loader.PreTrainedConfig, "from_pretrained", load)
    assert loader._load_pretrained_config() is expected
    assert calls == [((loader._MODEL_ID,), {
        "revision": loader._MODEL_REVISION, "local_files_only": True})]
    assert len(loader._MODEL_REVISION) == 40


def test_pretrained_config_rejects_wrong_registered_subclass(monkeypatch):
    loader = _loader()
    monkeypatch.setattr(loader.PreTrainedConfig, "from_pretrained", lambda *args, **kwargs: object())
    with pytest.raises(TypeError, match="expected SmolVLAConfig"):
        loader._load_pretrained_config()


def test_deferred_stagewise_capability_passes_one_source_and_explicit_calibration(monkeypatch):
    from torch import nn
    from m2m.capture import stagewise_pt2e

    loader = _loader()
    shared = nn.Linear(2, 2)
    capture = loader.SmolVLAMultiProgramCapture(
        shared, (), paper_ready=False, full_checkpoint=False, provenance={})
    source_session = object()
    monkeypatch.setattr(capture, "external_runtime_session", lambda: source_session)
    calls = []

    def convert(*args, **kwargs):
        calls.append((args, kwargs))
        return "converted"

    monkeypatch.setattr(stagewise_pt2e, "quantize_stagewise_pt2e_session", convert)
    quant = object()
    calibration = {"prefix_encode": [()], "flow_denoise": [()]}
    snapshot = {"fixture": "explicit source owner"}
    assert capture.shared_model is shared
    assert capture.stagewise_pt2e_session(
        quant, calibration_inputs=calibration, source_snapshot=snapshot) == "converted"
    assert calls == [((shared, source_session), {
        "quant": quant,
        "shared_programs": ("prefix_encode", "flow_denoise"),
        "calibration_inputs": calibration,
        "source_snapshot": snapshot,
    })]


def test_float32_model_converts_every_floating_state_and_exports_no_bfloat16():
    import torch

    loader = _loader()

    class Mixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4).to(torch.bfloat16)
            self.head = torch.nn.Linear(4, 2)
            self.register_buffer("scale", torch.ones(4, dtype=torch.bfloat16))
            self.register_buffer("index", torch.arange(4))

        def forward(self, x):
            hidden = self.backbone(x.to(self.backbone.weight.dtype) * self.scale)
            return self.head(hidden.to(self.head.weight.dtype))

    model = loader._float32_model(Mixed().eval())
    floating = [t for t in (*model.parameters(), *model.buffers()) if t.is_floating_point()]
    assert floating and all(t.dtype == torch.float32 for t in floating)
    assert model.index.dtype == torch.long
    exported = torch.export.export(model, (torch.randn(1, 4),))
    dtypes = {str(getattr(node.meta.get("val"), "dtype", None)) for node in exported.graph.nodes}
    assert "torch.bfloat16" not in dtypes


def test_float32_model_refuses_floating_state_it_cannot_convert():
    import torch

    loader = _loader()

    class Stubborn(torch.nn.Linear):
        def to(self, *args, **kwargs):  # a conversion that silently keeps the narrow weights
            return self

    with pytest.raises(ValueError, match="non-float32 floating state"):
        loader._float32_model(Stubborn(2, 2).to(torch.float32).bfloat16())
