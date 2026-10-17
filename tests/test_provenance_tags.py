"""Printed MLIR keeps every ``prov.*`` tag; pools, relu, copies and sub-ops are tagged.

Merlin reads the text handoff, so a tag that exists on the in-memory module but
is dropped by an op's custom syntax (``linalg.reduce``, ``linalg.broadcast``,
``tensor.empty``, ``linalg.transpose``) is a tag it never sees.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401
from torch import nn

import m2m
from m2m.capture.torch_mlir_bridge import _parse_mlir_text_to_xdsl

A = torch.ops.aten


class _Call(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def _convert(fn, *inputs):
    result = m2m.convert(_Call(fn).eval(), inputs, backend="fx_importer")
    assert result.ok and "func.call" not in result.mlir_text
    return result


def _prov(op):
    return {key: str(value) for key, value in op.attributes.items() if key.startswith("prov.")}


def _ops_with(text, **attrs):
    rows = []
    for op in _parse_mlir_text_to_xdsl(text).walk():
        tags = _prov(op)
        if all(f'"{value}"' == tags.get(f"prov.{key}") for key, value in attrs.items()):
            rows.append((op.name, tags))
    return rows


CASES = {
    "mean": (lambda x: A.mean.dim(x, [1]), (torch.randn(2, 3, 4),)),
    "permute": (lambda x: A.permute.default(x, [1, 0]), (torch.randn(2, 3),)),
    "conv_bias": (lambda x, w, b: A.convolution.default(x, w, b, [1, 1], [1, 1], [1, 1], False, [0, 0], 1),
                  (torch.randn(2, 3, 5, 5), torch.randn(4, 3, 3, 3), torch.randn(4))),
    "addmm": (lambda c, a, b: A.addmm.default(c, a, b, beta=0.5, alpha=2.0),
              (torch.randn(5), torch.randn(3, 4), torch.randn(4, 5))),
    "avg_pool": (lambda x: A.avg_pool2d.default(x, [2, 2]), (torch.randn(1, 2, 4, 4),)),
    "expand": (lambda x: A.expand.default(x, [4, 3]), (torch.randn(1, 3),)),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_printed_text_round_trips_every_prov_attribute(name):
    fn, inputs = CASES[name]
    result = _convert(fn, *inputs)
    expected = [(op.name, _prov(op)) for op in result.module.walk() if _prov(op)]
    parsed = [(op.name, _prov(op)) for op in _parse_mlir_text_to_xdsl(result.mlir_text).walk() if _prov(op)]
    assert parsed == expected


def test_mean_reduce_is_tagged_in_text():
    result = _convert(*CASES["mean"][:1], *CASES["mean"][1])
    reduces = [tags for name, tags in _ops_with(result.mlir_text, aten="aten.mean.dim") if name == "linalg.reduce"]
    assert len(reduces) == 1
    assert reduces[0]["prov.family"] == '"reduce"' and reduces[0]["prov.role"] == '"sum"'
    roles = Counter(tags.get("prov.role") for _, tags in _ops_with(result.mlir_text, aten="aten.mean.dim"))
    assert roles['"scale"'] >= 1


@pytest.mark.parametrize("op, fn, inputs", [
    ("avg_pool2d", lambda x: A.avg_pool2d.default(x, [3, 3], [2, 2], [1, 1]), (torch.randn(1, 2, 7, 7),)),
    ("_adaptive_avg_pool2d", lambda x: A._adaptive_avg_pool2d.default(x, [3, 3]), (torch.randn(1, 2, 7, 7),)),
    ("avg_pool2d_backward", lambda g, x: A.avg_pool2d_backward.default(g, x, [2, 2], [2, 2], [0, 0], False, True,
                                                                       None),
     (torch.randn(1, 2, 2, 2), torch.randn(1, 2, 4, 4))),
    ("_adaptive_avg_pool2d_backward", lambda g, x: A._adaptive_avg_pool2d_backward.default(g, x),
     (torch.randn(1, 2, 3, 3), torch.randn(1, 2, 7, 7))),
])
def test_average_pools_are_reduce_family(op, fn, inputs):
    result = _convert(fn, *inputs)
    rows = _ops_with(result.mlir_text, aten=f"aten.{op}.default")
    assert rows and all(tags.get("prov.family") == '"reduce"' for _, tags in rows)


def test_relu_keeps_its_name_and_clone_is_movement():
    relu = _convert(lambda x: A.relu.default(x), torch.randn(3, 4))
    rows = [tags for name, tags in _ops_with(relu.mlir_text, aten="aten.relu.default") if name == "linalg.generic"]
    assert rows and rows[0]["prov.op"] == '"relu"' and rows[0]["prov.family"] == '"elementwise"'
    clone = _convert(lambda x: A.clone.default(x), torch.randn(3, 4))
    rows = [tags for name, tags in _ops_with(clone.mlir_text, aten="aten.clone.default") if name == "linalg.generic"]
    assert rows and rows[0]["prov.family"] == '"layout"'


def test_convolution_sub_ops_carry_roles():
    result = _convert(*CASES["conv_bias"][:1], *CASES["conv_bias"][1])
    roles = Counter((name, tags.get("prov.role"))
                    for name, tags in _ops_with(result.mlir_text, aten="aten.convolution.default"))
    assert roles[("linalg.matmul", '"contraction"')] == 1
    assert roles[("linalg.generic", '"im2col"')] == 1
    assert roles[("linalg.generic", '"bias_add"')] == 1
    assert any(role == '"pad"' for _, role in roles)
    assert any(role == '"acc_init"' for _, role in roles)
    assert all(role is not None for (name, role) in roles if name.startswith("linalg."))


def test_addmm_sub_ops_carry_roles():
    result = _convert(*CASES["addmm"][:1], *CASES["addmm"][1])
    roles = {tags.get("prov.role") for name, tags in _ops_with(result.mlir_text, aten="aten.addmm.default")
             if name.startswith("linalg.")}
    assert {'"contraction"', '"alpha_scale"', '"beta_scale"', '"bias_add"'} <= roles


def test_convolution_backward_outputs_carry_roles():
    def fn(go, x, w):
        return A.convolution_backward.default(go, x, w, [4], [1, 1], [1, 1], [1, 1], False, [0, 0], 1,
                                              [True, True, True])

    result = _convert(fn, torch.randn(1, 4, 5, 5), torch.randn(1, 3, 5, 5), torch.randn(4, 3, 3, 3))
    roles = {tags.get("prov.role") for _, tags in _ops_with(result.mlir_text,
                                                             aten="aten.convolution_backward.default")}
    assert {'"grad_input"', '"grad_weight"', '"grad_bias"'} <= roles


def test_integer_readout_seed_broadcast_prints_its_role():
    from m2m.capture.pt2e_compat import allow_exported_model_train_eval
    from m2m.capture.pt2e_integerize import integerize_pt2e

    Q = torch.ops.quantized_decomposed.quantize_per_tensor.default
    DQ = torch.ops.quantized_decomposed.dequantize_per_tensor.default

    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("w", torch.randint(-5, 6, (4, 8), dtype=torch.int8))
            self.register_buffer("b", torch.randn(4))

        def forward(self, x):
            dx = DQ(Q(x, 0.05, 0, -128, 127, torch.int8), 0.05, 0, -128, 127, torch.int8)
            return A.linear.default(dx, DQ(self.w, 0.02, 0, -127, 127, torch.int8), self.b)

    inputs = (torch.randn(3, 8),)
    model = torch.export.export(Linear().eval(), inputs).module()
    integer, _ = integerize_pt2e(model, inputs, numerics="integer_readout")
    allow_exported_model_train_eval(integer)
    text = m2m.convert(integer, inputs, backend="fx_importer").mlir_text
    seeds = [tags for name, tags in _ops_with(text, role="acc_seed") if name == "linalg.broadcast"]
    assert len(seeds) == 1
    roles = {tags.get("prov.role") for _, tags in _ops_with(text)}
    assert {'"weight_layout"', '"contraction"', '"readout"', '"acc_seed"'} <= roles
    assert re.search(r'"linalg.broadcast"', text)
