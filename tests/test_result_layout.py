"""Frontend storage semantics survive tensor-value lowering without eager answers."""

import json

import m2m
import pytest
import torch
from m2m.ir.result_layout import ATTRIBUTE


class Model(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: torch.ops.aten.alias.default(x),
        lambda x: x.permute(2, 0, 1),
        lambda x: x.unsqueeze(0).squeeze(0),
        lambda x: x.unsqueeze(0).unsqueeze(-1).squeeze((0, 4)),
        lambda x: x.unsqueeze(1),
        lambda x: x.view(6, 4),
        lambda x: x[:, 1, 1::2],
        lambda x: x.diagonal(1, 0, 2),
        lambda x: x.as_strided((2, 3), (4, 1), 3),
        lambda x: x.reshape(6, 4).split([2, 4]),
        lambda x: x[:, :1, :].expand(2, 5, 4),
        lambda x: x.sin().transpose(0, 2),
        lambda x: (x.sin(), x.cos()),
    ],
)
def test_result_layout_comes_from_same_graph(fn):
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    result = m2m.convert(Model(fn), (x,), backend="fx_importer")
    assert result.ok, result.diagnostics
    result.module.verify()
    contract = json.loads(result.module.attributes[ATTRIBUTE].data)
    expected = fn(x)
    expected = list(expected) if isinstance(expected, (tuple, list)) else [expected]
    assert len(contract["results"]) == len(expected)
    for record, value in zip(contract["results"], expected):
        assert record["shape"] == list(value.shape)
        assert record["stride"] == list(value.stride())
        assert record["storage_offset"] == value.storage_offset()
        alias = value.untyped_storage()._cdata == x.untyped_storage()._cdata
        assert record["alias_inputs"] == ([0] if alias else [])


def test_dynamic_layout_uses_runtime_result_dimensions():
    result = m2m.convert(
        Model(lambda x: torch.nonzero(x)),
        (torch.tensor([[True, False, True], [False, True, True]]),),
        backend="fx_importer",
    )
    assert result.ok, result.diagnostics
    contract = json.loads(result.module.attributes[ATTRIBUTE].data)
    output = contract["results"][0]
    assert output["shape"] == [-1, 2]
    assert output["stride"] == [1, {"shape_axes": [0]}]
    assert output["alias_inputs"] == []


def test_fresh_intermediate_result_alias_is_retained():
    def fn(x):
        y = x.sin()
        return y, y.transpose(0, 1)

    result = m2m.convert(Model(fn), (torch.rand(3, 5),), backend="fx_importer")
    assert result.ok
    records = json.loads(result.module.attributes[ATTRIBUTE].data)["results"]
    assert records[1]["alias_inputs"] == []
    assert records[1]["alias_results"] == [0]


def test_metadata_mutation_retains_input_and_result_storage_roles():
    result = m2m.convert(
        Model(lambda x: torch.ops.aten.resize_.default(x, [3, 2])),
        (torch.arange(6, dtype=torch.float32).reshape(2, 3),),
        backend="fx_importer",
    )
    assert result.ok, result.diagnostics
    records = json.loads(result.module.attributes[ATTRIBUTE].data)["results"]
    assert [record["role"] for record in records] == ["user_input_mutation", "user_output"]
    assert records[0]["input_index"] == 0
    assert all(record["alias_inputs"] == [0] for record in records)
    assert all(record["shape"] == [3, 2] and record["stride"] == [2, 1] for record in records)


def test_complex_layout_uses_the_importer_component_units():
    result = m2m.convert(
        Model(lambda x: torch.view_as_real(x)), (torch.ones(3, 5, dtype=torch.complex64),), backend="fx_importer"
    )
    assert result.ok, result.diagnostics
    contract = json.loads(result.module.attributes[ATTRIBUTE].data)
    assert contract["inputs"][0]["shape"] == [3, 5, 2]
    assert contract["inputs"][0]["stride"] == [10, 2, 1]
    assert contract["results"][0]["stride"] == [10, 2, 1]
    assert contract["results"][0]["alias_inputs"] == [0]
