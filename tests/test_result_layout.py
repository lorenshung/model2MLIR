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


def _contract(fn, x):
    """Convert ``fn(x)`` and return its verified module's result-layout contract."""
    result = m2m.convert(Model(fn), (x,), backend="fx_importer")
    assert result.ok, result.diagnostics
    result.module.verify()
    return json.loads(result.module.attributes[ATTRIBUTE].data)


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
@pytest.mark.parametrize('layout', ['contiguous', 'padded', 'transposed', 'expanded', 'empty'])
@pytest.mark.parametrize('train', [False, True, None])
def test_native_dropout_allocation_contract(dtype, layout, train):
    base = torch.arange(24.).to(dtype)
    x = {'contiguous': lambda: base[:6].reshape(2, 3),
         'padded': lambda: base.as_strided((2, 3), (6, 2)),
         'transposed': lambda: base[:6].reshape(3, 2).t(),
         'expanded': lambda: base[:3].reshape(1, 3).expand(2, 3),
         'empty': lambda: base[:0].reshape(2, 0, 3)}[layout]()
    def fn(a):
        output, mask = torch.ops.aten.native_dropout.default(a, .25, train)
        return output, mask, torch.ops.aten.transpose.int(output, 0, 1)
    expected = fn(x)
    records = _contract(fn, x)['results']
    for index, (record, eager) in enumerate(zip(records, expected, strict=True)):
        physical = torch.view_as_real(eager) if dtype.is_complex and eager.dtype.is_complex else eager
        assert record['shape'] == list(physical.shape)
        assert record['stride'] == list(physical.stride())
        assert record['storage_offset'] == physical.storage_offset()
        aliases = eager.untyped_storage()._cdata == x.untyped_storage()._cdata
        assert record['alias_inputs'] == ([0] if aliases else [])
        assert record['alias_results'] == ([0] if index == 2 else [])


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
    contract = _contract(fn, x)
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
    contract = _contract(torch.nonzero, torch.tensor([[True, False, True], [False, True, True]]))
    output = contract["results"][0]
    assert output["shape"] == [-1, 2]
    assert output["stride"] == [1, {"shape_axes": [0]}]
    assert output["alias_inputs"] == []


def test_fresh_intermediate_result_alias_is_retained():
    def fn(x):
        y = x.sin()
        return y, y.transpose(0, 1)

    records = _contract(fn, torch.rand(3, 5))["results"]
    assert records[1]["alias_inputs"] == []
    assert records[1]["alias_results"] == [0]


def test_metadata_mutation_retains_input_and_result_storage_roles():
    records = _contract(lambda x: torch.ops.aten.resize_.default(x, [3, 2]),
                        torch.arange(6, dtype=torch.float32).reshape(2, 3))["results"]
    assert [record["role"] for record in records] == ["user_input_mutation", "user_output"]
    assert records[0]["input_index"] == 0
    assert all(record["alias_inputs"] == [0] for record in records)
    assert all(record["shape"] == [3, 2] and record["stride"] == [2, 1] for record in records)


def test_complex_layout_uses_the_importer_component_units():
    contract = _contract(torch.view_as_real, torch.ones(3, 5, dtype=torch.complex64))
    assert contract["inputs"][0]["shape"] == [3, 5, 2]
    assert contract["inputs"][0]["stride"] == [10, 2, 1]
    assert contract["results"][0]["stride"] == [10, 2, 1]
    assert contract["results"][0]["alias_inputs"] == [0]


@pytest.mark.parametrize("layout", ["padded", "offset", "zero_stride", "channels_last"])
@pytest.mark.parametrize("operation", ["elementwise", "reduction", "view", "mutation"])
def test_strided_inputs_preserve_storage_provenance(layout, operation):
    values = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    if layout == "padded":
        x = torch.empty(1, 2, 3, 8)[..., ::2]
        x.copy_(values)
    elif layout == "offset":
        x = torch.empty(40).as_strided(values.shape, (24, 12, 4, 1), 7)
        x.copy_(values)
    elif layout == "zero_stride":
        x = values[:, :1].expand(1, 2, 3, 4)
    else:
        x = values.contiguous(memory_format=torch.channels_last)

    def fn(x):
        if operation == "elementwise":
            return x.sin()
        if operation == "reduction":
            return x.sum(dim=2)
        if operation == "view":
            return x.transpose(1, 3)
        return x.fill_(3.25)

    contract = _contract(fn, x)
    input_record = contract["inputs"][0]
    assert input_record["shape"] == list(x.shape)
    assert input_record["stride"] == list(x.stride())
    assert input_record["storage_offset"] == x.storage_offset()
    expected = fn(x)
    for record in contract["results"]:
        assert record["shape"] == list(expected.shape)
        assert record["stride"] == list(expected.stride())
        assert record["storage_offset"] == expected.storage_offset()
        assert record["alias_inputs"] == ([0] if operation in ("view", "mutation") else [])
    if operation == "mutation":
        assert [r["role"] for r in contract["results"]] == ["user_input_mutation", "user_output"]
        assert contract["results"][0]["input_index"] == 0


def test_views_after_strided_mutation_use_caller_storage():
    x = torch.empty(4, 10)[:, ::2]

    def fn(x):
        x.fill_(2)
        return x.transpose(0, 1), x[:, 1:4]

    contract = _contract(fn, x)
    assert [r["stride"] for r in contract["results"]] == [[10, 2], [2, 10], [10, 2]]
    assert [r["storage_offset"] for r in contract["results"]] == [0, 0, 2]
    assert all(r["alias_inputs"] == [0] for r in contract["results"])
