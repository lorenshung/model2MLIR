import copy, json
import m2m, pytest, torch


class Model(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *inputs):
        return self.fn(*inputs)


@pytest.mark.parametrize("operation", ["copy", "select_scatter", "slice_scatter", "diagonal_scatter"])
@pytest.mark.parametrize("layout", ["offset", "padded", "transposed", "expanded", "empty"])
@pytest.mark.parametrize("trace", [False, True])
def test_preserving_fresh_layout(operation, layout, trace):
    base = torch.arange(50, dtype=torch.float32)
    x = {
        "offset": lambda: base.as_strided((2, 3), (3, 1), 3),
        "padded": lambda: base.as_strided((2, 3), (8, 2), 3),
        "transposed": lambda: base.as_strided((2, 3), (1, 2), 3),
        "expanded": lambda: base.as_strided((2, 3), (0, 1), 3),
        "empty": lambda: base.as_strided((0, 3), (3, 1), 3),
    }[layout]()
    fn = {
        "copy": lambda x: torch.ops.aten.copy.default(x, torch.ones_like(x)),
        "select_scatter": lambda x: torch.ops.aten.select_scatter.default(x, torch.ones(x.shape[0]), 1, 1),
        "slice_scatter": lambda x: torch.ops.aten.slice_scatter.default(x, torch.ones((x.shape[0], 1)), 1, 1, 2),
        "diagonal_scatter": lambda x: torch.ops.aten.diagonal_scatter.default(x, torch.ones(min(x.shape)), 0, 0, 1),
    }[operation]

    def views(x):
        y = fn(x)
        return y, y.transpose(0, 1)

    expected = views(x)
    r = m2m.convert(Model(views), (x,), backend="fx_importer", capture_trace=trace)
    assert r.ok, r.diagnostics
    records = json.loads(r.module.attributes["m2m.result_layout"].data)["results"]
    for a, b in zip(records, expected):
        assert (a["shape"], a["stride"], a["storage_offset"]) == (list(b.shape), list(b.stride()), b.storage_offset())
        assert a["alias_inputs"] == []
    assert records[1]["alias_results"] == [0]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("padding", [-1, 1])
@pytest.mark.parametrize("stride", [1, 2])
def test_embedding_bag_fast_shape(dtype, padding, stride):
    w = torch.arange(30, dtype=dtype).reshape(5, 6)[:, ::stride]
    i = torch.tensor([0, 1, 2, 1])
    o = torch.tensor([0, 2])
    fn = lambda w, i, o: torch.ops.aten._embedding_bag.default(w, i, o, False, 0, False, None, False, padding)
    expected = fn(w, i, o)
    r = m2m.convert(Model(fn), (w, i, o), backend="fx_importer", capture_trace=True)
    assert r.ok, r.diagnostics
    records = json.loads(r.module.attributes["m2m.result_layout"].data)["results"]
    assert [a["shape"] for a in records] == [list(v.shape) for v in expected]


@pytest.mark.parametrize("offset", [0, 3])
def test_resizing_out_preserves_offset(offset):
    x = torch.ones(2, 1, 5)
    y = torch.ones(1, 3, 1)
    out = torch.empty(20).as_strided(x.shape, x.stride(), offset)
    fn = lambda x, y, out: torch.ops.aten.atan2.out(x, y, out=out)
    r = m2m.convert(Model(fn), (x, y, copy.deepcopy(out)), backend="fx_importer", capture_trace=True)
    assert r.ok, r.diagnostics
    expected = fn(x, y, out)
    records = json.loads(r.module.attributes["m2m.result_layout"].data)["results"]
    assert len(records) == 2
    assert all(a["storage_offset"] == expected.storage_offset() == offset for a in records)
    assert all(a["shape"] == list(expected.shape) for a in records)


@pytest.mark.parametrize("operation", ["select_scatter", "slice_scatter", "diagonal_scatter"])
@pytest.mark.parametrize("shape, strides", [((2, 3), (1, 0)), ((2, 3, 4), (0, 4, 1)), ((3, 3), (0, 0))])
def test_overlapping_source_scatter_records_dense_fresh_result(operation, shape, strides):
    """Expanded inputs: the CPU result is empty_like(source) (dense, offset 0)."""
    x = torch.arange(50, dtype=torch.float32).as_strided(shape, strides, 3)
    tail = x.shape[2:]
    fn = {
        "select_scatter": lambda x: torch.ops.aten.select_scatter.default(x, torch.ones(x.shape[0], *tail), 1, 1),
        "slice_scatter": lambda x: torch.ops.aten.slice_scatter.default(x, torch.ones(x.shape[0], 1, *tail), 1, 1, 2),
        "diagonal_scatter": lambda x: torch.ops.aten.diagonal_scatter.default(
            x, torch.ones(*tail, min(x.shape[:2])), 0, 0, 1),
    }[operation]
    expected = fn(x)
    r = m2m.convert(Model(fn), (x,), backend="fx_importer")
    assert r.ok, r.diagnostics
    record = json.loads(r.module.attributes["m2m.result_layout"].data)["results"][0]
    assert (record["shape"], record["stride"], record["storage_offset"]) == (
        list(expected.shape), list(expected.stride()), expected.storage_offset())
