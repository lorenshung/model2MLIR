"""A capture receipt binds the actual bundle bytes and observed source."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import sympy
import torch
from m2m.capture.bundle import (
    _lifted_constants,
    _numpy_safe,
    _opaque_call_count,
    _tensor_abi,
    write_bundle,
)
from m2m.capture.provenance import _package_python_sources, capture_receipt
from m2m.capture.torch_export import _serialize_range_constraints


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)


class _LiteralRange(torch.nn.Module):
    def __init__(self, start, end):
        super().__init__()
        self.start = start
        self.end = end

    def forward(self, x):
        return torch.arange(self.start, self.end, dtype=torch.int64, device=x.device)


def test_bundle_preserves_exact_integer_golden_and_receipt(tmp_path):
    ranges = (
        (2**24 + 1, 2**24 + 6),
        (-(2**63) + 9, -(2**63) + 14),
        (2**63 - 14, 2**63 - 9),
    )
    for index, (start, end) in enumerate(ranges):
        out = tmp_path / f"range-{index}"
        write_bundle(_LiteralRange(start, end).eval(), (torch.ones(1),), out,
                     capture_regions=False, capture_trace=True)
        expected = torch.arange(start, end, dtype=torch.int64).numpy()
        actual = np.load(out / "golden.npy", allow_pickle=False)
        assert actual.dtype == np.dtype("int64")
        np.testing.assert_array_equal(actual.view("uint64"), expected.view("uint64"))
        assert capture_receipt(out)["artifacts"]["golden.npy"]["sha256"] == hashlib.sha256(
            (out / "golden.npy").read_bytes()
        ).hexdigest()


def test_bundle_golden_numpy_storage_policy_leaves_representable_dtypes_exact():
    f32 = torch.tensor([-0.0, 1.25], dtype=torch.float32)
    i64 = torch.tensor([2**63 - 1, -(2**63)], dtype=torch.int64)
    np.testing.assert_array_equal(_numpy_safe(f32).view("uint32"), f32.numpy().view("uint32"))
    np.testing.assert_array_equal(_numpy_safe(i64).view("uint64"), i64.numpy().view("uint64"))
    assert _numpy_safe(f32).dtype == np.dtype("float32")
    assert _numpy_safe(i64).dtype == np.dtype("int64")
    bf16 = torch.tensor([1.25, -0.0], dtype=torch.bfloat16)
    assert _numpy_safe(bf16).dtype == np.dtype("float32")
    np.testing.assert_array_equal(_numpy_safe(bf16), bf16.float().numpy())


def test_bundle_receipt_binds_source_tool_framework_and_materialized_abi(tmp_path):
    source = tmp_path / "example.pt2"
    source.write_bytes(b"observed source archive")
    bundle = tmp_path / "bundle"
    write_bundle(_Tiny().eval(), (torch.randn(2, 4),), bundle,
                 capture_regions=False, source_path=source)

    receipt = json.loads((bundle / "capture_receipt.json").read_text())
    assert receipt["schema"] == "m2m.capture-receipt.v1"
    assert receipt["source_closure_verified"] is False
    assert receipt["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt["tool"]["commit"]
    assert receipt["tool"]["source_sha256"]["m2m/ir/decompositions.py"]
    assert receipt["tool"]["source_inventory_status"] == "complete"
    assert receipt["tool"]["source_inventory_scope"] == "m2m_package_python_sources_only"
    assert not receipt["tool"]["unavailable_sources"]
    assert all(name.startswith("m2m/") for name in receipt["tool"]["source_sha256"])
    assert "m2m/capture/stagewise_pt2e.py" in receipt["tool"]["source_sha256"]
    assert "m2m/capture/external_runtime.py" in receipt["tool"]["source_sha256"]
    assert receipt["framework"]["torch"] == torch.__version__
    assert receipt["lifted_constants"] == {}
    assert receipt["materialized_abi"] == {"complete": True, "inputs": 1, "lifted_constants": []}
    for name in ("model.mlir", "weights.safetensors", "inputs.npz", "golden.npy",
                 "extra.npz", "input_order.json"):
        record = receipt["artifacts"][name]
        data = (bundle / name).read_bytes()
        assert record == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    # The original PT2 can lift a scalar. NumPy's ascontiguousarray promotes it
    # to [1], which would silently put the wrong ABI shape in the receipt.
    manifest_path = bundle / "weights.safetensors.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[str(len(manifest))] = {"kind": "constant", "name": "c_lifted_tensor_0"}
    manifest_path.write_text(json.dumps(manifest))
    np.savez(bundle / "extra.npz", c_lifted_tensor_0=np.array(1.0, dtype=np.float32))
    assert capture_receipt(bundle)["lifted_constants"]["c_lifted_tensor_0"]["shape"] == []


def test_python_source_inventory_finds_new_nested_modules_and_flags_links(tmp_path):
    package = tmp_path / "m2m"
    nested = package / "capture"
    nested.mkdir(parents=True)
    (package / "__init__.py").write_text("# package\n")
    (nested / "future_stage.py").write_text("VALUE = 1\n")
    cache = nested / "__pycache__"
    cache.mkdir()
    (cache / "generated.py").write_text("VALUE = 0\n")

    hashes, unavailable = _package_python_sources(tmp_path)
    assert set(hashes) == {"m2m/__init__.py", "m2m/capture/future_stage.py"}
    assert hashes["m2m/capture/future_stage.py"] == hashlib.sha256(b"VALUE = 1\n").hexdigest()
    assert unavailable == []

    (nested / "linked.py").symlink_to(nested / "future_stage.py")
    hashes, unavailable = _package_python_sources(tmp_path)
    assert "m2m/capture/linked.py" not in hashes
    assert unavailable == ["m2m/capture/linked.py"]


def test_integer_reference_is_bound_to_capture_receipt(tmp_path):
    import pytest

    bundle = tmp_path / "bundle"
    write_bundle(_Tiny().eval(), (torch.randn(2, 4),), bundle, capture_regions=False)
    reference = bundle / "integer-reference.json"
    reference.write_text('{"outputs": [1]}\n')
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    meta = {"integerization_receipt": {"golden_agreement": {
        "reference": "pt2e_integer", "output": {"path": reference.name, "sha256": digest},
    }}}
    (bundle / "meta.json").write_text(json.dumps(meta))
    receipt = capture_receipt(bundle)
    assert receipt["artifacts"][reference.name]["sha256"] == digest

    reference.write_text('{"outputs": [2]}\n')
    with pytest.raises(ValueError, match="differs"):
        capture_receipt(bundle)
    meta["integerization_receipt"]["golden_agreement"]["output"]["path"] = "../integer-reference.json"
    (bundle / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="bundle-local"):
        capture_receipt(bundle)


def test_unbounded_export_constraint_does_not_break_lifted_constant_capture():
    exported = SimpleNamespace(range_constraints={
        sympy.Symbol("s0"): SimpleNamespace(lower=1, upper=sympy.oo),
    })
    assert _serialize_range_constraints(exported)[0].minimum == 1
    assert _serialize_range_constraints(exported)[0].maximum is None


def test_original_export_supplies_lifted_constants_without_second_export():
    from torch.export.graph_signature import ConstantArgument, InputKind, InputSpec

    exported = SimpleNamespace(
        constants={"lifted_tensor_0": torch.tensor([True, False])},
        state_dict={},
        graph_signature=SimpleNamespace(input_specs=[
            InputSpec(InputKind.CONSTANT_TENSOR, ConstantArgument("c_lifted_tensor_0", None),
                      "lifted_tensor_0"),
        ]),
    )
    extra = {}
    _lifted_constants(None, (), extra, exported_program=exported)
    assert extra["c_lifted_tensor_0"].tolist() == [True, False]


def test_traced_bundle_and_reused_conversion_bind_exact_bytes_without_recapture(tmp_path, monkeypatch):
    import m2m
    import m2m.capture.torch_export as frontend

    original_capture = frontend.capture_model
    calls = []

    def capture(*args, **kwargs):
        calls.append(True)
        return original_capture(*args, **kwargs)

    monkeypatch.setattr(frontend, "capture_model", capture)
    model, inputs = _Tiny().eval(), (torch.ones(2, 4),)
    out = tmp_path / "bundle"
    out.mkdir()
    (out / "meta.json").write_text(json.dumps({"loader_provenance": {"source": "authored"}, "opaque": 7}))
    write_bundle(model, inputs, out, capture_trace=True, capture_regions=False,
                 metadata={"workload_role": "iteration"})
    assert len(calls) == 1  # no separate lifted-constant capture
    trace = json.loads((out / "frontend-trace.json").read_text())
    meta = json.loads((out / "meta.json").read_text())
    receipt = json.loads((out / "capture_receipt.json").read_text())
    assert trace["status"] == "complete" and meta["workload_role"] == "iteration"
    assert meta["opaque"] == 0
    assert meta["weights"] == str(out / "weights.safetensors")
    assert meta["input_abi"] == [{"shape": [2, 4], "dtype": "f32"}]
    assert meta["output_abi"] == [{"shape": [2, 3], "dtype": "f32"}]
    assert meta["loader_provenance"] == {"source": "authored"}
    for name in ("model.mlir", "frontend-trace.json", "meta.json"):
        assert receipt["artifacts"][name]["sha256"] == hashlib.sha256((out / name).read_bytes()).hexdigest()

    reused = tmp_path / "reused"
    reused.mkdir()
    result = m2m.convert(model, inputs, backend="fx_importer", capture_trace=True,
                         weights_path=str(reused / "weights.safetensors"))
    expected_weights = (reused / "weights.safetensors").read_bytes()
    before = len(calls)
    write_bundle(model, inputs, reused, capture_trace=True, capture_regions=False,
                 conversion_result=result)
    assert len(calls) == before
    assert (reused / "weights.safetensors").read_bytes() == expected_weights
    assert (reused / "model.mlir").read_text() == result.mlir_text

    # Byte relocation/normalization requires a new receipt, never blind rebind.
    (reused / "model.mlir").write_text(result.mlir_text + "\n")
    import pytest
    with pytest.raises(ValueError, match="exact model.mlir"):
        capture_receipt(reused)


def test_opaque_count_is_structural_and_unknown_without_module():
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.dialects.func import CallOp

    assert _opaque_call_count(ModuleOp([CallOp("unlowered", [], [])])) == 1
    assert _opaque_call_count(None) is None
    assert _tensor_abi(torch.empty(2, 3, dtype=torch.bfloat16)) == {"shape": [2, 3], "dtype": "bf16"}


def test_static_pt2e_bundle_golden_is_actual_quantized_model(tmp_path, monkeypatch):
    import m2m.capture.torchao_pipeline as pipeline
    from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization

    torch.manual_seed(0)
    model, inputs = _Tiny().eval(), (torch.randn(2, 4),)
    expected = apply_quantization(model, QuantizationConfig(scheme="int8_static_act_int8_weight"),
                                  example_inputs=inputs)
    expected_golden = expected(*inputs).detach().numpy()
    # Use the actual returned PT2E GraphModule at the seam where the writer used
    # to ignore it; no second quantization instance is necessary.
    monkeypatch.setattr(pipeline, "apply_quantization", lambda *args, **kwargs: expected)
    out = tmp_path / "pt2e"
    write_bundle(model, inputs, out, capture_regions=False, capture_trace=True,
                 quant=QuantizationConfig(scheme="int8_static_act_int8_weight"))
    np.testing.assert_array_equal(np.load(out / "golden.npy"), expected_golden)


def test_bundle_extra_preserves_registered_buffer_dtypes_and_values(tmp_path):
    class Buffered(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("large_ids", torch.tensor([2**40 + 1], dtype=torch.int64))
            self.register_buffer("quantized", torch.tensor([-128, 127], dtype=torch.int8))
            self.register_buffer("mask", torch.tensor([True, False]))
            self.register_buffer("wide", torch.tensor([1.0 + 2**-40], dtype=torch.float64))
            self.register_buffer("bf16", torch.tensor([1.25], dtype=torch.bfloat16))

        def forward(self, x):
            return x + self.quantized.float().sum()

    model = Buffered().eval()
    write_bundle(model, (torch.ones(2),), tmp_path, capture_regions=False)
    with np.load(tmp_path / "extra.npz") as extra:
        for name, tensor in model.named_buffers():
            expected = tensor.float().numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
            actual = extra["buf::" + name]
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)
def test_bundle_preserves_all_output_dtypes_and_aliases(tmp_path):
    class Outputs(torch.nn.Module):
        def forward(self,x):
            return (x, torch.tensor([2**60+3],dtype=torch.int64), x>0, None)
    out=tmp_path/'bundle'
    write_bundle(Outputs(),(torch.tensor([-1.,2.]),),out,capture_regions=False)
    abi=json.loads((out/'result_abi.json').read_text())
    assert abi['results'][0]['alias_input_index']==0
    assert abi['results'][1]['dtype']=='int64'
    assert abi['results'][2]['dtype']=='bool'
    assert abi['results'][3]['literal'] is None
    with np.load(out/'goldens.npz') as arrays:
        assert arrays['out1'].dtype==np.int64
        assert arrays['out1'][0]==2**60+3
        assert arrays['out2'].dtype==np.bool_
    receipt=capture_receipt(out)
    for name in ('result_abi.json','goldens.npz'):
        assert receipt['artifacts'][name]['sha256']==hashlib.sha256((out/name).read_bytes()).hexdigest()


def test_bundle_routes_input_mutation_and_checks_result_abi(tmp_path):
    import pytest
    class Out(torch.nn.Module):
        def forward(self,x,y,out): return torch.ops.aten.atan2.out(x,y,out=out)
    destination=tmp_path/'bundle';z=torch.zeros(2)
    write_bundle(Out(),(torch.ones(2),torch.ones(2),z),destination,capture_regions=False)
    abi=json.loads((destination/'result_abi.json').read_text())
    assert abi['results'][0]['role']=='user_input_mutation'
    assert abi['results'][1]['alias_input_index']==2
    with np.load(destination/'inputs.npz') as arrays: assert (arrays['in2']==0).all()
    with np.load(destination/'goldens.npz') as arrays:
        np.testing.assert_equal(arrays['mutation0'],arrays['out0'])
    abi['results'][0]['shape']=[1]
    (destination/'result_abi.json').write_text(json.dumps(abi))
    with pytest.raises(ValueError,match='result golden shape'):
        capture_receipt(destination)


def test_result_abi_serializes_dtype_and_symbolic_literal_leaves(tmp_path):
    from m2m.capture.bundle import _write_result_abi
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    symbol = ShapeEnv().create_unbacked_symint()
    outputs = (torch.ones(2), torch.float32, symbol, None, True, 7)
    _write_result_abi(tmp_path, outputs, (), torch.nn.Module(), None)
    records = json.loads((tmp_path / 'result_abi.json').read_text())['results']
    assert records[1]['literal'] == {'kind': 'dtype', 'value': 'torch.float32'}
    assert records[2]['literal'] == {'kind': 'SymInt', 'value': str(symbol)}
    assert [r['literal'] for r in records[3:]] == [None, True, 7]


def test_bundle_preserves_dtype_output_leaf(tmp_path):
    import m2m
    class TensorOutput(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class DtypeOutput(torch.nn.Module):
        def forward(self, x):
            return x + 1, x.dtype

    # This torch version cannot export dtype literals. Reuse the tensor graph
    # through the public prepared-conversion path; eager retains the literal.
    inputs = (torch.ones(2),)
    conversion = m2m.convert(TensorOutput(), inputs, backend='fx_importer',
                             weights_path=str(tmp_path / 'weights.safetensors'))
    write_bundle(DtypeOutput(), inputs, tmp_path, capture_regions=False,
                 conversion_result=conversion)
    abi = json.loads((tmp_path / 'result_abi.json').read_text())
    assert abi['results'][1]['literal'] == {'kind': 'dtype', 'value': 'torch.float32'}
    assert capture_receipt(tmp_path)['artifacts']['result_abi.json']


def test_declared_result_artifacts_cannot_be_deleted(tmp_path):
    import pytest

    write_bundle(_Tiny().eval(), (torch.ones(2, 4),), tmp_path, capture_regions=False)
    meta = json.loads((tmp_path / 'meta.json').read_text())
    assert meta['result_abi'] == {'path': 'result_abi.json', 'goldens': 'goldens.npz'}
    saved = {name: (tmp_path / name).read_bytes() for name in ('result_abi.json', 'goldens.npz')}
    for deleted in (('result_abi.json',), ('goldens.npz',), tuple(saved)):
        for name in deleted:
            (tmp_path / name).unlink()
        with pytest.raises(ValueError, match='required result artifact'):
            capture_receipt(tmp_path)
        for name, data in saved.items():
            (tmp_path / name).write_bytes(data)
