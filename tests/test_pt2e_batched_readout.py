"""Keep the batched integer MAC while flattening only its host readout."""

import pytest
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401

import m2m
from m2m.capture.pt2e_compat import allow_exported_model_train_eval
from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
from m2m.capture.pt2e_integerize import integerize_pt2e
from m2m.coverage import opaque_report


class QuantizedBmm(torch.nn.Module):
    def forward(self, a, b):
        def quant(value, scale):
            q = torch.ops.quantized_decomposed.quantize_per_tensor.default(value, scale, 0, -128, 127, torch.int8)
            return torch.ops.quantized_decomposed.dequantize_per_tensor.default(q, scale, 0, -128, 127, torch.int8)

        return torch.ops.aten.bmm.default(quant(a, 0.0137), quant(b, 0.0193))


@pytest.mark.parametrize("shape", [(1, 1, 1, 1), (2, 3, 4, 2), (3, 5, 17, 7)])
def test_batched_integerization_and_flat_readout_are_exact(shape):
    batch, m, k, n = shape
    generator = torch.Generator().manual_seed(107)
    inputs = (torch.randn(batch, m, k, generator=generator), torch.randn(batch, k, n, generator=generator))
    model = torch.export.export(QuantizedBmm(), inputs).module()
    reference = run_pt2e_integer_reference(model, inputs, expected_contractions=1)
    integer, receipt = integerize_pt2e(model, inputs)
    assert receipt["integer_bmm_emitted"] == receipt["integer_mm_emitted"] == 1
    assert receipt["matmul_integerized"] == 1 and not receipt["refusals"]
    assert torch.equal(integer(*inputs).view(torch.int32), reference.output.view(torch.int32))
    nodes = list(integer.graph.nodes)
    bmm = next(node for node in nodes if node.target == torch.ops.aten.bmm.default)
    flat = next(iter(bmm.users))
    assert flat.target == torch.ops.aten.reshape.default and flat.args[1] == [batch * m, n]
    cast = next(iter(flat.users))
    assert cast.target == torch.ops.aten._to_copy.default and cast.kwargs["dtype"] == torch.float32
    first = next(iter(cast.users))
    second = next(iter(first.users))
    assert first.target == second.target == torch.ops.aten.mul.Tensor
    restored = next(iter(second.users))
    assert restored.target == torch.ops.aten.reshape.default and restored.args[1] == [batch, m, n]

    allow_exported_model_train_eval(integer)
    result = m2m.convert(integer, inputs, backend="fx_importer", capture_trace=True)
    assert result.ok and result.capture_trace["status"] == "complete"
    assert opaque_report(result.mlir_text) == {}
    assert 'prov.aten = "aten.bmm.default"' in result.mlir_text
    assert 'prov.aten = "aten._int_mm.default"' not in result.mlir_text
    assert "arith.extsi" in result.mlir_text
    operations = result.capture_trace["mlir"]["operations"]
    assert all(operation["role"] != "unresolved" for operation in operations)
    assert all(row["status"] in {"lowered", "alias"} for row in result.capture_trace["mlir"]["source_correspondence"])


def test_plain_i32_bmm_is_not_narrowed():
    class Plain(torch.nn.Module):
        def forward(self, a, b):
            return torch.bmm(a, b)

    inputs = (torch.full((2, 3, 4), 1000, dtype=torch.int32), torch.full((2, 4, 2), -1000, dtype=torch.int32))
    result = m2m.convert(Plain(), inputs, backend="fx_importer", capture_trace=True)
    assert result.ok and opaque_report(result.mlir_text) == {}
    assert "arith.extsi" not in result.mlir_text
    assert "tensor<2x3x4xi32>" in result.mlir_text


def test_unsigned_origin_cannot_gain_signed_i8_mac():
    class Promoted(torch.nn.Module):
        def forward(self, a, b):
            return torch.bmm(a.int(), b.int())

    inputs = (torch.full((2, 3, 4), 200, dtype=torch.uint8), torch.full((2, 4, 2), 237, dtype=torch.uint8))
    result = m2m.convert(Promoted(), inputs, backend="fx_importer", capture_trace=True)
    assert result.ok and opaque_report(result.mlir_text) == {}
    contractions = [
        op
        for op in result.module.walk()
        if op.name == "linalg.generic" and any(iterator.data.value == "reduction" for iterator in op.iterator_types)
    ]
    assert len(contractions) == 1
    assert [str(value.type.element_type) for value in contractions[0].inputs] == ["i32", "i32"]
    assert contractions[0].attributes["prov.op"].data != "int_matmul"


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
def test_other_integer_sources_never_gain_i8_mac(dtype):
    class Promoted(torch.nn.Module):
        def forward(self, a, b):
            return torch.bmm(a.int(), b.int())

    inputs = (torch.full((2, 3, 4), 1000, dtype=dtype), torch.full((2, 4, 2), -1000, dtype=dtype))
    result = m2m.convert(Promoted(), inputs, backend="fx_importer", capture_trace=True)
    assert result.ok and opaque_report(result.mlir_text) == {}
    assert "tensor<2x3x4xi8>" not in result.mlir_text


@pytest.mark.parametrize("layout", ["dense", "offset", "padded", "expanded"])
def test_batched_flat_readout_compiles_and_restores_source_boundary(tmp_path, layout):
    from native_execution import assert_native_matches_eager

    class Readout(torch.nn.Module):
        def forward(self, a, b):
            accumulator = torch.bmm(a.int(), b.int())
            return ((accumulator.reshape(15, 4).float() * 0.0137) * 0.0193).reshape(3, 5, 4)

    a = torch.arange(105, dtype=torch.int8).reshape(3, 5, 7)
    b = torch.arange(84, dtype=torch.int8).reshape(3, 7, 4)
    if layout == "offset":
        a = torch.cat((torch.zeros(1, dtype=torch.int8), a.flatten()))[1:].reshape(a.shape)
    elif layout == "padded":
        storage = torch.zeros(3, 5, 14, dtype=torch.int8)
        storage[..., ::2].copy_(a)
        a = storage[..., ::2]
    elif layout == "expanded":
        a = a[:1].clone().expand(a.shape)
    assert_native_matches_eager(tmp_path, Readout(), (a, b), rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["nonidentity", "dynamic", "extra_body"])
def test_extension_fold_refuses_unproved_sources(invalid):
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, TensorType
    from xdsl.ir.affine import AffineExpr, AffineMap

    from m2m.ir.decompositions import _signed_i8_extension_source

    class Widen(torch.nn.Module):
        def forward(self, a):
            return a.int()

    result = m2m.convert(Widen(), (torch.ones(2, 3, 4, dtype=torch.int8),), backend="fx_importer")
    extension = next(op for op in result.module.walk() if op.name == "linalg.generic")
    assert _signed_i8_extension_source(extension.results[0]) is not None
    if invalid == "nonidentity":
        d = AffineExpr.dimension
        maps = list(extension.get_indexing_maps())
        maps[0] = AffineMapAttr(AffineMap(3, 0, (d(0), d(2), d(1))))
        extension.properties["indexing_maps"] = ArrayAttr(maps)
    elif invalid == "dynamic":
        source = extension.inputs[0]
        source._type = TensorType(source.type.element_type, [-1, 3, 4])
    else:
        block = extension.body.block
        block.insert_op_before(ConstantOp.from_int_and_width(0, 32), block.last_op)
    assert _signed_i8_extension_source(extension.results[0]) is None
