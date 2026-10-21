"""Compile a captured single-op program with upstream MLIR and run it natively.

Shared by regression tests that must check values, not only that a lowering is
free of opaque calls. Local tests skip when tools are absent and unconfigured;
point ``MODEL2MLIR_MLIR_TRANSLATE`` (with ``mlir-opt`` beside it) and
``MODEL2MLIR_CLANG`` at an LLVM build to enable them. Explicit but broken
configuration fails; see docs/NATIVE_TESTS.md for CI.
"""

from __future__ import annotations

import ctypes
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import m2m
import numpy as np
import pytest
import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401  (registers quantized_decomposed)

PIPELINE = (
    "builtin.module(one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=fully-dynamic-layout-map},"
    "buffer-results-to-out-params{modify-public-functions},convert-linalg-to-loops,"
    "expand-strided-metadata,lower-affine,convert-scf-to-cf,arith-expand,math-expand-ops{ops=rsqrt},"
    "convert-math-to-libm,convert-math-to-llvm,convert-index-to-llvm,"
    "convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,"
    "convert-cf-to-llvm,reconcile-unrealized-casts)"
)


Q = torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ = torch.ops.quantized_decomposed.dequantize_per_tensor.default


def qdq(value, scale, low=-128):
    """PT2E's symmetric int8 per-tensor quantize/dequantize pair."""
    return DQ(Q(value, scale, 0, low, 127, torch.int8), scale, 0, low, 127, torch.int8)


def seeded_randn(*shape, dtype=torch.float32, seed=0):
    """A standard normal sample seeded by ``seed`` and the shape."""
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed + sum(shape))).to(dtype)


def padded(t):
    """``t``'s values every other element of a last dimension twice as wide (uninitialized holes)."""
    storage = torch.empty((*t.shape[:-1], t.shape[-1] * 2), dtype=t.dtype)
    storage[..., ::2] = t
    return storage[..., ::2]


class _Call(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def _configured() -> bool:
    return bool(os.environ.get("MODEL2MLIR_MLIR_TRANSLATE") or os.environ.get("MODEL2MLIR_CLANG"))


def _tools():
    """Return ``(mlir-opt, mlir-translate, clang)`` paths for native tests.

    ``MODEL2MLIR_*`` is the suite-wide configuration; the legacy
    ``MERLIN_MLIR_TRANSLATE``/``MERLIN_CLANG`` spellings are still honoured.
    Configured but broken tools fail; unconfigured and absent tools skip.
    """
    translate = (os.environ.get("MODEL2MLIR_MLIR_TRANSLATE") or os.environ.get("MERLIN_MLIR_TRANSLATE")
                 or shutil.which("mlir-translate"))
    clang = os.environ.get("MODEL2MLIR_CLANG") or os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    if not translate or not clang:
        if _configured():
            pytest.fail("configured native toolchain requires both MLIR and clang")
        pytest.skip("native execution requires MLIR/LLVM tools")
    opt = Path(translate).with_name("mlir-opt")
    if not opt.is_file():
        if _configured():
            pytest.fail("configured native toolchain lacks mlir-opt beside mlir-translate")
        pytest.skip("native execution requires mlir-opt beside mlir-translate")
    for tool in (opt, translate, clang):
        assert Path(tool).is_file() and os.access(tool, os.X_OK), f"native tool is not executable: {tool}"
    return opt, translate, clang


def native_tools() -> tuple[str, str, str]:
    """Public form of :func:`_tools` for tests with their own native pipeline."""
    return tuple(str(tool) for tool in _tools())


def _descriptor(array: np.ndarray):
    rank = array.ndim
    fields = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
    ]
    if rank:
        fields += [("sizes", ctypes.c_int64 * rank), ("strides", ctypes.c_int64 * rank)]

    class Descriptor(ctypes.Structure):
        _fields_ = fields

    if not rank:
        return Descriptor(array.ctypes.data, array.ctypes.data, 0)
    return Descriptor(
        array.ctypes.data,
        array.ctypes.data,
        0,
        (ctypes.c_int64 * rank)(*array.shape),
        (ctypes.c_int64 * rank)(*(stride // array.itemsize for stride in array.strides)),
    )


def _as_tuple(value):
    return tuple(torch.utils._pytree.tree_leaves(value))


def _lower_rng_globals(module):
    """Snapshot ml_program tensor globals into upstream memref globals for CPU tests."""
    from xdsl.dialects import bufferization, memref, ml_program
    from xdsl.dialects.builtin import MemRefType, UnregisteredOp
    from xdsl.rewriter import Rewriter
    types = {}
    for op in list(module.body.block.ops):
        if isinstance(op, ml_program.GlobalOp):
            typ = MemRefType(op.type.element_type, op.type.get_shape())
            types[op.sym_name.data] = typ
            Rewriter.replace_op(op, memref.GlobalOp.get(op.sym_name, typ, op.value))
    for op in list(module.walk()):
        if not isinstance(op, UnregisteredOp) or op.op_name.data not in ('ml_program.global_load', 'ml_program.global_store'):
            continue
        name = op.properties['global'].root_reference.data
        typ = types[name]
        get = memref.GetGlobalOp(name, typ)
        if op.op_name.data.endswith('load'):
            alloc = memref.AllocOp.get(typ.element_type, shape=typ.get_shape())
            copy_op = memref.CopyOp(get.memref, alloc.memref)
            snapshot = bufferization.ToTensorOp(alloc.memref, restrict=True)
            Rewriter.replace_op(op, [get, alloc, copy_op, snapshot], [snapshot.tensor])
        else:
            buffer = bufferization.ToBufferOp.create(operands=[op.operands[0]], result_types=[typ])
            copy_op = memref.CopyOp(buffer.memref, get.memref)
            Rewriter.replace_op(op, [get, buffer, copy_op], [])


def _with_lifted_state(exported_program, user_inputs, numpy_storage):
    from torch.export.graph_signature import InputKind

    assert exported_program is not None, "stateful capture needs its exported program"
    user = iter(user_inputs)
    arrays = []
    for spec in exported_program.graph_signature.input_specs:
        if spec.kind == InputKind.USER_INPUT:
            arrays.append(numpy_storage(next(user)))
            continue
        state = exported_program.state_dict.get(spec.target)
        if state is None:
            state = exported_program.constants[spec.target]
        arrays.append(numpy_storage(copy.deepcopy(state.detach())))
    return arrays


def run_native(tmp_path: Path, fn, inputs: tuple[torch.Tensor, ...], *, capture_trace=False,
               convert_options: dict | None = None) -> list[np.ndarray]:
    """Return the natively computed outputs of ``fn(*inputs)`` as numpy arrays.

    The capture must be free of opaque calls.  Output buffers are shaped from
    the eager result, so only statically shaped results are supported.
    ``convert_options`` are extra ``m2m.convert`` keywords (for example
    ``fully_standard=True`` to expand PT2E quantize/dequantize ops).
    """
    from xdsl.dialects.builtin import UnitAttr
    from xdsl.dialects.func import FuncOp

    opt, translate, clang = _tools()
    converted = m2m.convert(_Call(fn).eval(), inputs, backend="fx_importer", capture_trace=capture_trace,
                            **(convert_options or {}))
    assert converted.ok, converted.diagnostics
    converted.module.verify()
    assert "func.call" not in converted.mlir_text, converted.mlir_text
    _lower_rng_globals(converted.module)
    entry_name = None
    entry_types = []
    entry_args = []
    for op in converted.module.body.block.ops:
        if isinstance(op, FuncOp) and not op.is_declaration:
            op.attributes["llvm.emit_c_interface"] = UnitAttr()
            entry_name = op.sym_name.data
            entry_types = list(op.function_type.outputs)
            entry_args = list(op.function_type.inputs)
            break
    assert entry_name is not None
    source, lowered, llvm, library = (
        tmp_path / name for name in ("source.mlir", "lowered.mlir", "model.ll", "model.so")
    )
    from m2m.capture.torch_mlir_bridge import module_to_text
    source.write_text(module_to_text(converted.module, generic=capture_trace), encoding="utf-8")
    lowering = subprocess.run(
        [str(opt), str(source), "--pass-pipeline=" + PIPELINE, "-o", str(lowered)], capture_output=True, text=True
    )
    assert lowering.returncode == 0, f"mlir-opt failed on {source}:\n{lowering.stderr[:4000]}"
    subprocess.run([translate, "--mlir-to-llvmir", str(lowered), "-o", str(llvm)], check=True, capture_output=True)
    runner_lib = Path(translate).parent.parent / "lib"
    subprocess.run(
        [
            clang,
            "-O1",
            "-fPIC",
            "-shared",
            str(llvm),
            "-lm",
            "-L" + str(runner_lib),
            "-lmlir_c_runner_utils",
            "-Wl,-rpath," + str(runner_lib),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )

    # The bufferized entry point may update its argument buffers in place, so
    # hand it private copies and leave the caller's tensors untouched.
    def numpy_storage(t):
        value = t.detach()
        if value.is_complex():
            return torch.view_as_real(value).numpy()
        return value.view(torch.uint16).numpy() if value.dtype == torch.bfloat16 else value.numpy()

    # Deepcopy preserves storage holes, offsets and zero strides for resize_.
    private = copy.deepcopy(tuple(t.detach() for t in inputs))
    arrays = [numpy_storage(t) for t in private]
    if len(entry_args) != len(arrays):
        # Lifted parameters, buffers and tensor constants of a stateful module are
        # leading entry arguments in the exported program's input order.
        arrays = _with_lifted_state(converted.exported_program, private, numpy_storage)
        assert len(entry_args) == len(arrays), "captured entry arguments differ from the exported inputs"
    expected = _as_tuple(fn(*copy.deepcopy(tuple(t.detach() for t in inputs))))
    layout = converted.module.attributes.get("m2m.result_layout")
    records = json.loads(layout.data)["results"] if layout is not None else None
    if records is None:
        assert len(entry_types) == len(expected), "capture returned additional unbound results"
        user_indices = list(range(len(expected)))
        outputs = [np.empty(tuple(t.shape), dtype=numpy_storage(t).dtype) for t in expected]
    else:
        assert len(records) == len(entry_types), "result role count differs from captured ABI"
        user_indices = [i for i, r in enumerate(records) if r["role"] == "user_output"]
        assert len(user_indices) == len(expected), "user result count differs from eager"
        numpy_types = {
            "f16": np.float16,
            "bf16": np.uint16,
            "f32": np.float32,
            "f64": np.float64,
            "i64": np.int64,
            "i32": np.int32,
            "i16": np.int16,
            "i8": np.int8,
            "i1": np.bool_,
        }
        outputs = []
        for i, typ in enumerate(entry_types):
            assert all(n >= 0 for n in typ.get_shape()), "native probes require static outputs"
            dtype = numpy_types[str(typ.element_type)]
            if i in user_indices:
                # Signless MLIR integers carry their public signedness in the
                # eager result contract; both use identical descriptor bytes.
                want = expected[user_indices.index(i)]
                public_dtype = numpy_storage(want).dtype
                assert np.dtype(dtype).itemsize == public_dtype.itemsize
                assert tuple(typ.get_shape()) == tuple(numpy_storage(want).shape)
                dtype = public_dtype
            outputs.append(np.empty(tuple(typ.get_shape()), dtype=dtype))
    entry = getattr(ctypes.CDLL(str(library)), "_mlir_ciface_" + entry_name)
    descriptors = [_descriptor(a) for a in (*arrays, *outputs)]
    entry.argtypes = [ctypes.c_void_p] * len(descriptors)
    entry.restype = None
    entry(*[ctypes.byref(d) for d in descriptors])
    values = [outputs[i] for i in user_indices]
    return [a.view(numpy_storage(want).dtype if not want.is_complex() else want.numpy().dtype).reshape(want.shape)
            if want.is_complex() else a for a, want in zip(values, expected)]


def assert_integerized_exact(tmp_path: Path, captured, inputs, want, *, view=None):
    """The PT2E integer reference, the integerized program and its native build all equal ``want``.

    ``want`` is one array or a sequence of arrays; ``view`` compares storage words.
    """
    from m2m.capture.pt2e_compat import allow_exported_model_train_eval
    from m2m.capture.pt2e_integer_reference import run_pt2e_integer_reference
    from m2m.capture.pt2e_integerize import integerize_pt2e

    want = list(want) if isinstance(want, (list, tuple)) else [want]
    bits = (lambda a: np.asarray(a).view(view)) if view is not None else np.asarray
    reference = run_pt2e_integer_reference(captured, inputs, expected_contractions=1)
    integer, receipt = integerize_pt2e(captured, inputs)
    assert receipt["refusals"] == [] and receipt["quantized_contractions_integerized"] == 1, receipt["refusals"]
    for outputs in (integer(*inputs), reference.output):
        for got, expected in zip(_as_tuple(outputs), want, strict=True):
            np.testing.assert_array_equal(bits(got.numpy()), bits(expected))
    allow_exported_model_train_eval(integer)
    native = run_native(tmp_path, integer, inputs, convert_options={"fully_standard": True})
    for got, expected in zip(native, want, strict=True):
        np.testing.assert_array_equal(bits(got), bits(expected))
    return integer, receipt


def assert_native_matches_eager(tmp_path: Path, fn, inputs: tuple[torch.Tensor, ...], *,
                                convert_options: dict | None = None, **close):
    """Run natively and compare every result with eager PyTorch (NaNs equal)."""
    expected = _as_tuple(fn(*copy.deepcopy(tuple(t.detach() for t in inputs))))
    actual = run_native(tmp_path, fn, inputs, convert_options=convert_options)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        actual_tensor = torch.from_numpy(got)
        if want.dtype == torch.bfloat16:
            actual_tensor = actual_tensor.view(torch.bfloat16)
        torch.testing.assert_close(actual_tensor, want, equal_nan=True, **close)
