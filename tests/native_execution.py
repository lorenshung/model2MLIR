"""Compile a captured single-op program with upstream MLIR and run it natively.

Shared by regression tests that must check values, not only that a lowering is
free of opaque calls.  Tests skip when the MLIR/LLVM tools are unavailable;
point ``MODEL2MLIR_MLIR_TRANSLATE`` (with ``mlir-opt`` beside it) and
``MODEL2MLIR_CLANG`` at an LLVM build to enable them.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
from pathlib import Path

import m2m
import numpy as np
import pytest
import torch

PIPELINE = (
    "builtin.module(one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},"
    "buffer-results-to-out-params{modify-public-functions},convert-linalg-to-loops,"
    "expand-strided-metadata,lower-affine,convert-scf-to-cf,arith-expand,math-expand-ops{ops=rsqrt},"
    "convert-math-to-libm,convert-math-to-llvm,convert-index-to-llvm,"
    "convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,"
    "convert-cf-to-llvm,reconcile-unrealized-casts)"
)


class _Call(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def _tools():
    translate = os.environ.get("MODEL2MLIR_MLIR_TRANSLATE") or shutil.which("mlir-translate")
    clang = os.environ.get("MODEL2MLIR_CLANG") or shutil.which("clang")
    if not translate or not clang:
        pytest.skip("native execution requires MLIR/LLVM tools")
    opt = Path(translate).with_name("mlir-opt")
    if not opt.is_file():
        pytest.skip("native execution requires mlir-opt beside mlir-translate")
    return opt, translate, clang


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
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def run_native(tmp_path: Path, fn, inputs: tuple[torch.Tensor, ...]) -> list[np.ndarray]:
    """Return the natively computed outputs of ``fn(*inputs)`` as numpy arrays.

    The capture must be free of opaque calls.  Output buffers are shaped from
    the eager result, so only statically shaped results are supported.
    """
    from xdsl.dialects.builtin import UnitAttr
    from xdsl.dialects.func import FuncOp

    opt, translate, clang = _tools()
    converted = m2m.convert(_Call(fn).eval(), inputs, backend="fx_importer")
    assert converted.ok, converted.diagnostics
    converted.module.verify()
    assert "func.call" not in converted.mlir_text, converted.mlir_text
    entry_name = None
    entry_types = []
    for op in converted.module.body.block.ops:
        if isinstance(op, FuncOp) and not op.is_declaration:
            op.attributes["llvm.emit_c_interface"] = UnitAttr()
            entry_name = op.sym_name.data
            entry_types = list(op.function_type.outputs)
            break
    assert entry_name is not None
    source, lowered, llvm, library = (
        tmp_path / name for name in ("source.mlir", "lowered.mlir", "model.ll", "model.so")
    )
    source.write_text(str(converted.module), encoding="utf-8")
    subprocess.run(
        [str(opt), str(source), "--pass-pipeline=" + PIPELINE, "-o", str(lowered)], check=True, capture_output=True
    )
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
        value = t.detach().contiguous()
        return value.view(torch.uint16).numpy() if value.dtype == torch.bfloat16 else value.numpy()

    arrays = [np.array(numpy_storage(t), copy=True, order="C") for t in inputs]
    expected = _as_tuple(fn(*[t.clone() for t in inputs]))
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
                assert tuple(typ.get_shape()) == tuple(want.shape)
                dtype = public_dtype
            outputs.append(np.empty(tuple(typ.get_shape()), dtype=dtype))
    entry = getattr(ctypes.CDLL(str(library)), "_mlir_ciface_" + entry_name)
    descriptors = [_descriptor(a) for a in (*arrays, *outputs)]
    entry.argtypes = [ctypes.c_void_p] * len(descriptors)
    entry.restype = None
    entry(*[ctypes.byref(d) for d in descriptors])
    return [outputs[i] for i in user_indices]


def assert_native_matches_eager(tmp_path: Path, fn, inputs: tuple[torch.Tensor, ...], **close):
    """Run natively and compare every result with eager PyTorch (NaNs equal)."""
    expected = _as_tuple(fn(*[t.clone() for t in inputs]))
    actual = run_native(tmp_path, fn, inputs)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        actual_tensor = torch.from_numpy(got)
        if want.dtype == torch.bfloat16:
            actual_tensor = actual_tensor.view(torch.bfloat16)
        torch.testing.assert_close(actual_tensor, want, equal_nan=True, **close)
