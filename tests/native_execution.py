"""Compile a captured single-op program with upstream MLIR and run it natively.

Shared by regression tests that must check values, not only that a lowering is
free of opaque calls.  Tests skip when the MLIR/LLVM tools are unavailable;
point ``MODEL2MLIR_MLIR_TRANSLATE`` (with ``mlir-opt`` beside it) and
``MODEL2MLIR_CLANG`` at an LLVM build to enable them.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

import m2m

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
        array.ctypes.data, array.ctypes.data, 0,
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
    for op in converted.module.body.block.ops:
        if isinstance(op, FuncOp) and not op.is_declaration:
            op.attributes["llvm.emit_c_interface"] = UnitAttr()
            entry_name = op.sym_name.data
            break
    assert entry_name is not None
    source, lowered, llvm, library = (
        tmp_path / name for name in ("source.mlir", "lowered.mlir", "model.ll", "model.so")
    )
    source.write_text(str(converted.module), encoding="utf-8")
    subprocess.run([str(opt), str(source), "--pass-pipeline=" + PIPELINE, "-o", str(lowered)],
                   check=True, capture_output=True)
    subprocess.run([translate, "--mlir-to-llvmir", str(lowered), "-o", str(llvm)],
                   check=True, capture_output=True)
    runner_lib = Path(translate).parent.parent / "lib"
    subprocess.run([
        clang, "-O1", "-fPIC", "-shared", str(llvm), "-lm",
        "-L" + str(runner_lib), "-lmlir_c_runner_utils", "-Wl,-rpath," + str(runner_lib),
        "-o", str(library),
    ], check=True, capture_output=True)

    # The bufferized entry point may update its argument buffers in place, so
    # hand it private copies and leave the caller's tensors untouched.
    arrays = [np.array(t.detach().numpy(), copy=True, order="C") for t in inputs]
    expected = _as_tuple(fn(*[t.clone() for t in inputs]))
    outputs = [np.empty(tuple(t.shape), dtype=t.detach().numpy().dtype) for t in expected]
    entry = getattr(ctypes.CDLL(str(library)), "_mlir_ciface_" + entry_name)
    descriptors = [_descriptor(a) for a in (*arrays, *outputs)]
    entry.argtypes = [ctypes.c_void_p] * len(descriptors)
    entry.restype = None
    entry(*[ctypes.byref(d) for d in descriptors])
    return outputs


def assert_native_matches_eager(tmp_path: Path, fn, inputs: tuple[torch.Tensor, ...], **close):
    """Run natively and compare every result with eager PyTorch (NaNs equal)."""
    expected = _as_tuple(fn(*[t.clone() for t in inputs]))
    actual = run_native(tmp_path, fn, inputs)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(torch.from_numpy(got), want, equal_nan=True, **close)
