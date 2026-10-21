"""Native execution vs eager PyTorch for every one of the 39 support-set ops.

The variant grid (alpha/beta, strides, groups, dilation, transposed and 3-D conv,
backward masks, pool options, adaptive windows, pads, scatters, casts, fills,
integer dtypes) is the core-aten review probe (m2m_probe/probe39.py), made a
test. Each variant must lower with no opaque call, and its
native result must match eager: bit for bit for movement, elementwise, window
and integer ops; within float reassociation for float contractions and means.
Integerized int8 PT2E forms of the contractions are checked natively as well.
"""

from __future__ import annotations

import re

import pytest
import torch

import m2m
from m2m.capture.pt2e_compat import allow_exported_model_train_eval
from m2m.capture.pt2e_integerize import integerize_pt2e
from tests.native_execution import Q, _Call, assert_native_matches_eager, qdq

A = torch.ops.aten
g = torch.Generator().manual_seed(0)


def r(*s, dtype=torch.float32):
    if dtype.is_floating_point:
        return torch.randn(*s, generator=g).to(dtype)
    return torch.randint(-50, 50, s, generator=g).to(dtype)


V = []


def add(op, label, fn, *inputs):
    V.append((op, label, fn, tuple(inputs)))


x4 = r(1, 3, 7, 7)
add("_adaptive_avg_pool2d", "7x7->3x3 (non-divisible)", lambda x: A._adaptive_avg_pool2d.default(x, [3, 3]), x4)
add("_adaptive_avg_pool2d", "7x7->1x1", lambda x: A._adaptive_avg_pool2d.default(x, [1, 1]), x4)
add("_adaptive_avg_pool2d", "8x8->4x4 divisible", lambda x: A._adaptive_avg_pool2d.default(x, [4, 4]), r(2, 3, 8, 8))
add("_adaptive_avg_pool2d_backward", "7x7->3x3", lambda gr, x: A._adaptive_avg_pool2d_backward.default(gr, x), r(1, 3, 3, 3), x4)
add("_adaptive_avg_pool2d_backward", "8x8->4x4", lambda gr, x: A._adaptive_avg_pool2d_backward.default(gr, x), r(1, 3, 4, 4), r(1, 3, 8, 8))
add("_adaptive_avg_pool3d", "5x6x7->2x3x4", lambda x: A._adaptive_avg_pool3d.default(x, [2, 3, 4]), r(1, 2, 5, 6, 7))
add("adaptive_avg_pool1d", "10->4", lambda x: A.adaptive_avg_pool1d.default(x, [4]), r(1, 3, 10))
add("_to_copy", "f32->f16", lambda x: A._to_copy.default(x, dtype=torch.float16), r(4, 5))
add("_to_copy", "f32->bf16", lambda x: A._to_copy.default(x, dtype=torch.bfloat16), r(4, 5))
add("_to_copy", "f32->i32 (trunc)", lambda x: A._to_copy.default(x * 10, dtype=torch.int32), r(4, 5))
add("_to_copy", "f32->i8 in-range", lambda x: A._to_copy.default(x * 10, dtype=torch.int8), r(4, 5))
add("_to_copy", "i8->f32", lambda x: A._to_copy.default(x, dtype=torch.float32), r(4, 5, dtype=torch.int8))
add("_to_copy", "i8->i32", lambda x: A._to_copy.default(x, dtype=torch.int32), r(4, 5, dtype=torch.int8))
add("_to_copy", "i32->i8 (wrap)", lambda x: A._to_copy.default(x * 7, dtype=torch.int8), r(4, 5, dtype=torch.int32))
add("_to_copy", "f32->bool", lambda x: A._to_copy.default(x, dtype=torch.bool), r(4, 5))
add("add.Scalar", "x+2.5 alpha=2", lambda x: A.add.Scalar(x, 2.5, alpha=2), r(3, 4))
add("add.Scalar", "i32 + 3", lambda x: A.add.Scalar(x, 3), r(3, 4, dtype=torch.int32))
add("add.Tensor", "bcast alpha=0.5", lambda a, b: A.add.Tensor(a, b, alpha=0.5), r(3, 4), r(4))
add("add.Tensor", "i8+i8 (wrap)", lambda a, b: A.add.Tensor(a, b), (r(3, 4, dtype=torch.int8) * 3).to(torch.int8), (r(3, 4, dtype=torch.int8) * 3).to(torch.int8))
add("add.Tensor", "i32 alpha=3", lambda a, b: A.add.Tensor(a, b, alpha=3), r(3, 4, dtype=torch.int32), r(3, 4, dtype=torch.int32))
add("addmm", "beta=0.5 alpha=2", lambda c, a, b: A.addmm.default(c, a, b, beta=0.5, alpha=2.0), r(3, 5), r(3, 4), r(4, 5))
add("addmm", "1-D bias default", lambda c, a, b: A.addmm.default(c, a, b), r(5), r(3, 4), r(4, 5))
add("addmm", "beta=0 ignores NaN self", lambda c, a, b: A.addmm.default(c, a, b, beta=0.0), torch.full((3, 5), float("nan")), r(3, 4), r(4, 5))
add("addmm", "i32 (int)", lambda c, a, b: A.addmm.default(c, a, b), r(3, 5, dtype=torch.int32), r(3, 4, dtype=torch.int32), r(4, 5, dtype=torch.int32))
add("avg_pool1d", "k3 s2 p1 ceil cip=False", lambda x: A.avg_pool1d.default(x, [3], [2], [1], True, False), r(1, 3, 10))
add("avg_pool1d", "k2 default", lambda x: A.avg_pool1d.default(x, [2]), r(1, 3, 10))
add("avg_pool2d", "k3 s2 p1 default", lambda x: A.avg_pool2d.default(x, [3, 3], [2, 2], [1, 1]), x4)
add("avg_pool2d", "k3 s2 p1 ceil cip=False", lambda x: A.avg_pool2d.default(x, [3, 3], [2, 2], [1, 1], True, False), x4)
add("avg_pool2d", "k3 s2 p1 ceil cip=True", lambda x: A.avg_pool2d.default(x, [3, 3], [2, 2], [1, 1], True, True), x4)
add("avg_pool2d", "k3 s2 p1 divisor_override=5", lambda x: A.avg_pool2d.default(x, [3, 3], [2, 2], [1, 1], False, True, 5), x4)
add("avg_pool2d", "k(2,3) s(1,2) rect", lambda x: A.avg_pool2d.default(x, [2, 3], [1, 2]), x4)
add("avg_pool2d_backward", "k3 s2 p1 ceil cip=False", lambda gr, x: A.avg_pool2d_backward.default(gr, x, [3, 3], [2, 2], [1, 1], True, False, None),
    r(1, 3, 4, 4), x4)
add("avg_pool2d_backward", "k2 s2 default", lambda gr, x: A.avg_pool2d_backward.default(gr, x, [2, 2], [2, 2], [0, 0], False, True, None), r(1, 3, 4, 4), r(1, 3, 8, 8))
add("avg_pool2d_backward", "k3 s2 p1 divisor=4", lambda gr, x: A.avg_pool2d_backward.default(gr, x, [3, 3], [2, 2], [1, 1], False, True, 4),
    r(1, 3, 4, 4), x4)
add("avg_pool2d_backward", "empty stride", lambda gr, x: A.avg_pool2d_backward.default(gr, x, [2, 2], [], [0, 0], False, True, None), r(1, 3, 4, 4), r(1, 3, 8, 8))
add("avg_pool3d", "k2 s2 p1 ceil cip=False", lambda x: A.avg_pool3d.default(x, [2, 2, 2], [2, 2, 2], [1, 1, 1], True, False), r(1, 2, 5, 5, 5))
add("avg_pool3d", "k2 divisor=3", lambda x: A.avg_pool3d.default(x, [2, 2, 2], [2, 2, 2], [0, 0, 0], False, True, 3), r(1, 2, 4, 4, 4))
add("bmm", "f32", lambda a, b: A.bmm.default(a, b), r(2, 3, 4), r(2, 4, 5))
add("bmm", "i32", lambda a, b: A.bmm.default(a, b), r(2, 3, 4, dtype=torch.int32), r(2, 4, 5, dtype=torch.int32))
add("cat", "dim1", lambda a, b: A.cat.default([a, b], 1), r(2, 3), r(2, 4))
add("cat", "dim-1 3 inputs", lambda a, b, c: A.cat.default([a, b, c], -1), r(2, 3, 1), r(2, 3, 2), r(2, 3, 3))
add("cat", "with empty 1-D legacy", lambda a, b: A.cat.default([a, b], 0), r(2, 3), torch.empty(0))
add("cat", "mixed dtype promote", lambda a, b: A.cat.default([a, b], 0), r(2, 3), r(1, 3, dtype=torch.int32))
add("clone", "plain", lambda x: A.clone.default(x), r(2, 3))
add("clone", "channels_last", lambda x: A.clone.default(x, memory_format=torch.channels_last), r(1, 3, 4, 5))
add("col2im", "out(4,5) k(2,2) s1", lambda x: A.col2im.default(x, [4, 5], [2, 2], [1, 1], [0, 0], [1, 1]), r(1, 3 * 4, 12))
add("col2im", "out(5,6) k(2,3) d(1,2) p1 s(2,1)", lambda x: A.col2im.default(x, [5, 6], [2, 3], [1, 2], [1, 1], [2, 1]),
    r(1, 2 * 6, 3 * 4))
add("constant_pad_nd", "pad(1,2,0,1) v=1.5", lambda x: A.constant_pad_nd.default(x, [1, 2, 0, 1], 1.5), r(1, 2, 3, 4))
add("constant_pad_nd", "negative pad", lambda x: A.constant_pad_nd.default(x, [-1, 1], 0.0), r(2, 5))
add("constant_pad_nd", "i8 pad v=-3", lambda x: A.constant_pad_nd.default(x, [1, 1], -3), r(2, 5, dtype=torch.int8))
W = r(4, 3, 3, 3)
add("convolution", "2d s1 p1 bias", lambda x, w, b: A.convolution.default(x, w, b, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), x4, W, r(4))
add("convolution", "2d s2 p1 d2 nobias", lambda x, w: A.convolution.default(x, w, None, [2, 2], [1, 1], [2, 2], False, [0, 0], 1), r(1, 3, 9, 9), W)
add("convolution", "2d groups=2", lambda x, w, b: A.convolution.default(x, w, b, [1, 1], [0, 0], [1, 1], False, [0, 0], 2), r(1, 4, 6, 6), r(6, 2, 3, 3), r(6))
add("convolution", "2d depthwise", lambda x, w: A.convolution.default(x, w, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4), r(1, 4, 6, 6), r(4, 1, 3, 3))
add("convolution", "2d asym pad (0,2) stride(1,2)", lambda x, w: A.convolution.default(x, w, None, [1, 2], [0, 2], [1, 1], False, [0, 0], 1), x4, W)
add("convolution", "transposed s2 p1 op1", lambda x, w, b: A.convolution.default(x, w, b, [2, 2], [1, 1], [1, 1], True, [1, 1], 1), r(1, 4, 5, 5), r(4, 3, 3, 3), r(3))
add("convolution", "transposed groups2 d2", lambda x, w: A.convolution.default(x, w, None, [2, 2], [0, 0], [2, 2], True, [0, 0], 2), r(1, 4, 4, 4), r(4, 3, 3, 3))
add("convolution", "1d s2 p1", lambda x, w, b: A.convolution.default(x, w, b, [2], [1], [1], False, [0], 1), r(1, 3, 10), r(4, 3, 3), r(4))
add("convolution", "3d s1 p1", lambda x, w: A.convolution.default(x, w, None, [1, 1, 1], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), r(1, 2, 4, 4, 4), r(3, 2, 2, 2, 2))
add("convolution", "1x1 patch-embed s4 k4", lambda x, w: A.convolution.default(x, w, None, [4, 4], [0, 0], [1, 1], False, [0, 0], 1), r(1, 3, 8, 8), r(8, 3, 4, 4))
add("convolution", "i32 conv (int)", lambda x, w: A.convolution.default(x, w, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), r(1, 2, 4, 4, dtype=torch.int32), r(2, 2, 3, 3, dtype=torch.int32))
gout = r(1, 4, 7, 7)
add("convolution_backward", "mask TTT s1 p1", lambda go, x, w: A.convolution_backward.default(go, x, w, [4], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True]), gout, x4, W)
add("convolution_backward", "mask FTF", lambda go, x, w: A.convolution_backward.default(go, x, w, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])[1], gout, x4, W)
add("convolution_backward", "mask TFF", lambda go, x, w: A.convolution_backward.default(go, x, w, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, False, False])[0], gout, x4, W)
add("convolution_backward", "s2 p1 d2 groups2 TTT", lambda go, x, w: A.convolution_backward.default(go, x, w, [6], [2, 2], [1, 1], [2, 2], False, [0, 0], 2, [True, True, True]),
    r(1, 6, 4, 4), r(1, 4, 9, 9), r(6, 2, 3, 3))
add("convolution_backward", "transposed s2 TTT", lambda go, x, w: A.convolution_backward.default(go, x, w, [3], [2, 2], [1, 1], [1, 1], True, [1, 1], 1, [True, True, True]),
    r(1, 3, 10, 10), r(1, 4, 5, 5), r(4, 3, 3, 3))
add("convolution_backward", "1d TTT", lambda go, x, w: A.convolution_backward.default(go, x, w, [4], [1], [0], [1], False, [0], 1, [True, True, True]),
    r(1, 4, 8), r(1, 3, 10), r(4, 3, 3))
add("div.Scalar", "x/3", lambda x: A.div.Scalar(x, 3.0), r(3, 4))
add("div.Scalar", "i32/2 -> f32", lambda x: A.div.Scalar(x, 2), r(3, 4, dtype=torch.int32))
add("fill.Scalar", "f32 2.0", lambda x: A.fill.Scalar(x, 2.0), r(3, 4))
add("fill.Scalar", "i8 -5", lambda x: A.fill.Scalar(x, -5), r(3, 4, dtype=torch.int8))
add("full", "[2,3] 7.0", lambda x: x + A.full.default([2, 3], 7.0), r(2, 3))
add("full", "int32", lambda x: x + A.full.default([2, 3], 7, dtype=torch.int32), r(2, 3, dtype=torch.int32))
add("full_like", "2.0", lambda x: A.full_like.default(x, 2.0), r(2, 3))
add("full_like", "dtype=int32", lambda x: A.full_like.default(x, 2, dtype=torch.int32), r(2, 3))
add("mean.dim", "dim[1] keepdim=F", lambda x: A.mean.dim(x, [1]), r(2, 3, 4))
add("mean.dim", "dim[1] keepdim=T", lambda x: A.mean.dim(x, [1], True), r(2, 3, 4))
add("mean.dim", "dim[-1,-2] keepdim=T", lambda x: A.mean.dim(x, [-1, -2], True), r(2, 3, 4, 5))
add("mean.dim", "dtype=f64", lambda x: A.mean.dim(x, [1], False, dtype=torch.float64), r(2, 3, 4))
add("mm", "f32", lambda a, b: A.mm.default(a, b), r(3, 4), r(4, 5))
add("mm", "i32", lambda a, b: A.mm.default(a, b), r(3, 4, dtype=torch.int32), r(4, 5, dtype=torch.int32))
add("mul.Scalar", "x*0.25", lambda x: A.mul.Scalar(x, 0.25), r(3, 4))
add("mul.Scalar", "i32*3", lambda x: A.mul.Scalar(x, 3), r(3, 4, dtype=torch.int32))
add("neg", "f32", lambda x: A.neg.default(x), r(3, 4))
add("neg", "i8", lambda x: A.neg.default(x), r(3, 4, dtype=torch.int8))
add("permute", "[2,0,1]", lambda x: A.permute.default(x, [2, 0, 1]), r(2, 3, 4))
add("permute", "[0,2,3,1] NCHW->NHWC", lambda x: A.permute.default(x, [0, 2, 3, 1]), x4)
add("reflection_pad1d", "(2,1)", lambda x: A.reflection_pad1d.default(x, [2, 1]), r(1, 2, 5))
add("reflection_pad2d", "(1,2,2,1)", lambda x: A.reflection_pad2d.default(x, [1, 2, 2, 1]), r(1, 2, 4, 5))
add("reflection_pad3d", "(1,1,1,0,0,1)", lambda x: A.reflection_pad3d.default(x, [1, 1, 1, 0, 0, 1]), r(1, 2, 3, 4, 3))
add("relu", "f32", lambda x: A.relu.default(x), r(3, 4))
add("relu", "i8", lambda x: A.relu.default(x), r(3, 4, dtype=torch.int8))
add("repeat", "(2,1,3) on 2D", lambda x: A.repeat.default(x, [2, 1, 3]), r(2, 3))
add("repeat", "(1,2)", lambda x: A.repeat.default(x, [1, 2]), r(2, 3))
add("replication_pad2d", "(1,2,2,1)", lambda x: A.replication_pad2d.default(x, [1, 2, 2, 1]), r(1, 2, 4, 5))
add("replication_pad3d", "(1,1,1,0,0,1)", lambda x: A.replication_pad3d.default(x, [1, 1, 1, 0, 0, 1]), r(1, 2, 3, 4, 3))
add("select_scatter", "dim1 idx-1", lambda x, s: A.select_scatter.default(x, s, 1, -1), r(2, 3, 4), r(2, 4))
add("select_scatter", "dim0 idx1", lambda x, s: A.select_scatter.default(x, s, 0, 1), r(3, 4), r(4))
add("slice_scatter", "dim1 1:None:2", lambda x, s: A.slice_scatter.default(x, s, 1, 1, None, 2), r(2, 6), r(2, 3))
add("slice_scatter", "dim0 -2:", lambda x, s: A.slice_scatter.default(x, s, 0, -2), r(4, 3), r(2, 3))
add("slice_scatter", "dim1 0:3 step1", lambda x, s: A.slice_scatter.default(x, s, 1, 0, 3), r(2, 5), r(2, 3))
add("sub.Scalar", "x-1.5 alpha=2", lambda x: A.sub.Scalar(x, 1.5, alpha=2), r(3, 4))
add("sub.Tensor", "alpha=2 bcast", lambda a, b: A.sub.Tensor(a, b, alpha=2), r(3, 4), r(4))
add("sub.Tensor", "i32", lambda a, b: A.sub.Tensor(a, b), r(3, 4, dtype=torch.int32), r(3, 4, dtype=torch.int32))


SUPPORT_SET = {
    "_adaptive_avg_pool2d", "_adaptive_avg_pool2d_backward", "_adaptive_avg_pool3d", "_to_copy",
    "adaptive_avg_pool1d", "add.Scalar", "add.Tensor", "addmm", "avg_pool1d", "avg_pool2d",
    "avg_pool2d_backward", "avg_pool3d", "bmm", "cat", "clone", "col2im", "constant_pad_nd",
    "convolution", "convolution_backward", "div.Scalar", "fill.Scalar", "full", "full_like", "mean.dim",
    "mm", "mul.Scalar", "neg", "permute", "reflection_pad1d", "reflection_pad2d", "reflection_pad3d",
    "relu", "repeat", "replication_pad2d", "replication_pad3d", "select_scatter", "slice_scatter",
    "sub.Scalar", "sub.Tensor",
}
# Float contractions and means reassociate relative to the CPU kernels.
REASSOCIATED = {"addmm", "bmm", "mm", "convolution", "convolution_backward", "mean.dim"}
GATHER = ("scf.for", "scf.while", "tensor.extract ")
ORDERED_CPU_CONTRACTIONS = {"addmm", "convolution", "convolution_backward"}


def test_grid_covers_the_39_support_set_ops():
    assert {op for op, *_ in V} == SUPPORT_SET and len(SUPPORT_SET) == 39


@pytest.mark.parametrize("op, label, fn, inputs", V, ids=[f"{op}:{label}" for op, label, *_ in V])
def test_support_set_op_matches_eager(tmp_path, op, label, fn, inputs):
    result = m2m.convert(_Call(fn).eval(), inputs, backend="fx_importer")
    assert result.ok and "func.call" not in result.mlir_text
    # Native contraction order needs explicit scalar loops/loads. linalg.index
    # also describes SIMD slots in ordinary pointwise kernels; it is no gather.
    if op not in ORDERED_CPU_CONTRACTIONS:
        assert not [token for token in GATHER if token in result.mlir_text], "gather loop in " + label
    floating = any(t.is_floating_point() for t in inputs)
    tolerance = 1e-5 if op in REASSOCIATED and floating else 0
    assert_native_matches_eager(tmp_path, fn, inputs, rtol=tolerance, atol=tolerance)


INT8 = {
    "mm": (lambda a, b: Q(A.mm.default(qdq(a, 0.05), qdq(b, 0.04, -127)), 0.2, 0, -128, 127, torch.int8),
           (r(4, 8), r(8, 5))),
    "bmm": (lambda a, b: A.bmm.default(qdq(a, 0.05), qdq(b, 0.04)), (r(2, 4, 8), r(2, 8, 5))),
    "addmm": (lambda c, a, b: A.addmm.default(c, qdq(a, 0.05), qdq(b, 0.04), beta=0.5, alpha=2.0),
              (r(5), r(4, 8), r(8, 5))),
    "convolution": (lambda x, w, b: A.convolution.default(qdq(x, 0.05), qdq(w, 0.02), b, [1, 1], [1, 1], [1, 1],
                                                          False, [0, 0], 2), (r(1, 4, 5, 5), r(6, 2, 3, 3), r(6))),
    "convolution_backward": (lambda go, x, w: A.convolution_backward.default(
        qdq(go, 0.03), qdq(x, 0.05), qdq(w, 0.02), [4], [1, 1], [1, 1], [1, 1], False, [0, 0], 1,
        [True, True, True]), (r(1, 4, 5, 5), r(1, 3, 5, 5), r(4, 3, 3, 3))),
}


@pytest.mark.parametrize("numerics", ["float_readout", "integer_readout"])
@pytest.mark.parametrize("op", sorted(INT8))
def test_int8_contraction_integerizes_and_executes_natively(tmp_path, op, numerics):
    fn, inputs = INT8[op]
    model = torch.export.export(_Call(fn).eval(), inputs).module()
    integer, receipt = integerize_pt2e(model, inputs, numerics=numerics)
    assert receipt["refusals"] == [] and receipt["quantized_contractions_integerized"] == 1
    allow_exported_model_train_eval(integer)
    lowered = m2m.convert(integer, inputs, backend="fx_importer")
    contractions = [line for line in lowered.mlir_text.splitlines()
                    if "linalg.generic" in line and '"int_matmul"' in line]
    if op == "bmm":
        assert re.search(r'prov.aten = "aten.bmm.default"', lowered.mlir_text) and "arith.extsi" in lowered.mlir_text
    else:
        assert contractions and all("xi8>, tensor<" in line for line in contractions)
    assert not [token for token in GATHER if token in lowered.mlir_text]
    assert_native_matches_eager(tmp_path, integer, inputs, rtol=0, atol=0, convert_options={"fully_standard": True})
