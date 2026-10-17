"""Integer-accumulator readout operators for integerized PT2E graphs.

``integerize_pt2e(..., numerics="integer_readout")`` expresses an int8 x int8 ->
int32 contraction the way an int8 accelerator executes it:

* ``m2m_q.int_mm_seeded(a, b, seed)``: ``a @ b`` accumulated in int32 starting
  from ``seed``, the int32 bias ``round(b / (s_a * s_w))`` (the accumulator
  preload), broadcast to the result.
* ``m2m_q.requantize(acc, scale, zero_point, quant_min, quant_max, relu)``: one
  fused readout ``clamp(roundeven(f32(acc) * f32(scale)) + zero_point)`` with
  ``scale = s_a * s_w / s_out``. ``relu`` raises the lower bound to the zero
  point (ReLU before the scale is equivalent for a positive scale).
* ``m2m_q.acc_readout(acc, scale)``: an f32 result ``f32(acc) * f32(scale)``
  with the single scale ``s_a * s_w`` when no quantize follows.

``scale`` is a list: one value is per tensor, otherwise one value per channel
along the last result axis. Scales stay literal arguments so the importer can
emit them as constants and as ``prov.quant.*`` attributes. These are the
gate-3-validated semantics (int32 bias at full width, one RNE readout with
saturation); they are not PT2E's portable dequantize-then-float evaluation.
"""

from __future__ import annotations

import torch

_LIB = torch.library.Library("m2m_q", "DEF")
_LIB.define("int_mm_seeded(Tensor a, Tensor b, Tensor seed) -> Tensor")
_LIB.define(
    "requantize(Tensor acc, float[] scale, int zero_point, int quant_min, int quant_max, bool relu=False) -> Tensor"
)
_LIB.define("acc_readout(Tensor acc, float[] scale) -> Tensor")


def _check_int8(name: str, value: torch.Tensor) -> None:
    if value.dtype != torch.int8:
        raise TypeError(f"m2m_q.int_mm_seeded {name} must be int8, got {value.dtype}")


def _scale_tensor(acc: torch.Tensor, scale: list[float]) -> torch.Tensor:
    if len(scale) == 1:
        return torch.tensor(scale[0], dtype=torch.float32, device=acc.device)
    if acc.ndim < 1 or acc.shape[-1] != len(scale):
        raise ValueError(f"{len(scale)} channel scales do not match the last axis of {tuple(acc.shape)}")
    return torch.tensor(scale, dtype=torch.float32, device=acc.device)


@torch.library.impl(_LIB, "int_mm_seeded", "CompositeExplicitAutograd")
def _int_mm_seeded(a: torch.Tensor, b: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    _check_int8("lhs", a)
    _check_int8("rhs", b)
    if seed.dtype != torch.int32:
        raise TypeError(f"m2m_q.int_mm_seeded seed must be int32, got {seed.dtype}")
    # Exact in int64; the integerizer refuses K for which i32 could overflow,
    # and int32 wraparound of the sum equals an i32 MAC chain's wraparound.
    acc = torch.matmul(a.to(torch.int64), b.to(torch.int64)) + seed.to(torch.int64)
    return acc.to(torch.int32)


@torch.library.register_fake("m2m_q::int_mm_seeded")
def _int_mm_seeded_fake(a, b, seed):
    return a.new_empty((a.shape[0], b.shape[1]), dtype=torch.int32)


@torch.library.impl(_LIB, "requantize", "CompositeExplicitAutograd")
def _requantize(acc, scale, zero_point, quant_min, quant_max, relu=False):
    if acc.dtype != torch.int32:
        raise TypeError(f"m2m_q.requantize accumulator must be int32, got {acc.dtype}")
    if not -128 <= quant_min < quant_max <= 127:
        raise ValueError("m2m_q.requantize requires a signed int8 range")
    value = torch.round(acc.to(torch.float32) * _scale_tensor(acc, scale))  # half to even
    value = value + float(zero_point)
    low = max(quant_min, zero_point) if relu else quant_min
    return value.clamp(float(low), float(quant_max)).to(torch.int8)


@torch.library.register_fake("m2m_q::requantize")
def _requantize_fake(acc, scale, zero_point, quant_min, quant_max, relu=False):
    return acc.new_empty(acc.shape, dtype=torch.int8)


@torch.library.impl(_LIB, "acc_readout", "CompositeExplicitAutograd")
def _acc_readout(acc, scale):
    if acc.dtype != torch.int32:
        raise TypeError(f"m2m_q.acc_readout accumulator must be int32, got {acc.dtype}")
    return acc.to(torch.float32) * _scale_tensor(acc, scale)


@torch.library.register_fake("m2m_q::acc_readout")
def _acc_readout_fake(acc, scale):
    return acc.new_empty(acc.shape, dtype=torch.float32)


int_mm_seeded = torch.ops.m2m_q.int_mm_seeded.default
requantize = torch.ops.m2m_q.requantize.default
acc_readout = torch.ops.m2m_q.acc_readout.default

__all__ = ["acc_readout", "int_mm_seeded", "requantize"]
