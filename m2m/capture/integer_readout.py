"""Exact-arithmetic readout operators for integerized PT2E graphs.

Original binary scale factors and bias are evaluated as rational numbers.
There is one round-half-to-even operation at the final integer code (before
adding the zero point), or one IEEE rounding at the final f32 result. No
reciprocal approximation, combined-scale rounding, bias rounding or saturation
precedes it. Per-channel parameters broadcast along the last result axis.

int_mm_seeded is admitted only with a declaration-wide i32 overflow bound.
The integerizer uses it only for bias that is exactly an integral accumulator
seed. Other finite f32 bias remains an operand of exact_readout.
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


@torch.library.impl(_LIB, "int_mm_seeded", "CompositeExplicitAutograd")
def _int_mm_seeded(a: torch.Tensor, b: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    _check_int8("lhs", a)
    _check_int8("rhs", b)
    if seed.dtype != torch.int32:
        raise TypeError(f"m2m_q.int_mm_seeded seed must be int32, got {seed.dtype}")
    bound = a.shape[-1] * 128 * 128
    if bound > (1 << 31) - 1 or bool(((seed.to(torch.int64) < -(1 << 31) + bound)
                                                     | (seed.to(torch.int64) > (1 << 31) - 1 - bound)).any()):
        raise ValueError("seeded contraction may overflow its int32 accumulator")
    acc = torch.matmul(a.to(torch.int64), b.to(torch.int64)) + seed.to(torch.int64)
    return acc.to(torch.int32)


@torch.library.register_fake("m2m_q::int_mm_seeded")
def _int_mm_seeded_fake(a, b, seed):
    return a.new_empty((a.shape[0], b.shape[1]), dtype=torch.int32)


@torch.library.impl(_LIB, "requantize", "CompositeExplicitAutograd")
def _requantize(acc, scale, zero_point, quant_min, quant_max, relu=False):
    from m2m.capture.exact_quant import exact_readout
    return exact_readout(acc, 1.0, scale, output_scale=1.0, zero_point=zero_point,
                         quant_min=quant_min, quant_max=quant_max, relu=relu)


@torch.library.register_fake("m2m_q::requantize")
def _requantize_fake(acc, scale, zero_point, quant_min, quant_max, relu=False):
    return acc.new_empty(acc.shape, dtype=torch.int8)


@torch.library.impl(_LIB, "acc_readout", "CompositeExplicitAutograd")
def _acc_readout(acc, scale):
    from m2m.capture.exact_quant import exact_readout
    return exact_readout(acc, 1.0, scale)


@torch.library.register_fake("m2m_q::acc_readout")
def _acc_readout_fake(acc, scale):
    return acc.new_empty(acc.shape, dtype=torch.float32)


int_mm_seeded = torch.ops.m2m_q.int_mm_seeded.default
requantize = torch.ops.m2m_q.requantize.default
acc_readout = torch.ops.m2m_q.acc_readout.default

__all__ = ["acc_readout", "exact_readout", "int_mm_seeded", "quantize", "quantize_channel", "requantize"]

_LIB.define("exact_readout(Tensor acc, float activation_scale, float[] weight_scale, float alpha=1., Tensor? bias=None, float beta=1., float? output_scale=None, int zero_point=0, int quant_min=-128, int quant_max=127, bool relu=False, bool unsigned=False) -> Tensor")


@torch.library.impl(_LIB, "exact_readout", "CompositeExplicitAutograd")
def _exact_readout(*args, **kwargs):
    from m2m.capture.exact_quant import exact_readout as evaluate
    return evaluate(*args, **kwargs)


@torch.library.register_fake("m2m_q::exact_readout")
def _exact_readout_fake(acc, activation_scale, weight_scale, alpha=1., bias=None, beta=1.,
                        output_scale=None, zero_point=0, quant_min=-128, quant_max=127,
                        relu=False, unsigned=False):
    dtype = torch.float32 if output_scale is None else torch.uint8 if unsigned else torch.int8
    return acc.new_empty(acc.shape, dtype=dtype)


exact_readout = torch.ops.m2m_q.exact_readout.default

_LIB.define("quantize(Tensor value, float scale, int zero_point, int quant_min, int quant_max, bool unsigned=False) -> Tensor")


@torch.library.impl(_LIB, "quantize", "CompositeExplicitAutograd")
def _quantize(value, scale, zero_point, quant_min, quant_max, unsigned=False):
    from m2m.capture.exact_quant import quantize_tensor
    return quantize_tensor(value, scale, zero_point, 0, quant_min, quant_max, unsigned)


@torch.library.register_fake("m2m_q::quantize")
def _quantize_fake(value, scale, zero_point, quant_min, quant_max, unsigned=False):
    return value.new_empty(value.shape, dtype=torch.uint8 if unsigned else torch.int8)


quantize = torch.ops.m2m_q.quantize.default

_LIB.define("quantize_channel(Tensor value, Tensor scales, Tensor zero_points, int axis, int quant_min, int quant_max, bool unsigned=False) -> Tensor")


@torch.library.impl(_LIB, "quantize_channel", "CompositeExplicitAutograd")
def _quantize_channel(value, scales, zero_points, axis, quant_min, quant_max, unsigned=False):
    from m2m.capture.exact_quant import quantize_tensor
    return quantize_tensor(value, scales, zero_points, axis, quant_min, quant_max, unsigned)


@torch.library.register_fake("m2m_q::quantize_channel")
def _quantize_channel_fake(value, scales, zero_points, axis, quant_min, quant_max, unsigned=False):
    return value.new_empty(value.shape, dtype=torch.uint8 if unsigned else torch.int8)


quantize_channel = torch.ops.m2m_q.quantize_channel.default
