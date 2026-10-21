"""Exact arithmetic for quantized readout; eager Q/DQ is diagnostic only.

A scale denotes its stored binary value, not a rounded product of scales.
There is exactly one rounding at the final quantized code or float result.
"""
from fractions import Fraction
import math
import numbers
import struct


def rational(value):
    """Exact value of an integer of any width, a Fraction or a finite binary float."""
    if isinstance(value, numbers.Rational):
        return Fraction(value)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("exact quantization requires finite values")
    return Fraction(value)


def float_bits(value, width=32):
    """Correctly round a rational to IEEE binary32/64, without double rounding."""
    precision, bias, emin, emax = (24, 127, -126, 127) if width == 32 else (53, 1023, -1022, 1023)
    sign = int(value < 0) << (width - 1)
    value = abs(value)
    if not value:
        return sign
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Fraction(2) ** exponent:
        exponent -= 1
    unit = max(exponent, emin) - precision + 1
    mantissa = round(value / Fraction(2) ** unit)
    if mantissa == 1 << precision:
        mantissa >>= 1
        exponent += 1
    if exponent > emax:
        return sign | (((1 << (width - precision)) - 1) << (precision - 1))
    if mantissa < 1 << (precision - 1):
        return sign | mantissa
    return sign | ((max(exponent, emin) + bias) << (precision - 1)) | (mantissa - (1 << (precision - 1)))


def rounded_float(value, width=32):
    code = float_bits(value, width)
    return struct.unpack("<f" if width == 32 else "<d", code.to_bytes(width // 8, "little"))[0]


def validate_readout(activation_scale, weight_scale, alpha, beta, output_scale, zp, low, high, unsigned):
    if not weight_scale:
        raise ValueError("readout requires at least one scale")
    for scale in (activation_scale, *weight_scale):
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("readout scales must be finite and positive")
    if not math.isfinite(alpha) or alpha == 0:
        raise ValueError("readout alpha must be finite and nonzero")
    rational(beta)
    if output_scale is not None:
        if not math.isfinite(output_scale) or output_scale <= 0:
            raise ValueError("output scale must be finite and positive")
        bounds = (0, 255) if unsigned else (-128, 127)
        if not bounds[0] <= low < high <= bounds[1] or not bounds[0] <= zp <= bounds[1]:
            raise ValueError("invalid quantization range or zero point")


def exact_readout(acc, activation_scale, weight_scale, alpha=1.0, bias=None, beta=1.0,
                  output_scale=None, zero_point=0, quant_min=-128, quant_max=127,
                  relu=False, unsigned=False):
    import torch
    validate_readout(activation_scale, weight_scale, alpha, beta, output_scale,
                     zero_point, quant_min, quant_max, unsigned)
    if acc.dtype != torch.int32:
        raise TypeError("exact readout accumulator must be int32")
    if len(weight_scale) > 1 and (not acc.ndim or acc.shape[-1] != len(weight_scale)):
        raise ValueError("readout scales must match the last axis")
    scales = [rational(activation_scale) * rational(alpha) * rational(s) for s in weight_scale]
    bias_values = ([0.0] * acc.numel() if bias is None or beta == 0 else
                   torch.broadcast_to(bias, acc.shape).detach().cpu().reshape(-1).tolist())
    outputs = []
    for index, (integer, b) in enumerate(zip(acc.detach().cpu().reshape(-1).tolist(), bias_values)):
        value = integer * scales[index % len(scales)] + rational(b) * rational(beta)
        if output_scale is None:
            outputs.append(rounded_float(value))
        else:
            code = round(value / rational(output_scale)) + zero_point
            low = max(quant_min, zero_point) if relu else quant_min
            outputs.append(min(quant_max, max(low, code)))
    dtype = torch.float32 if output_scale is None else torch.uint8 if unsigned else torch.int8
    return torch.tensor(outputs, dtype=dtype, device=acc.device).reshape(acc.shape)


def quantize_tensor(value, scales, zero_points, axis, low, high, unsigned=False):
    import torch
    if not isinstance(scales, torch.Tensor):
        scales = torch.tensor(scales, dtype=torch.float64)
    if not isinstance(zero_points, torch.Tensor):
        zero_points = torch.tensor(zero_points, dtype=torch.int64)
    output = []
    scalar = scales.ndim == 0
    if not scalar:
        if value.ndim == 0:
            raise ValueError("per-channel quantization requires a tensor axis")
        axis %= value.ndim
        if scales.ndim != 1 or scales.numel() != value.shape[axis]:
            raise ValueError("per-channel scales must match the selected axis")
    if zero_points.shape != scales.shape:
        raise ValueError("scale and zero point shapes must match")
    flat = value.detach().cpu().reshape(-1).tolist()
    scale_list, zp_list = scales.reshape(-1).tolist(), zero_points.reshape(-1).tolist()
    for scale, zp in zip(scale_list, zp_list):
        validate_readout(1., [scale], 1., 1., 1., int(zp), low, high, unsigned)
    stride = 1 if scalar else math.prod(value.shape[axis + 1:])
    for index, v in enumerate(flat):
        channel = 0 if scalar else (index // stride) % value.shape[axis]
        if math.isnan(v):
            raise ValueError("exact quantize requires a non-NaN input")
        if math.isinf(v):
            code = high if v > 0 else low
        else:
            code = min(high, max(low, round(rational(v) / rational(scale_list[channel])) + int(zp_list[channel])))
        output.append(code)
    return torch.tensor(output, dtype=torch.uint8 if unsigned else torch.int8,
                        device=value.device).reshape(value.shape)
