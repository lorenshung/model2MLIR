"""CPU embedding-bag fast-path eligibility from tensor metadata only."""

import torch


def fbgemm_bfloat16_store(value, block):
    """FBGEMM's BF16 output conversion: add 0x8000, then discard low bits.

    Its halfway cases round away from zero, unlike arith.truncf's IEEE RNE.
    This is the native fast bag's storage boundary, not quantized readout.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, i16, i32, bf16
    bits = arith.BitcastOp(value, i32)
    half = arith.ConstantOp(IntegerAttr(0x8000, i32))
    added = arith.AddiOp(bits.result, half.result)
    shift = arith.ConstantOp(IntegerAttr(16, i32))
    high = arith.ShRUIOp(added.result, shift.result)
    narrow = arith.TruncIOp(high.result, i16)
    result = arith.BitcastOp(narrow.result, bf16)
    block.add_ops([bits, half, added, shift, high, narrow, result])
    return result.result


def fast_path_eligible(weight, per_sample_weights=None, *, padding_idx=-1):
    """Eligibility for the native contiguous newly allocated output overload.

    CPU index_select_add's fast path accepts Float/Half/BFloat16 sources
    with unit column stride and no padding. Its scaled counterpart also
    requires unit sample-weight stride. Indices and offsets are made
    contiguous by the native entry point and do not constrain this choice.
    The ordinary (non-out) overload allocates a unit-column-stride output.
    """
    if weight is None or weight.device.type != "cpu" or weight.ndim != 2:
        return False
    if weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if type(padding_idx) is not int or padding_idx >= 0 or type(weight.stride(1)) is not int:
        return False
    if weight.stride(1) != 1:
        return False
    return per_sample_weights is None or (
        per_sample_weights.ndim == 1 and type(per_sample_weights.stride(0)) is int and per_sample_weights.stride(0) == 1
    )
