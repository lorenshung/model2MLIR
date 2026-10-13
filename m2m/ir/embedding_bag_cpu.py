"""CPU embedding-bag fast-path eligibility from tensor metadata only."""

import torch


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
