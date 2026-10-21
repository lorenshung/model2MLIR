"""CPU LayerNorm backward parameter arithmetic and physical statistic reads."""

from math import prod

from xdsl.dialects import tensor
from xdsl.dialects.builtin import TensorType


def contiguous_storage_tensor(source, value):
    """Read the captured in-bounds scalar pointer span into a logical tensor."""
    shape = list(source.type.get_shape())
    count = prod(shape)
    capacity = int(value.untyped_storage().nbytes() // value.element_size())
    offset = int(value.storage_offset())
    if offset < 0 or offset + count > capacity:
        raise ValueError(
            f"native statistics pointer span {offset}+{count} exceeds captured backing storage {capacity}; "
            f"shape={value.shape}, stride={value.stride()}"
        )
    from .storage_resize import read_storage
    return read_storage(source, shape, count, count)


def weight_gradient(grad, source, mean, rstd, result_type):
    """Native single-thread dgamma: affine FMAs and a storage-width row buffer.

    layer_norm_backward_frame forms b = -rstd * mean, then accumulates
    dgamma = fma(dy, fma(rstd, x, b), dgamma). Centering x first changes both
    finite rounding and non-finite propagation. Half/bfloat16 buffers round
    after every row, even when the affine parameters themselves are float.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import Float64Type, f32
    from xdsl.dialects.math import FmaOp

    from .builder import Builder
    from .structural import coordinates, generate

    shape = list(source.type.get_shape())
    normalized = list(result_type.get_shape())
    if any(n < 0 for n in [*shape, *normalized, *mean.type.get_shape(), *rstd.type.get_shape()]):
        raise ValueError("native LayerNorm weight gradient requires static shapes")
    outer = shape[:len(shape) - len(normalized)]
    storage_elem = source.type.element_type
    computation_elem = storage_elem if isinstance(storage_elem, Float64Type) else f32

    def build(block, column):
        b = Builder(block, computation_elem)

        def load(bb, value, indices):
            scalar = bb.put(tensor.ExtractOp(value, indices, value.type.element_type))
            return bb.cast(scalar, computation_elem)

        def row(bb, i, total):
            coords = [*coordinates(bb.block, i, outer), *column]
            x, dy = load(bb, source, coords), load(bb, grad, coords)
            a = load(bb, rstd, coordinates(bb.block, i, list(rstd.type.get_shape())))
            m = load(bb, mean, coordinates(bb.block, i, list(mean.type.get_shape())))
            shift = bb.mul(bb.put(arith.NegfOp(a)), m)
            affine = bb.put(FmaOp(a, x, shift))
            total = bb.put(FmaOp(dy, affine, bb.cast(total, computation_elem)))
            return [bb.cast(total, storage_elem)]

        total, = b.loop(0, prod(outer), 1, [b.cast(b.fp(0), storage_elem)], row)
        # Native's second pass adds the thread buffer to an opmath +0 seed.
        total = b.add(b.fp(0), b.cast(total, computation_elem))
        return b.cast(total, result_type.element_type)

    return generate(normalized, result_type.element_type, build)


def layer_norm_backward(operands, meta, node_name):
    import torch
    from torch._decomp import decomposition_table

    from .core_aten import upstream_lowering
    from .decompositions import DecompResult, _fx_arg

    rewritten = list(operands)
    prefix = []
    # Tensor operands 2/3 are schema arguments 3/4 (normalized_shape is static).
    for operand, argument in ((2, 3), (3, 4)):
        value = _fx_arg(meta, argument).meta["val"]
        if not value.is_contiguous():
            emitted, rewritten[operand] = contiguous_storage_tensor(rewritten[operand], value)
            prefix += emitted
    lower = upstream_lowering(decomposition_table[torch.ops.aten.native_layer_norm_backward.default])
    values = meta["val"]
    if isinstance(values[1], torch.Tensor):
        # Keep the input/bias gradients independent of dgamma's affine form.
        # Suppress the upstream centered dgamma instead of emitting it unused.
        remaining = dict(meta, val=tuple(None if i == 1 else v for i, v in enumerate(values)))
        mask = list(_fx_arg(meta, 7))
        mask[1] = False
        args = list(meta["_fx_args"])
        if len(args) > 7:
            args[7] = mask
            remaining["_fx_args"] = tuple(args)
        else:
            remaining["_fx_kwargs"] = dict(meta.get("_fx_kwargs", {}), output_mask=mask)
        if any(isinstance(v, torch.Tensor) for v in remaining["val"]):
            result = lower(rewritten, remaining, node_name)
        else:
            result = DecompResult(results=[None, None, None], pattern_hint="composite")
        from .decompositions import _element_type_from_meta

        result_type = TensorType(_element_type_from_meta({"val": values[1]}), list(values[1].shape))
        ops, gradient = weight_gradient(*rewritten[:4], result_type)
        result.ops += ops
        result.results[1] = gradient
        result.result = next(v for v in result.results if v is not None)
    else:
        result = lower(rewritten, meta, node_name)
    result.ops = [*prefix, *result.ops]
    return result
