"""Torch-MLIR gap-op decompositions.

torch-mlir lowers hundreds of aten ops to linalg, but a handful have no converter
and torch-mlir marks them *illegal*, aborting the whole module. Several of these are
not even in the original graph -- torch's own decompositions introduce them (e.g.
``aten.empty_permuted``). The fix (not a monkeypatch): provide torch-level
decompositions that rewrite each gap op into ops torch-mlir *does* support, and feed
them into ``run_decompositions`` so the ExportedProgram handed to torch-mlir is free
of gap ops.

This is the growable registry: when torch-mlir reports a new "failed to legalize
operation 'torch.aten.<op>'", add a decomposition here.
"""

from __future__ import annotations

from typing import Any

import torch

_aten = torch.ops.aten


def _upstream_decompositions(*ops):
    """Return available registered PyTorch decompositions for explicit ops.

    ``torch.export.default_decompositions()`` intentionally preserves many
    Core ATen operators.  For the small set below, PyTorch's broader pinned
    decomposition registry rewrites the op entirely into payload primitives
    that model2MLIR already lowers.  Looking up each overload explicitly keeps
    this opt-in and auditable instead of enabling the whole private registry.
    """

    from torch._decomp import decomposition_table

    return {op: decomposition_table[op] for op in ops if op in decomposition_table}


def _log_softmax_cpu(x, dim, half_to_float=False):
    """CPU last-dimension log-softmax retains its scalar scratch storage dtype."""
    from torch._decomp import decomposition_table

    if (
        x.device.type != "cpu"
        or x.dtype not in (torch.float16, torch.bfloat16)
        or x.ndim == 0
        or dim % x.ndim != x.ndim - 1
        or half_to_float
    ):
        return decomposition_table[_aten._log_softmax.default](x, dim, half_to_float)
    dtype = x.dtype
    wide = x.float()
    maximum = wide.amax(dim, keepdim=True)
    shifted = wide - maximum
    # ATen's lastdim kernel stores the reduction and then log in scalar_t
    # scratch arrays. Those two storage observations precede the final map.
    summed = shifted.exp().sum(dim, keepdim=True).to(dtype).float()
    logsum = summed.log().to(dtype).float()
    return (shifted - logsum).to(dtype)


def _batch_norm_no_stats_cpu(input, weight, bias, training, momentum, eps):
    """CPU saved-stat storage, zero-variance handling and affine evaluation."""
    from torch._decomp import decomposition_table

    if input.device.type != "cpu" or not training:
        return decomposition_table[_aten._native_batch_norm_legit.no_stats](
            input, weight, bias, training, momentum, eps
        )
    dtype = input.dtype
    compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    dims = [0, *range(2, input.ndim)]
    contiguous = (
        input.is_contiguous()
        or (input.ndim == 4 and input.is_contiguous(memory_format=torch.channels_last))
        or (input.ndim == 5 and input.is_contiguous(memory_format=torch.channels_last_3d))
    )
    wide = input.to(compute_dtype)
    mean = wide.mean(dims, keepdim=True)
    saved_mean = mean.to(dtype)
    variance_input = wide if contiguous or dtype != torch.float32 else wide.double()
    center = mean if contiguous else saved_mean.to(variance_input.dtype)
    variance = ((variance_input - center) ** 2).mean(dims, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    if eps == 0:
        rstd = torch.where(variance == 0, 0.0, rstd)
    saved_rstd = rstd.to(dtype)
    mean, rstd = saved_mean.to(compute_dtype), saved_rstd.to(compute_dtype)
    broadcast = [1, input.shape[1], *([1] * (input.ndim - 2))]
    w = weight.to(compute_dtype).reshape(broadcast) if weight is not None else 1.0
    b = bias.to(compute_dtype).reshape(broadcast) if bias is not None else 0.0
    affine_contiguous = contiguous and all(t is None or t.is_contiguous() for t in (weight, bias))
    if affine_contiguous:
        alpha = rstd * w
        output = wide * alpha + (b - mean * alpha)
    else:
        output = ((wide - mean) * rstd) * w + b
    return output.to(dtype), saved_mean.reshape(-1), saved_rstd.reshape(-1)


def _empty_permuted(size, physical_layout, *, dtype=None, layout=None, device=None, pin_memory=None):
    """aten.empty_permuted -> aten.empty.

    Contents are uninitialized, so the physical (permuted) layout is irrelevant to the
    computation; the logical shape ``size`` is preserved. torch-mlir has no
    empty_permuted converter but lowers empty fine.
    """
    return torch.empty(
        size,
        dtype=dtype,
        device=device,
        pin_memory=bool(pin_memory) if pin_memory is not None else False,
    )


def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """aten.scaled_dot_product_attention -> the math reference, so export expands it into
    matmul/softmax/(mask) that m2m lowers. Quantized graphs retain SDPA where fp32 does
    not; expanding it here covers both."""
    import math

    if dropout_p != 0.0:
        raise NotImplementedError("SDPA decomposition does not support nonzero dropout_p")
    if enable_gqa:
        raise NotImplementedError("SDPA decomposition does not support enable_gqa=True")
    if is_causal and attn_mask is not None:
        raise ValueError("SDPA cannot combine is_causal=True with an explicit attn_mask")
    output_dtype = query.dtype
    if attn_mask is not None and attn_mask.dtype not in (torch.bool, torch.float32, output_dtype):
        raise TypeError("SDPA mask must be boolean, f32, or match query dtype")
    if output_dtype in (torch.float16, torch.bfloat16):
        query, key, value = query.float(), key.float(), value.float()
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.float()
    L, S = query.size(-2), key.size(-2)
    scale_factor = (1.0 / math.sqrt(query.size(-1))) if scale is None else scale
    attn = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    if is_causal:
        mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn = attn.masked_fill(~mask, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn = attn.masked_fill(~attn_mask, float("-inf"))
        else:
            attn = attn + attn_mask
    # SDPA defines an all-negative-infinity score row as zero probabilities.
    # Ordinary softmax alone yields NaNs; do not clear genuine NaN/+inf rows.
    empty = torch.amax(attn, dim=-1, keepdim=True) == float("-inf")
    attn = torch.softmax(attn, dim=-1)
    attn = torch.where(empty, 0.0, attn)
    return torch.matmul(attn, value).to(output_dtype)


def _dropout(input, p=0.5, train=False):
    """aten.dropout in inference (train=False) is the identity."""
    return input


def _chunk(input, chunks, dim=0):
    """aten.chunk -> tensor_split (slices); export lowers the slices to ops m2m handles."""
    return torch.tensor_split(input, chunks, dim=dim)


def inline_set_grad_hops(gm) -> bool:
    """Flatten ALL ``wrap_with_*`` higher-order ops (torch.no_grad / torch.autocast regions,
    arbitrarily nested) by recursively rebuilding the FX graph.

    These HOPs wrap a sub-GraphModule -- ``wrap_with_set_grad_enabled(enabled, submod, *ops)``
    / ``wrap_with_autocast(device, dtype, enabled, cache, submod, *ops)`` -- with
    ``getitem(hop, i)`` extracting outputs. The FXImporter can't recurse into them, so their
    bodies (e.g. RoPE precompute) never lower. Rather than fragile in-place surgery (which
    can't converge on nested HOPs), we copy the whole graph into a fresh one, splicing each
    HOP's subgraph inline (recursively) with its placeholders bound to the HOP's operands.
    get_attr'd attributes (tensor constants + nested submodules) are transferred to ``gm``.
    Returns True if any HOP was flattened. One call fully flattens (recursion handles nesting).
    """
    import operator

    import torch.fx as fx

    if not any(n.op == "call_function" and "wrap_with_" in str(n.target) for n in gm.graph.nodes):
        return False

    new_g = fx.Graph()

    def _same_tensor_value(left, right) -> bool:
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return False
        return (
            left.dtype == right.dtype
            and tuple(map(str, left.shape)) == tuple(map(str, right.shape))
            and left.device == right.device
            and left.layout == right.layout
            and (left.layout != torch.strided or tuple(map(str, left.stride())) == tuple(map(str, right.stride())))
        )

    def _carry_inlined_origin(source, result, *, value=None) -> bool:
        """Record an actual, exactly typed FX value substitution for trace mode."""
        source_value = source.meta.get("val") if value is None else value
        if not isinstance(result, fx.Node) or not _same_tensor_value(source_value, result.meta.get("val")):
            return False
        source_id = (source.meta.get("custom") or {}).get("m2m_node_id")
        if not source_id:
            return False
        carriers = [result]
        # A default decomposition may erase a same-dtype, no-copy `to`. Its
        # forward tensor value is the input's value, so bind the substitution
        # to that exact typed input too; do not infer this for a real cast.
        alias_proven = (
            result.op == "call_function"
            and str(result.target) == "aten.to.dtype"
            and len(result.args) == 2
            and isinstance(result.args[0], fx.Node)
            and result.args[1] == getattr(result.args[0].meta.get("val"), "dtype", None)
            and set(result.kwargs) <= {"non_blocking", "copy", "memory_format"}
            and result.kwargs.get("non_blocking", False) is False
            and result.kwargs.get("copy", False) is False
            and result.kwargs.get("memory_format") is None
            and _same_tensor_value(result.meta.get("val"), result.args[0].meta.get("val"))
        )
        if alias_proven:
            carriers.append(result.args[0])
        for carrier in carriers:
            custom = dict(carrier.meta.get("custom") or {})
            lineage = list(custom.get("m2m_lineage") or ())
            if source_id not in lineage:
                lineage.append(source_id)
            custom["m2m_lineage"] = lineage
            carrier.meta["custom"] = custom
        return True

    def _submod_idx(owner, n):
        return next(
            (
                k
                for k, a in enumerate(n.args)
                if isinstance(a, fx.Node) and a.op == "get_attr" and hasattr(getattr(owner, a.target, None), "graph")
            ),
            None,
        )

    def emit(owner, node_list, local_env):
        """Copy a (sub)graph's nodes into new_g using local_env (sub node -> new value).
        A HOP node maps to a LIST of its output values; getitem on it indexes that list;
        a direct single-output use is unwrapped in _remap. Returns the output value list."""

        def _remap(a):
            v = local_env[a]
            return v[0] if isinstance(v, list) and len(v) == 1 else v

        outputs = None
        for sn in node_list:
            if sn.op == "placeholder":
                continue  # bound by caller in local_env
            if sn.op == "output":
                o = sn.args[0]
                outputs = [
                    _remap(x) if isinstance(x, fx.Node) else x for x in (o if isinstance(o, (tuple, list)) else [o])
                ]
                continue
            if sn.op == "get_attr":
                if not hasattr(gm, sn.target):
                    try:
                        obj = getattr(owner, sn.target)
                        gm.add_module(sn.target, obj) if isinstance(obj, torch.nn.Module) else setattr(
                            gm, sn.target, obj
                        )
                    except Exception:  # noqa: BLE001
                        pass
                ga = new_g.get_attr(sn.target)
                ga.meta = dict(sn.meta)
                local_env[sn] = ga
                continue
            if sn.op == "call_function":
                # getitem on a HOP's output list
                if (
                    sn.target is operator.getitem
                    and isinstance(sn.args[0], fx.Node)
                    and isinstance(local_env.get(sn.args[0]), list)
                ):
                    selected = local_env[sn.args[0]][sn.args[1]]
                    _carry_inlined_origin(sn, selected)
                    local_env[sn] = selected
                    continue
                if "wrap_with_" in str(sn.target):
                    si = _submod_idx(owner, sn)
                    if si is not None:
                        sub = getattr(owner, sn.args[si].target)
                        operands = [_remap(a) for a in sn.args[si + 1 :] if isinstance(a, fx.Node)]
                        phs = [p for p in sub.graph.nodes if p.op == "placeholder"]
                        child = dict(zip(phs, operands))
                        outputs = emit(sub, list(sub.graph.nodes), child) or []
                        values = sn.meta.get("val")
                        values = list(values) if isinstance(values, (tuple, list)) else [values]
                        if len(values) == len(outputs):
                            for value, output in zip(values, outputs, strict=True):
                                if isinstance(output, fx.Node):
                                    _carry_inlined_origin(sn, output, value=value)
                        if not outputs and all(n.op in {"placeholder", "output", "get_attr"} for n in sub.graph.nodes):
                            lineage = list((sn.meta.get("custom") or {}).get("m2m_lineage") or ())
                            if lineage:
                                gm._m2m_trace_eliminations = [
                                    *getattr(gm, "_m2m_trace_eliminations", ()),
                                    {
                                        "source_ids": lineage,
                                        "kind": "eliminated",
                                        "reason": "inlined empty higher-order region with no call nodes",
                                        "proof": {
                                            "submodule": sn.args[si].target,
                                            "body_nodes": len(list(sub.graph.nodes)),
                                        },
                                    },
                                ]
                        local_env[sn] = outputs
                        continue
            # ordinary node (or unhandled HOP): copy with remapped args
            new_node = new_g.node_copy(sn, _remap)
            local_env[sn] = new_node
        return outputs

    top_env = {}
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            p = new_g.placeholder(n.name)
            p.meta = dict(n.meta)
            top_env[n] = p
    outs = emit(gm, [n for n in gm.graph.nodes if n.op != "placeholder"], top_env)
    new_g.output(tuple(outs) if outs else ())
    new_g.lint()
    gm.graph = new_g
    gm.recompile()
    return True


# op-overload -> decomposition function. Extend as new torch-mlir gap ops surface.
from m2m.ir.grid_sample import grid_sampler_2d_cpu

_GAP_DECOMPS: dict[Any, Any] = {
    _aten.empty_permuted.default: _empty_permuted,
    _aten.scaled_dot_product_attention.default: _sdpa,
    _aten.dropout.default: _dropout,
    _aten.chunk.default: _chunk,
    _aten._log_softmax.default: _log_softmax_cpu,
    _aten.grid_sampler_2d.default: grid_sampler_2d_cpu,
    _aten._native_batch_norm_legit.no_stats: _batch_norm_no_stats_cpu,
}
_GAP_DECOMPS.update(
    # Core ATen gaps whose official PyTorch decompositions close entirely over
    # model2MLIR's supported payload set in the pinned environment.
    _upstream_decompositions(
        _aten.hardtanh.default,
        _aten.index_select.default,
        _aten.isinf.default,
        _aten.leaky_relu.default,
        _aten.col2im.default,
        _aten.diagonal.default,
        _aten.embedding_dense_backward.default,
        _aten.max_pool2d_with_indices.default,
        _aten.max_pool2d_with_indices_backward.default,
        _aten.max_pool3d_with_indices.default,
        _aten.native_group_norm.default,
        _aten.resize_.default,
    )
)


def torch_mlir_gap_decompositions() -> dict[Any, Any]:
    """Decompositions to merge into the export decomposition table so torch-mlir's
    pipeline does not abort on unsupported ops."""
    return dict(_GAP_DECOMPS)
