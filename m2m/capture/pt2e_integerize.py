"""Turn supported PT2E Q/DQ contractions into integer contractions.

This operates on the converted PT2E ``GraphModule`` before capture.  The same
rewritten module must be used for the host golden: integer accumulation followed
by scaling is a different floating-point evaluation order from PT2E's portable
dequantize-then-matmul form.  Unsupported qparams remain unchanged and are
reported, so callers can enforce their own coverage gate.
"""

from __future__ import annotations

import math
import hashlib
from itertools import product
from typing import Any


def integerize_pt2e(
    model: Any,
    example_inputs: tuple[Any, ...],
    *,
    restore_result_layout: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Rewrite symmetric int8 contractions (linear, conv, addmm, matmul) in place.

    Each matched pair of dequantize nodes supplies the actual frozen activation
    and weight scales and integer tensors. Dynamic dimensions, nonzero zero
    points, non-int8 storage and unsupported graph shapes are left in portable
    PT2E form with an explicit refusal. Activation scales must be per tensor.
    Frozen per-output-channel weight scales are supported; a scale along the
    reduction dimension cannot be factored out.

    Covered contractions: ``aten.linear``, ``aten.conv{1,2,3}d`` and non-transposed
    ``aten.convolution`` (any groups, as one i8 GEMM per group), ``aten.addmm``
    (alpha and beta folded into the readout and bias) and ``aten.mm``/``bmm``/
    ``matmul``. Census kinds stay ``linear``/``conv2d``/``matmul``/``unsupported``;
    each decision and qparam receipt also names its ``source_op``.

    The readout is PT2E-portable: i32 accumulation, then the activation scale
    and the weight scale as two f32 multiplies, then the f32 bias (scaled by
    ``beta`` for addmm; ``alpha`` scales the readout).

    ``restore_result_layout`` materializes each convolution's source result
    layout (contiguous or channels-last) after the readout, instead of exposing
    the patch-ordered NHWC-backed view. Merlin's ``integerize_static_pt2e``
    performs the same restoration for ``aten.conv2d`` sources itself.
    """
    import torch
    from torch.fx import GraphModule, Node
    from torch.fx.passes.shape_prop import ShapeProp

    if not isinstance(model, GraphModule):
        raise TypeError("PT2E integerization requires a converted GraphModule")
    # Converted PT2E Q/DQ nodes commonly lack ``meta['val']`` even though their
    # adjacent linear node has it. Propagate the exact capture inputs once so
    # matching is based on observed shapes, never inferred from a name.
    ShapeProp(model).propagate(*example_inputs)

    dequant = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    channel_dequant = torch.ops.quantized_decomposed.dequantize_per_channel.default
    dequants = {dequant, channel_dequant}
    linear = torch.ops.aten.linear.default
    addmm = torch.ops.aten.addmm.default
    convolution = torch.ops.aten.convolution.default
    convs = {
        torch.ops.aten.conv1d.default: 1,
        torch.ops.aten.conv2d.default: 2,
        torch.ops.aten.conv3d.default: 3,
        convolution: None,
    }
    matmuls = {
        torch.ops.aten.matmul.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
    }
    contractions = {linear, addmm, *convs, *matmuls}
    refusals: list[dict[str, str]] = []
    precision_decisions: list[dict[str, str]] = []
    qparam_receipts: list[dict[str, Any]] = []
    quantized_by_kind = {
        kind: {"seen": 0, "integerized": 0, "remaining": 0}
        for kind in ("linear", "conv2d", "matmul", "unsupported")
    }
    counts = {
        "linear_seen": 0, "linear_integerized": 0,
        "conv2d_seen": 0, "conv2d_integerized": 0,
        "matmul_seen": 0, "matmul_integerized": 0,
        "quantized_contractions_seen": 0,
        "quantized_contractions_integerized": 0,
        "integer_mm_emitted": 0,
        "integer_bmm_emitted": 0,
        "max_reduction_k": 0,
    }
    source_ops: dict[str, dict[str, int]] = {}
    class _IncompatibleDequantPrecision(ValueError):
        def __init__(self, declared: Any, observed: Any) -> None:
            self.source_dtype = str(observed if observed is not None else declared)
            super().__init__(
                f"dequantize output is declared {declared} and observed {observed}; "
                "this i8xi8->i32 rewrite assumes float32 Q/DQ operands. Non-float32 "
                "dequantization changes per-operand rounding, so it needs an explicit "
                "precision conversion and a separate numerical qualification"
            )

    def _shape(node: Node) -> tuple[int, ...]:
        value = node.meta.get("tensor_meta") or node.meta.get("val")
        dims = getattr(value, "shape", None)
        if dims is None or not all(isinstance(dim, int) and dim > 0 for dim in dims):
            raise ValueError("static positive tensor shape is unavailable")
        return tuple(dims)

    def _frozen(node: Any, name: str) -> Any:
        if not isinstance(node, Node) or node.op != "get_attr":
            raise ValueError(f"per-channel {name} must be a frozen tensor attribute")
        value = model
        for component in str(node.target).split("."):
            value = getattr(value, component)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"per-channel {name} is not a frozen tensor")
        return value.detach()

    def _tensor_digest(value: Any) -> str:
        return hashlib.sha256(value.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()

    def _quant_operand(node: Any) -> tuple[Node, Any, int | None, dict[str, Any]]:
        if not isinstance(node, Node) or node.op != "call_function" or node.target not in dequants:
            raise ValueError("operand is not a supported PT2E dequantize")
        per_channel = node.target == channel_dequant
        if len(node.args) < (7 if per_channel else 6) or not isinstance(node.args[0], Node):
            raise ValueError("dequantize has no integer tensor or complete qparams")
        declared_output = node.kwargs.get("out_dtype", torch.float32) or torch.float32
        observed_output = getattr(node.meta.get("tensor_meta") or node.meta.get("val"), "dtype", None)
        if observed_output is None:
            raise ValueError("dequantize output dtype is unobserved")
        if declared_output != observed_output:
            raise ValueError(f"dequantize output dtype mismatch: declared {declared_output}, observed {observed_output}")
        if observed_output != torch.float32:
            raise _IncompatibleDequantPrecision(declared_output, observed_output)
        if per_channel:
            _, scale, zero_point, axis, qmin, qmax, dtype = node.args[:7]
        else:
            _, scale, zero_point, qmin, qmax, dtype = node.args[:6]
            axis = None
        if (
            not isinstance(qmin, int)
            or not isinstance(qmax, int)
            or qmin < -128
            or qmax > 127
            or qmin >= qmax
            or dtype != torch.int8
        ):
            raise ValueError("requires a valid signed int8 quantization range")
        storage_meta = node.args[0].meta.get("tensor_meta")
        if getattr(storage_meta, "dtype", None) != torch.int8:
            raise ValueError("dequantize source is not an observed int8 tensor")
        receipt: dict[str, Any] = {"granularity": "channel" if per_channel else "tensor",
                                   "quant_min": qmin, "quant_max": qmax, "zero_point": 0}
        if not per_channel:
            if (not isinstance(scale, (int, float)) or not math.isfinite(float(scale))
                    or float(scale) <= 0 or zero_point != 0):
                raise ValueError("requires finite positive scalar scale and zero point 0")
            receipt["scale"] = float(scale)
            return node.args[0], float(scale), None, receipt
        shape = _shape(node.args[0])
        if not isinstance(axis, int) or not 0 <= axis < len(shape):
            raise ValueError("per-channel axis must be a static nonnegative tensor dimension")
        scales = _frozen(scale, "scales")
        if (scales.dtype not in (torch.float32, torch.float64)
                or scales.ndim != 1 or scales.numel() != shape[axis]
                or not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all())):
            raise ValueError("per-channel scales must be finite positive frozen f32/f64 values matching the axis")
        if zero_point is not None:
            zeros = _frozen(zero_point, "zero points")
            if (zeros.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
                    or zeros.ndim != 1 or zeros.numel() != shape[axis] or bool((zeros != 0).any())):
                raise ValueError("per-channel weights require frozen integer zero point 0 for every channel")
            receipt["zero_points_sha256"] = _tensor_digest(zeros)
        receipt.update(axis=axis, channels=shape[axis], scale_dtype=str(scales.dtype),
                       scales_sha256=_tensor_digest(scales))
        return node.args[0], scale, axis, receipt

    def _ints(raw: Any, rank: int, name: str, *, minimum: int) -> tuple[int, ...]:
        values = (raw,) * rank if isinstance(raw, int) else tuple(raw) if isinstance(raw, (list, tuple)) else ()
        if len(values) == 1 and rank > 1:
            values = values * rank
        if len(values) != rank or not all(isinstance(v, int) and v >= minimum for v in values):
            raise ValueError(f"unsupported {name}: {raw!r}")
        return values

    def _arg(node: Node, index: int, name: str, default: Any) -> Any:
        return node.args[index] if len(node.args) > index else node.kwargs.get(name, default)

    graph = model.graph
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in contractions:
            continue
        source = str(node.target)
        operands = tuple(node.args[1:3] if node.target == addmm else node.args[:2])
        quantized = any(
            isinstance(arg, Node) and arg.op == "call_function" and arg.target in dequants
            for arg in operands
        )
        if quantized:
            counts["quantized_contractions_seen"] += 1
        if node.target == convolution and bool(_arg(node, 6, "transposed", False)):
            if quantized:
                quantized_by_kind["unsupported"]["seen"] += 1
                precision_decisions.append({"kind": "unsupported", "node": node.name, "source_op": source,
                                            "decision": "unresolved",
                                            "reason": "transposed convolution has no verified integer rewrite"})
                refusals.append({
                    "kind": source, "node": node.name,
                    "reason": "PT2E Q/DQ transposed convolution has no verified integer rewrite",
                })
            continue
        kind = "linear" if node.target == linear else "conv2d" if node.target in convs else "matmul"
        counts[f"{kind}_seen"] += 1
        if not quantized:
            # Plain floating-point islands were not selected by PT2E and do
            # not belong to the integer coverage claim.
            continue
        quantized_by_kind[kind]["seen"] += 1
        by_source = source_ops.setdefault(source, {"seen": 0, "integerized": 0})
        by_source["seen"] += 1
        decision = {"kind": kind, "node": node.name, "source_op": source, "decision": "unresolved"}
        precision_decisions.append(decision)
        alpha: float = 1
        beta: float = 1
        bias = None
        batch_shape: tuple[int, ...] = ()
        groups = 1
        try:
            if len(operands) < 2:
                raise ValueError(f"{kind} has fewer than two operands")
            activation, activation_scale, activation_axis, activation_qparams = _quant_operand(operands[0])
            weight, weight_scale, weight_axis, weight_qparams = _quant_operand(operands[1])
            if activation_axis is not None:
                raise ValueError("activation per-channel scaling has no verified integer layout; requires per-tensor activation")
            activation_shape = _shape(operands[0])
            weight_shape = _shape(operands[1])
            result_shape = _shape(node)
            if kind in ("linear", "conv2d"):
                bias = _arg(node, 2, "bias", None)
            elif node.target == addmm:
                bias = node.args[0]
                alpha, beta = node.kwargs.get("alpha", 1), node.kwargs.get("beta", 1)
                if (not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (alpha, beta))
                        or not math.isfinite(alpha) or not math.isfinite(beta)):
                    raise ValueError("addmm alpha and beta must be finite literals")
                if alpha == 0:
                    raise ValueError("addmm with alpha 0 has no contraction to integerize")
                if beta == 0:
                    bias = None  # beta == 0 ignores self, including NaN/inf.
            if kind == "linear":
                if (
                    len(activation_shape) < 2
                    or len(weight_shape) != 2
                    or activation_shape[-1] != weight_shape[-1]
                    or result_shape != (*activation_shape[:-1], weight_shape[0])
                ):
                    raise ValueError("linear has unsupported activation, weight or output geometry")
            elif kind == "conv2d":
                rank = convs[node.target] or len(activation_shape) - 2
                if rank not in (1, 2, 3) or {len(activation_shape), len(weight_shape), len(result_shape)} != {rank + 2}:
                    raise ValueError("convolution requires static N, C, spatial geometry of rank 3, 4 or 5")
                stride = _ints(_arg(node, 3, "stride", 1), rank, "stride", minimum=1)
                padding = _ints(_arg(node, 4, "padding", 0), rank, "padding", minimum=0)
                dilation = _ints(_arg(node, 5, "dilation", 1), rank, "dilation", minimum=1)
                groups = _arg(node, 8 if node.target == convolution else 6, "groups", 1)
                n, cin, *spatial = activation_shape
                cout, wcin, *kernel = weight_shape
                on, oc, *out_spatial = result_shape
                if not isinstance(groups, int) or groups < 1 or cin % groups or cout % groups or wcin * groups != cin:
                    raise ValueError(f"convolution channels {cin}->{cout} do not divide into {groups!r} groups")
                if groups > 256:
                    raise ValueError(f"grouped convolution would unroll {groups} integer GEMMs (>256)")
                expected = [
                    (spatial[i] + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1) // stride[i] + 1
                    for i in range(rank)
                ]
                if n != on or cout != oc or out_spatial != expected:
                    raise ValueError("convolution output disagrees with input, weight or window geometry")
            elif node.target == addmm:
                if (len(activation_shape) != 2 or len(weight_shape) != 2 or activation_shape[1] != weight_shape[0]
                        or result_shape != (activation_shape[0], weight_shape[1])):
                    raise ValueError("addmm has unsupported matrix geometry")
                if bias is not None:
                    bias_shape = _shape(bias) if isinstance(bias, Node) else ()
                    if tuple(torch.broadcast_shapes(bias_shape, result_shape)) != result_shape:
                        raise ValueError("addmm bias does not broadcast to the result")
            else:
                if (
                    len(activation_shape) != len(weight_shape)
                    or len(activation_shape) < 2
                    or activation_shape[:-2] != weight_shape[:-2]
                    or activation_shape[-1] != weight_shape[-2]
                    or result_shape != (*activation_shape[:-2], activation_shape[-2], weight_shape[-1])
                ):
                    raise ValueError("matmul has unsupported broadcasting or output geometry")
                batch_shape = activation_shape[:-2]
                batch_count = math.prod(batch_shape)
                if batch_count > 256:
                    raise ValueError(f"batched integer matmul would unroll {batch_count} instances (>256)")
            output_axis = len(weight_shape) - 1 if kind == "matmul" else 0
            if weight_axis is not None and weight_axis != output_axis:
                raise ValueError("weight channel axis must be the output dimension, not a reduction or batch dimension")
            reduction = activation_shape[-1] if kind != "conv2d" else wcin * math.prod(kernel)
            # Each product of signed int8 values is accumulated into i32.
            # Refuse shapes where the declared quant ranges could overflow it.
            # This check is conservative and independent of observed samples.
            if reduction * 128 * 128 > (1 << 31) - 1:
                raise ValueError(f"i32 accumulator may overflow for reduction K={reduction}")
            counts["max_reduction_k"] = max(counts["max_reduction_k"], reduction)
        except ValueError as exc:
            refusals.append({"kind": kind, "node": node.name, "reason": str(exc)})
            if isinstance(exc, _IncompatibleDequantPrecision):
                decision.update(decision="preserve_float_qdq", source_dtype=exc.source_dtype,
                                required_numeric_semantics="dequantize-before-floating-contraction")
            else:
                decision["reason"] = str(exc)
            continue

        preceding = set(graph.nodes)

        def call(target: Any, args: tuple = (), kwargs: dict | None = None) -> Node:
            return graph.call_function(target, args=args, kwargs=kwargs or {})

        with graph.inserting_before(node):
            pairs = []
            if kind == "linear":
                lhs = call(
                    torch.ops.aten.reshape.default,
                    args=(activation, [-1, activation_shape[-1]]),
                )
                rhs = call(torch.ops.aten.transpose.int, args=(weight, 0, 1))
                pairs.append((lhs, rhs))
            elif node.target == addmm:
                pairs.append((activation, weight))
            elif kind == "conv2d":
                # Every int8 value is exactly representable as f32, but no float
                # im2col is needed: zero-pad, slice and concatenate on int8 itself.
                # Each group is one GEMM with K ordered (taps..., channel).
                padded = activation
                if any(padding):
                    padded = call(
                        torch.ops.aten.constant_pad_nd.default,
                        args=(activation, [p for pad in reversed(padding) for p in (pad, pad)], 0),
                    )
                positions = n * math.prod(out_spatial)
                cin_group, cout_group = cin // groups, cout // groups
                to_channels_last = [0, *range(2, rank + 2), 1]
                for group in range(groups):
                    grouped = padded if groups == 1 else call(
                        torch.ops.aten.slice.Tensor,
                        args=(padded, 1, group * cin_group, (group + 1) * cin_group, 1),
                    )
                    patches = []
                    for tap in product(*(range(extent) for extent in kernel)):
                        patch = grouped
                        for axis, offset in enumerate(tap):
                            start = offset * dilation[axis]
                            patch = call(
                                torch.ops.aten.slice.Tensor,
                                args=(patch, axis + 2, start, start + out_spatial[axis] * stride[axis], stride[axis]),
                            )
                        patch = call(torch.ops.aten.permute.default, args=(patch, to_channels_last))
                        patches.append(call(
                            torch.ops.aten.reshape.default,
                            args=(patch, [positions, cin_group]),
                        ))
                    lhs = patches[0] if len(patches) == 1 else call(
                        torch.ops.aten.cat.default, args=(patches, 1)
                    )
                    rhs = weight if groups == 1 else call(
                        torch.ops.aten.slice.Tensor,
                        args=(weight, 0, group * cout_group, (group + 1) * cout_group, 1),
                    )
                    rhs = call(torch.ops.aten.permute.default, args=(rhs, to_channels_last))
                    rhs = call(
                        torch.ops.aten.reshape.default,
                        args=(rhs, [cout_group, cin_group * math.prod(kernel)]),
                    )
                    rhs = call(torch.ops.aten.transpose.int, args=(rhs, 0, 1))
                    pairs.append((lhs, rhs))
            batched_readout = kind == "matmul" and len(batch_shape) == 1
            if batched_readout:
                # CPU has no i8/i8 -> i32 bmm overload. Exact sign extensions
                # followed by an ordinary i32 bmm retain the batch axes; the
                # importer can fold these proven extensions into the MAC body.
                lhs = call(torch.ops.aten._to_copy.default,
                                          args=(activation,), kwargs={"dtype": torch.int32})
                rhs = call(torch.ops.aten._to_copy.default,
                                          args=(weight,), kwargs={"dtype": torch.int32})
                accumulator = call(torch.ops.aten.bmm.default, args=(lhs, rhs))
                counts["integer_bmm_emitted"] += 1
            elif kind == "matmul" and node.target != addmm:
                matrices = []
                for batch_indices in product(*(range(extent) for extent in batch_shape)):
                    lhs = activation
                    rhs = weight
                    for index in batch_indices:
                        lhs = call(torch.ops.aten.select.int, args=(lhs, 0, index))
                        rhs = call(torch.ops.aten.select.int, args=(rhs, 0, index))
                    matrices.append(call(torch.ops.aten._int_mm.default, args=(lhs, rhs)))
                accumulator = matrices[0] if not batch_shape else call(
                    torch.ops.aten.reshape.default,
                    args=(call(torch.ops.aten.stack.default, args=(matrices, 0)), list(result_shape)),
                )
                counts["integer_mm_emitted"] += len(matrices)
            else:
                accumulators = [call(torch.ops.aten._int_mm.default, args=(lhs, rhs)) for lhs, rhs in pairs]
                accumulator = accumulators[0] if len(accumulators) == 1 else call(
                    torch.ops.aten.cat.default, args=(accumulators, 1)
                )
                counts["integer_mm_emitted"] += len(accumulators)
            cast_input = accumulator
            if batched_readout:
                # Only host readout is flattened: i32 -> f32 and the two
                # separately rounded scale multiplies keep row-major order.
                cast_input = call(torch.ops.aten.reshape.default,
                    args=(accumulator, [math.prod(result_shape[:-1]), result_shape[-1]]))
            value = call(
                torch.ops.aten._to_copy.default,
                args=(cast_input,),
                kwargs={"dtype": torch.float32},
            )
            value = call(torch.ops.aten.mul.Tensor, args=(value, activation_scale))
            value = call(torch.ops.aten.mul.Tensor, args=(value, weight_scale))
            if weight_axis is not None:
                # Keep the actual scale tensor and its dtype. PT2E's per-channel
                # dequantize explicitly returns f32 even when its scales are f64.
                value = call(torch.ops.aten._to_copy.default, args=(value,),
                                            kwargs={"dtype": torch.float32})
            if batched_readout:
                value = call(torch.ops.aten.reshape.default, args=(value, list(result_shape)))
            if alpha != 1:
                value = call(torch.ops.aten.mul.Tensor, args=(value, alpha))
            if kind == "linear":
                value = call(torch.ops.aten.reshape.default, args=(value, list(result_shape)))
            elif kind == "conv2d":
                value = call(torch.ops.aten.reshape.default, args=(value, [n, *out_spatial, cout]))
                value = call(torch.ops.aten.permute.default,
                                            args=(value, [0, rank + 1, *range(1, rank + 1)]))
            if bias is not None:
                if kind == "conv2d":
                    bias = call(torch.ops.aten.reshape.default, args=(bias, [1, cout, *([1] * rank)]))
                elif beta != 1:
                    bias = call(torch.ops.aten.mul.Tensor, args=(bias, beta))
                value = call(torch.ops.aten.add.Tensor, args=(value, bias))
            if restore_result_layout and kind == "conv2d":
                source_value = node.meta.get("val")
                formats = [torch.contiguous_format, torch.channels_last, torch.channels_last_3d]
                layout = next((fmt for fmt in formats if isinstance(source_value, torch.Tensor)
                               and (rank + 2 == 4 or fmt is not torch.channels_last)
                               and (rank + 2 == 5 or fmt is not torch.channels_last_3d)
                               and source_value.is_contiguous(memory_format=fmt)), None)
                if layout is not None:
                    value = call(torch.ops.aten.clone.default, args=(value,),
                                 kwargs={"memory_format": layout})
        value.meta["pt2e_integerized_from"] = node.name
        replaced = [node]
        # Every generated operation has an exact rewritten source consumer.
        # Preserve its carried source identities through the subsequent export;
        # names remain a human-readable debug label, never identity evidence.
        origins = set()
        for source_node in [*replaced, *operands]:
            if isinstance(source_node, Node):
                origins.update((source_node.meta.get("custom") or {}).get("m2m_lineage") or ())
        if origins:
            for created in graph.nodes:
                if created not in preceding:
                    metadata = dict(created.meta.get("custom") or {})
                    metadata["m2m_lineage"] = sorted(set([*metadata.get("m2m_lineage", ()), *origins]))
                    metadata["m2m_transform"] = "pt2e_integerization"
                    created.meta["custom"] = metadata
        replaced[0].replace_all_uses_with(value)
        for stale in replaced:
            graph.erase_node(stale)
        counts[f"{kind}_integerized"] += 1
        counts["quantized_contractions_integerized"] += 1
        quantized_by_kind[kind]["integerized"] += 1
        by_source["integerized"] += 1
        decision.update(decision="integerized_i32", source_dtype="torch.float32",
                        required_numeric_semantics="integer-accumulate-then-scale")
        qparam_receipts.append({"kind": kind, "source_op": source, "activation": activation_qparams,
                               "weight": weight_qparams, "reduction_k": reduction,
                               "accumulator_dtype": "torch.int32", "output_dtype": "torch.float32",
                               "groups": groups,
                               "bias_application": "post_scale_f32" if bias is not None else "none",
                               **({"alpha": alpha, "beta": beta} if node.target == addmm else {})})

    if counts["integer_mm_emitted"] or counts["integer_bmm_emitted"]:
        graph.eliminate_dead_code()
        graph.lint()
        model.recompile()
    remaining_dequant = sum(
        1 for node in graph.nodes if node.op == "call_function" and node.target in dequants
    )
    for row in quantized_by_kind.values():
        row["remaining"] = row["seen"] - row["integerized"]
    decision_counts = {label: sum(row["decision"] == label for row in precision_decisions)
                       for label in ("integerized_i32", "preserve_float_qdq", "unresolved")}
    return model, {
        "schema": "m2m.pt2e-integerize.v1",
        **counts,
        "quantized_contractions_remaining": (
            counts["quantized_contractions_seen"] - counts["quantized_contractions_integerized"]
        ),
        "remaining_dequant_count": remaining_dequant,
        "quantized_by_kind": quantized_by_kind,
        "accumulator_bound_checked": True,
        "qparams": qparam_receipts,
        "refusals": refusals,
        # Numeric decisions over every PT2E-selected contraction. A target must
        # separately admit each lane and implement any host/device partition;
        # this frontend receipt is never a device-placement certificate.
        "precision_decisions": precision_decisions,
        "precision_decision_counts": decision_counts,
        "source_ops": source_ops,
    }

