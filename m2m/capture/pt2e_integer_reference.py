"""Independent integer-contraction reference for TorchAO PT2E Q/DQ graphs.

PT2E's portable converted form expresses quantized Conv2d/Linear as dequantize ->
floating-point contraction.  Integer targets instead accumulate the frozen qvalues in
i32 and apply the two scales after the reduction.  Those expressions are algebraically
equivalent over the reals but not bit-identical after a long floating-point network.

This executor keeps every non-contraction operation in Torch and replaces only a
Conv2d/Linear whose two operands are float32 PT2E dequantize operations. Other
dequantize precisions retain their source floating contraction and are not counted
as integer work. It therefore provides a framework-side integer reference without
consuming compiler IR, target code, or target output. Unsupported integer layouts
fail closed rather than silently falling back while claiming integer coverage.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class IntegerReferenceResult:
    """Reference outputs and in-memory, exact-operand reduction diagnostics.

    ``partial_sum_evidence`` describes the recorded mathematical K order, not a
    hardware accumulation order or a serialized bundle certificate. The
    storage-wide centered-operand bound remains the order-independent i32 safety gate.
    """

    output: Any
    conv2d_count: int
    linear_count: int
    matmul_count: int = 0
    partial_sum_evidence: tuple[dict[str, Any], ...] = ()

    @property
    def contraction_count(self) -> int:
        return self.conv2d_count + self.linear_count + self.matmul_count


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    values = tuple(int(v) for v in value)
    if len(values) != 2:
        raise ValueError(f"expected a 2-D convolution parameter, got {value!r}")
    return values


def _operand_identity(value: Any) -> dict[str, Any]:
    """Bind a diagnostic to exact centered byte-operand values and shape."""
    import torch

    if value.numel() < 1 or bool(((value < -255) | (value > 255)).any()):
        raise ValueError("partial-sum evidence requires nonempty centered byte operands")
    wide = bool(((value < -128) | (value > 127)).any())
    raw = value.detach().to(device="cpu", dtype=torch.int32 if wide else torch.int8).contiguous().numpy().tobytes()
    return {"dtype": "i32" if wide else "i8", "shape": [int(dim) for dim in value.shape],
            "sha256": hashlib.sha256(raw).hexdigest()}


#: Every intermediate of a float64 sum of integer products is exact while the sum of the
#: products' magnitudes stays below 2**53. The block bound below asks for 2**49 instead, a margin
#: that also covers a BLAS that recombines operands before multiplying.
_EXACT_FLOAT64_BOUND = 1 << 49
#: K extent of one block: an exact boundary prefix per block, an exact scan only inside a block
#: that could hold a larger prefix than one already found.
_PREFIX_BLOCK = 128
#: Upper limit on one float64 [blocks, M, N] tile of block sums (elements).
_TILE_ELEMENTS = 1 << 22
#: Candidate blocks scanned exactly at once.
_SCAN_ROWS = 1 << 15
#: Below this many output positions (one decode token) the direct scan is the cheaper exact form.
_DIRECT_ROWS = 4


def _max_partial_sum_tiled(lhs: Any, rhs: Any) -> int:
    """The direct int64 scan of every prefix: the reference :func:`_max_partial_sum` reproduces."""
    import torch

    maximum = 0
    for row in range(0, lhs.shape[0], 64):
        left = lhs[row : row + 64]
        for col in range(0, rhs.shape[0], 32):
            right = rhs[col : col + 32]
            carry = torch.zeros((left.shape[0], right.shape[0]), dtype=torch.int64)
            for start in range(0, left.shape[1], 128):
                products = (
                    left[:, None, start : start + 128]
                    * right[None, :, start : start + 128]
                )
                prefixes = products.cumsum(dim=-1) + carry[..., None]
                maximum = max(maximum, int(prefixes.abs().amax()))
                carry = prefixes[..., -1]
    return maximum


def _max_partial_sum(lhs: Any, rhs: Any) -> int:
    """Exact maximum over every prefix of every output reduction, initial sum zero.

    Inputs are rank-2 [output positions, K] and [output channels, K]. The result is the
    maximum of ``|sum(lhs[m, :k] * rhs[n, :k])|`` over every m, n and k, computed without
    visiting every prefix. K is cut into blocks. Block sums, and block sums of magnitudes,
    are float64 matrix products, exact because every partial sum is an integer below
    :data:`_EXACT_FLOAT64_BOUND`. Their running sums are exact prefixes at block ends, so
    their largest magnitude is a lower bound on the answer. Inside a block, no prefix can
    exceed ``|prefix before the block| + block sum of magnitudes``; only blocks whose bound
    is above the best prefix found so far are scanned exactly in int64. Operands whose
    magnitudes could break float64 exactness, and fewer than :data:`_DIRECT_ROWS` output
    positions, use the direct int64 scan.
    """
    import torch

    if (
        lhs.ndim != 2
        or rhs.ndim != 2
        or lhs.shape[1] != rhs.shape[1]
        or min(*lhs.shape, *rhs.shape) < 1
    ):
        raise ValueError("partial-sum evidence has invalid contraction geometry")
    lhs, rhs = (
        lhs.to(device="cpu", dtype=torch.int64),
        rhs.to(device="cpu", dtype=torch.int64),
    )
    rows, depth = lhs.shape
    columns = rhs.shape[0]

    def magnitude(value: Any) -> int:
        return max(abs(int(value.min())), abs(int(value.max())))

    if (
        rows < _DIRECT_ROWS
        or magnitude(lhs) * magnitude(rhs) * depth >= _EXACT_FLOAT64_BOUND
    ):
        return _max_partial_sum_tiled(lhs, rhs)
    blocks = -(-depth // _PREFIX_BLOCK)
    padded = blocks * _PREFIX_BLOCK
    # Zero padding repeats the last prefix; it adds no new prefix value.
    left = torch.nn.functional.pad(lhs, (0, padded - depth))
    right = torch.nn.functional.pad(rhs, (0, padded - depth))
    left_blocks = (
        left.to(torch.float64).reshape(rows, blocks, _PREFIX_BLOCK).permute(1, 0, 2)
    )
    left_magnitude = left_blocks.abs()
    width = max(1, min(columns, _TILE_ELEMENTS // max(1, blocks * rows)))
    best = 0
    found = []
    for col in range(0, columns, width):
        tile = (
            right[col : col + width]
            .to(torch.float64)
            .reshape(-1, blocks, _PREFIX_BLOCK)
            .permute(1, 2, 0)
        )
        sums = torch.bmm(left_blocks, tile)
        ends = sums.cumsum(dim=0)
        best = max(best, int(ends.abs().amax()))
        before = ends.sub_(sums)
        bound = torch.bmm(left_magnitude, tile.abs_()).add_(before.abs())
        block, row, column = torch.nonzero(bound > best, as_tuple=True)
        if block.numel():
            found.append(
                (
                    bound[block, row, column],
                    before[block, row, column],
                    block,
                    row,
                    column + col,
                )
            )
    if not found:
        return best
    # A block whose bound is no larger than the best prefix holds no larger prefix; scan the rest
    # exactly, largest bound first, so a prefix found early prunes the blocks after it.
    limits, before, block, row, column = (
        torch.cat(part) for part in zip(*found, strict=True)
    )
    order = torch.argsort(limits, descending=True, stable=True)
    limits, before, block, row, column = (
        limits[order],
        before[order],
        block[order],
        row[order],
        column[order],
    )
    offsets = torch.arange(_PREFIX_BLOCK)
    for first in range(0, limits.numel(), _SCAN_ROWS):
        keep = (
            torch.nonzero(limits[first : first + _SCAN_ROWS] > best, as_tuple=True)[0]
            + first
        )
        if keep.numel() == 0:
            break
        positions = (block[keep] * _PREFIX_BLOCK)[:, None] + offsets[None, :]
        products = (
            left[row[keep][:, None], positions]
            * right[column[keep][:, None], positions]
        )
        prefixes = products.cumsum(dim=-1) + before[keep].to(torch.int64)[:, None]
        best = max(best, int(prefixes.abs().amax()))
    return best


def run_pt2e_integer_reference(
    model: Any,
    inputs: Iterable[Any],
    *,
    expected_contractions: int | None = None,
    numerics: str = "integer_readout",
) -> IntegerReferenceResult:
    """Execute a converted TorchAO PT2E graph with explicit i32 contractions.

    With explicit host-only ``float_readout``, quantize/dequantize remain TorchAO
    operations, preserving their reciprocal-first rounding behavior. Only the
    contraction is reassociated:
    centered integer operands -> i32 reduction -> activation scale -> weight scale ->
    f32 bias. This legacy mode is approximate and cannot be used as a device oracle.

    The default ``numerics="integer_readout"`` evaluates the integer reduction,
    fp32-canonical scale factors, alpha/beta and bias in exact rational arithmetic, followed
    by one final RNE and clamp or one final IEEE f32 rounding. Its source
    quantize nodes also use exact division rather than eager's f32 reciprocal.

    Covered contractions: Linear, addmm, mm/bmm/matmul, Conv1d/2d/3d and non-transposed
    ``aten.convolution`` with any groups.
    """
    import torch
    import torch.nn.functional as F
    from torch.fx import Interpreter, Node

    from m2m.capture.pt2e_integerize import fusable_requantize

    if numerics not in ("float_readout", "integer_readout"):
        raise ValueError(f"unknown integer reference numerics {numerics!r}")
    integer_readout = numerics == "integer_readout"
    if integer_readout:
        from m2m.capture.quant_scales import canonicalize_pt2e_scales
        canonicalize_pt2e_scales(model)
    conv2d = torch.ops.aten.conv2d.default
    convolution = torch.ops.aten.convolution.default
    convs = (torch.ops.aten.conv1d.default, conv2d, torch.ops.aten.conv3d.default, convolution)
    linear = torch.ops.aten.linear.default
    addmm = torch.ops.aten.addmm.default
    matmuls = (
        torch.ops.aten.matmul.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
    )
    dequant_tensor = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    dequant_channel = torch.ops.quantized_decomposed.dequantize_per_channel.default

    class _IntegerInterpreter(Interpreter):
        def __init__(self, graph_module: Any) -> None:
            # Raw qvalues/scales are read through their dequant nodes at contraction time.
            super().__init__(graph_module, garbage_collect_values=False)
            self.conv2d_count = 0
            self.linear_count = 0
            self.matmul_count = 0
            self.partial_sum_evidence: list[dict[str, Any]] = []
            self.fused_results: dict[Any, Any] = {}

        def record(self, node: Any, kind: str, xq: Any, wq: Any, maximum: int,
                   *, reduction_k: int, reduction_order: str, bias: Any,
                   geometry: dict[str, Any] | None = None) -> None:
            self.partial_sum_evidence.append({
                "node": node.name, "kind": kind, "reduction_k": reduction_k,
                "initial_sum": 0, "max_abs_partial_sum": maximum,
                "reduction_order": reduction_order,
                "bias_application": ("none" if bias is None else
                                     "exact_readout_bias" if integer_readout else "post_scale_f32"),
                "activation": _operand_identity(xq), "weight": _operand_identity(wq),
                "geometry": geometry or {},
            })

        def value(self, item: Any) -> Any:
            return self.env[item] if isinstance(item, Node) else item

        def dequant_precision(self, node: Any) -> Any:
            """Check declared and executed Q/DQ precision, plus exported metadata if present."""
            declared = node.kwargs.get("out_dtype", torch.float32) or torch.float32
            observed = getattr(node.meta.get("tensor_meta") or node.meta.get("val"), "dtype", None)
            value = self.value(node)
            if (
                not isinstance(value, torch.Tensor)
                or (observed is not None and declared != observed)
                or value.dtype != declared
            ):
                raise ValueError(
                    "integer reference dequantize output dtype disagrees with its declaration: "
                    f"declared={declared}, exported={observed}, executed={getattr(value, 'dtype', None)}"
                )
            return declared

        def qparams(self, node: Any, *, weight_axis: int | None) -> tuple[Any, Any, Any, int | None]:
            if not isinstance(node, Node) or node.op != "call_function" or node.target not in (
                (dequant_tensor, dequant_channel) if weight_axis is not None else (dequant_tensor,)
            ):
                raise ValueError(
                    "integer PT2E contraction operand is not produced by a supported "
                    f"dequantize op: {node!r}"
                )
            per_channel = node.target == dequant_channel
            if len(node.args) < (7 if per_channel else 6):
                raise ValueError(f"PT2E dequantize node has incomplete qparams: {node}")
            if node.kwargs.get("out_dtype", torch.float32) not in (None, torch.float32):
                raise ValueError("integer reference requires float32 dequantize output")
            qvalue, scale, zero_point = (self.value(arg) for arg in node.args[:3])
            axis = node.args[3] if per_channel else None
            qmin, qmax, dtype = node.args[4:7] if per_channel else node.args[3:6]
            low, high = (-128, 127) if dtype == torch.int8 else (0, 255)
            if (
                not isinstance(qvalue, torch.Tensor)
                or qvalue.dtype != dtype
                or not isinstance(qmin, int)
                or not isinstance(qmax, int)
                or qmin < low
                or qmax > high
                or qmin >= qmax
                or dtype not in (torch.int8, torch.uint8)
            ):
                raise ValueError("integer reference requires a valid int8/uint8 dequantize operand")
            # DQ's quant_min/max are pattern metadata, not a runtime clamp.
            # Every storage word is supported; the bound below is storage-wide.
            if per_channel:
                if isinstance(axis, int) and -qvalue.ndim <= axis < qvalue.ndim:
                    axis %= qvalue.ndim
                if axis != weight_axis or qvalue.ndim < 1:
                    raise ValueError(f"integer reference requires weight channel axis {weight_axis}")
                if not isinstance(node.args[1], Node) or node.args[1].op != "get_attr":
                    raise ValueError("integer reference requires frozen weight channel scales")
                if (
                    not isinstance(scale, torch.Tensor)
                    or scale.dtype not in (torch.float32, torch.float64)
                    or scale.ndim != 1
                        or scale.numel() != qvalue.shape[axis]
                    or not bool(torch.isfinite(scale).all())
                    or not bool((scale > 0).all())
                ):
                    raise ValueError("integer reference requires positive weight channel scales")
                if zero_point is not None:
                    if not isinstance(node.args[2], Node) or node.args[2].op != "get_attr":
                        raise ValueError("integer reference requires frozen weight channel zero points")
                    if (
                        not isinstance(zero_point, torch.Tensor)
                        or zero_point.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                        or zero_point.ndim != 1
                        or zero_point.numel() != qvalue.shape[axis]
                        or bool(((zero_point.to(torch.int64) < low) | (zero_point.to(torch.int64) > high)).any())
                    ):
                        raise ValueError("integer reference requires storage-representable weight channel zero points")
                return qvalue, scale, zero_point, axis
            if (
                not isinstance(node.args[1], (int, float))
                or not math.isfinite(float(scale))
                or float(scale) <= 0
                or not isinstance(node.args[2], int)
                or not low <= zero_point <= high
            ):
                raise ValueError("integer reference requires a positive scalar scale and storage-representable zero point")
            return qvalue, scale, zero_point, None

        def centered(self, value: Any, zero_point: Any, *, channel_axis: int | None) -> Any:
            q = value.to(torch.int32)
            zp = torch.as_tensor(zero_point, dtype=torch.int32, device=q.device)
            if channel_axis is not None:
                zp = zp.reshape((1,) * channel_axis + (-1,) + (1,) * (q.ndim - channel_axis - 1))
            return q - zp

        def bound(self, value, zero_point):
            low, high = (-128, 127) if value.dtype == torch.int8 else (0, 255)
            zeros = (zero_point.reshape(-1).tolist() if isinstance(zero_point, torch.Tensor)
                     else [0 if zero_point is None else zero_point])
            return max(max(abs(low - z), abs(high - z)) for z in zeros)

        def run_node(self, node: Any) -> Any:
            quantize_ops = (torch.ops.quantized_decomposed.quantize_per_tensor.default,
                            torch.ops.quantized_decomposed.quantize_per_channel.default)
            if (integer_readout and node.op == "call_function" and node.target in quantize_ops
                    and node not in self.fused_results):
                from m2m.capture.exact_quant import quantize_tensor
                args, kwargs = self.fetch_args_kwargs_from_env(node)
                if node.target == quantize_ops[0]:
                    value, scale, zp, low, high, dtype = args
                    axis = 0
                else:
                    value, scale, zp, axis, low, high, dtype = args
                return quantize_tensor(value, scale, zp, axis, low, high, dtype == torch.uint8)
            if node in self.fused_results:
                # The quantize consuming an integer-readout contraction: the
                # contraction already produced its fused int8 requantization.
                return self.fused_results.pop(node)
            if node.op == "call_function" and node.target == torch.ops.aten.convolution_backward.default:
                return self.convolution_backward(node)
            if node.op != "call_function" or node.target not in (*convs, linear, addmm, *matmuls):
                return super().run_node(node)
            if node.target == convolution and bool(node.args[6] if len(node.args) > 6
                                                   else node.kwargs.get("transposed", False)):
                return super().run_node(node)
            operand_nodes = tuple(node.args[1:3] if node.target == addmm else node.args[:2])
            # The integerizer only rewrites contractions whose operands are
            # direct PT2E dequantize nodes.  An attention matmul over the
            # results of integerized linears (often reshaped/transposed) is a
            # floating-point contraction, not another int8 contraction.
            if len(operand_nodes) < 2 or not all(
                isinstance(operand, Node)
                and operand.op == "call_function"
                and operand.target in (dequant_tensor, dequant_channel)
                for operand in operand_nodes
            ):
                return super().run_node(node)
            # Integerization preserves non-f32 Q/DQ contractions in their
            # source floating precision: dequantize first, then contract.
            # Reassociating either BF16/FP16 operand into i32 changes rounding.
            precisions = tuple(self.dequant_precision(operand) for operand in operand_nodes)
            if any(dtype != torch.float32 for dtype in precisions):
                return super().run_node(node)

            args, kwargs = self.fetch_args_kwargs_from_env(node)
            weight_value = args[2] if node.target == addmm else args[1]
            output_axis = weight_value.ndim - 1 if node.target in (addmm, *matmuls) else 0
            qx, sx, zx, _ = self.qparams(operand_nodes[0], weight_axis=None)
            qw, sw, zw, weight_channel = self.qparams(operand_nodes[1], weight_axis=output_axis)
            xq = self.centered(qx, zx, channel_axis=None)
            wq = self.centered(qw, 0 if zw is None else zw, channel_axis=weight_channel)
            product_bound = self.bound(qx, zx) * self.bound(qw, zw)
            alpha = beta = 1
            if node.target == addmm:
                bias = args[0]
                alpha, beta = kwargs.get("alpha", 1), kwargs.get("beta", 1)
                if integer_readout and any(isinstance(v, int) and v != int(float(v)) for v in (alpha, beta)):
                    raise ValueError("exact readout requires alpha and beta to be exactly representable as binary64 scalars")
                if beta == 0:
                    bias = None
            else:
                bias = args[2] if len(args) > 2 else kwargs.get("bias")

            if node.target in (linear, addmm):
                if node.target == linear and (xq.ndim < 2 or wq.ndim != 2 or xq.shape[-1] != wq.shape[-1]):
                    raise ValueError(f"invalid integer Linear geometry: {xq.shape}/{wq.shape}")
                if node.target == addmm and (xq.ndim != 2 or wq.ndim != 2 or xq.shape[1] != wq.shape[0]):
                    raise ValueError(f"invalid integer addmm geometry: {xq.shape}/{wq.shape}")
                if xq.shape[-1] * product_bound > (1 << 31) - 1:
                    raise ValueError("integer Linear may overflow its i32 accumulator")
                right = wq if node.target == linear else wq.transpose(0, 1)
                acc = torch.matmul(xq, right.transpose(-1, -2))
                if integer_readout:
                    out = self.integer_readout(node, acc, sx, sw, bias, alpha, beta, channel_axis=-1)
                else:
                    out = acc.to(torch.float32) * torch.as_tensor(
                        sx, dtype=torch.float32, device=acc.device)
                    out = out * torch.as_tensor(sw, device=acc.device)
                    if weight_channel is not None:
                        out = out.to(torch.float32)
                    if alpha != 1:
                        out = out * alpha
                    if bias is not None:
                        out = out + (bias * beta if beta != 1 else bias)
                left = xq.reshape(-1, xq.shape[-1])
                kind = "linear" if node.target == linear else "addmm"
                self.record(node, kind, xq, wq, _max_partial_sum(left, right),
                            reduction_k=int(xq.shape[-1]), reduction_order="k_ascending", bias=bias)
                if node.target == linear:
                    self.linear_count += 1
                else:
                    self.matmul_count += 1
                return out

            if node.target in matmuls:
                if (
                    xq.ndim < 2
                    or xq.ndim != wq.ndim
                    or xq.shape[:-2] != wq.shape[:-2]
                    or xq.shape[-1] != wq.shape[-2]
                ):
                    raise ValueError(f"invalid integer Matmul geometry: {xq.shape}/{wq.shape}")
                if xq.shape[-1] * product_bound > (1 << 31) - 1:
                    raise ValueError("integer Matmul may overflow its i32 accumulator")
                acc = torch.matmul(xq, wq)
                if integer_readout:
                    out = self.integer_readout(node, acc, sx, sw, None, 1, 1, channel_axis=-1)
                else:
                    out = acc.to(torch.float32) * torch.as_tensor(sx, dtype=torch.float32, device=acc.device)
                    out = out * torch.as_tensor(sw, device=acc.device)
                    if weight_channel is not None:
                        out = out.to(torch.float32)
                maximum = 0
                for index in range(math.prod(xq.shape[:-2])):
                    left = xq.reshape(-1, *xq.shape[-2:])[index]
                    right = wq.reshape(-1, *wq.shape[-2:])[index].transpose(0, 1)
                    maximum = max(maximum, _max_partial_sum(left, right))
                self.record(node, "matmul", xq, wq, maximum,
                            reduction_k=int(xq.shape[-1]), reduction_order="k_ascending", bias=None)
                self.matmul_count += 1
                return out

            rank = xq.ndim - 2
            if rank not in (1, 2, 3) or wq.ndim != xq.ndim:
                raise ValueError(f"integer convolution requires N, C, spatial rank 3-5, got {xq.shape}/{wq.shape}")

            def parameter(index, name, default, minimum):
                raw = args[index] if len(args) > index else kwargs.get(name, default)
                values = (raw,) * rank if isinstance(raw, int) else tuple(int(v) for v in raw)
                if len(values) == 1 and rank > 1:
                    values = values * rank
                if len(values) != rank or min(values) < minimum:
                    raise ValueError(f"invalid convolution {name}: {raw!r}")
                return values

            stride = parameter(3, "stride", 1, 1)
            padding = parameter(4, "padding", 0, 0)
            dilation = parameter(5, "dilation", 1, 1)
            group_index = 8 if node.target == convolution else 6
            groups = int(args[group_index] if len(args) > group_index else kwargs.get("groups", 1))
            if groups < 1 or xq.shape[1] % groups or wq.shape[0] % groups:
                raise ValueError(f"invalid grouped convolution geometry: {xq.shape}/{wq.shape}, groups={groups}")
            kernel = [int(v) for v in wq.shape[2:]]
            taps = math.prod(kernel)
            if int(wq.shape[1]) * taps * product_bound > (1 << 31) - 1:
                raise ValueError("integer convolution may overflow its i32 accumulator")
            batch = int(xq.shape[0])
            cin_group = int(xq.shape[1]) // groups
            cout_group = int(wq.shape[0]) // groups
            out_spatial = [
                (int(xq.shape[2 + i]) + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1) // stride[i] + 1
                for i in range(rank)
            ]
            positions = math.prod(out_spatial)
            # Independent im2col: explicit per-tap strided windows of the zero-padded
            # integer input, K ordered (taps..., channel) as the integerized GEMM.
            padded = F.pad(xq, [p for pad in reversed(padding) for p in (pad, pad)]) if any(padding) else xq
            windows = []
            for tap in __import__("itertools").product(*(range(k) for k in kernel)):
                index = [slice(None), slice(None)]
                for axis, offset in enumerate(tap):
                    start = offset * dilation[axis]
                    index.append(slice(start, start + (out_spatial[axis] - 1) * stride[axis] + 1, stride[axis]))
                windows.append(padded[tuple(index)].reshape(batch, groups, cin_group, positions))
            cols = torch.stack(windows, dim=2).to(torch.int64)  # [N, G, T, Cg, P]
            weights = wq.reshape(groups, cout_group, cin_group, taps).permute(0, 1, 3, 2).to(torch.int64)
            acc = torch.einsum("gotc,ngtcp->ngop", weights, cols)
            acc = acc.reshape(batch, groups * cout_group, *out_spatial).to(torch.int32)
            if integer_readout:
                out = self.integer_readout(node, acc, sx, sw, bias, 1, 1, channel_axis=1)
            else:
                out = acc.to(torch.float32)
                out = out * torch.as_tensor(sx, dtype=torch.float32, device=acc.device)
                weight_scale = torch.as_tensor(sw, device=acc.device)
                if weight_channel is not None:
                    weight_scale = weight_scale.reshape(1, -1, *([1] * rank))
                out = out * weight_scale
                if weight_channel is not None:
                    out = out.to(torch.float32)
                if bias is not None:
                    out = out + bias.reshape(1, -1, *([1] * rank))
            maximum = 0
            prefix_cols = cols.reshape(batch, groups, taps * cin_group, positions)
            prefix_weights = weights.reshape(groups, cout_group, taps * cin_group)
            for batch_index in range(batch):
                for group_index in range(groups):
                    left = prefix_cols[batch_index, group_index].transpose(0, 1)
                    maximum = max(maximum, _max_partial_sum(left, prefix_weights[group_index]))
            kind = "conv2d" if rank == 2 else f"conv{rank}d"
            self.record(node, kind, xq, wq, maximum,
                        reduction_k=cin_group * taps,
                        reduction_order="kernel_taps_channel" if rank != 2 else "kh_kw_channel", bias=bias,
                        geometry={"stride": list(stride), "padding": list(padding),
                                  "dilation": list(dilation), "groups": groups})
            self.conv2d_count += 1
            return out

        def convolution_backward(self, node):
            """Exact integer gradients (float64 sums of int8 products), then the readout."""
            operands = node.args[:3]
            if not all(isinstance(a, Node) and a.op == "call_function" and a.target == dequant_tensor
                       for a in operands) or any(self.dequant_precision(a) != torch.float32 for a in operands):
                return super().run_node(node)
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            (qg, sg, zg, _), (qx, sx, zx, _), (qw, sw, zw, _) = (self.qparams(a, weight_axis=None)
                                                                   for a in operands)
            g, x, w = (self.centered(q, z, channel_axis=None).to(torch.float64)
                       for q, z in ((qg, zg), (qx, zx), (qw, zw)))
            stride, padding, dilation = (list(args[i]) for i in (4, 5, 6))
            if bool(args[7]) or g.ndim != 4:
                raise ValueError("integer reference covers 2-D non-transposed convolution_backward")
            groups, mask = int(args[9]), list(args[10])
            bg, bx, bw = (self.bound(q, z) for q, z in ((qg, zg), (qx, zx), (qw, zw)))
            positions = qg.shape[0] * math.prod(qg.shape[2:])
            if max((qg.shape[1] // groups) * math.prod(qw.shape[2:]) * bg * bw if mask[0] else 0,
                   positions * bg * bx if mask[1] else 0,
                   positions * bg if mask[2] else 0) > (1 << 31) - 1:
                raise ValueError("integer convolution_backward may overflow its i32 accumulator")
            exact = [
                torch.nn.grad.conv2d_input(tuple(x.shape), w, g, stride, padding, dilation, groups) if mask[0] else None,
                torch.nn.grad.conv2d_weight(x, tuple(w.shape), g, stride, padding, dilation, groups) if mask[1] else None,
                g.sum((0, 2, 3)) if mask[2] else None,
            ]
            scales = ([sg, sw], [sg, sx], [sg])
            out = []
            for value, factors in zip(exact, scales):
                if value is None:
                    out.append(None)
                    continue
                acc = value.to(torch.int32)
                if integer_readout:
                    from m2m.capture.exact_quant import exact_readout
                    out.append(exact_readout(acc, float(factors[0]),
                                              [float(factors[1]) if len(factors) > 1 else 1.0]))
                else:
                    value = acc.to(torch.float32)
                    for factor in factors:
                        value = value * float(factor)
                    out.append(value)
            self.conv2d_count += 1
            return tuple(out)

        def integer_readout(self, node, acc, sx, sw, bias, alpha, beta, *, channel_axis):
            """Evaluate exact scale factors and bias, with one final rounding."""
            from m2m.capture.exact_quant import exact_readout
            weights = sw.detach().reshape(-1).tolist() if isinstance(sw, torch.Tensor) else [float(sw)]
            # The shared arithmetic helper consumes no compiler artifacts; the
            # reduction above remains independent of the integerizer.
            transposed = acc.movedim(channel_axis, -1).contiguous()
            moved_bias = bias
            if bias is not None and channel_axis == 1:
                moved_bias = bias.reshape([1] * (acc.ndim - 1) + [-1])
            fused = fusable_requantize(node)
            if fused is not None:
                q, relu, output_scale, zp, low, high, chain = fused
                value = exact_readout(transposed, float(sx), weights, float(alpha), moved_bias, float(beta),
                                      output_scale, zp, low, high, relu is not None, q.args[5] == torch.uint8)
                value = value.movedim(-1, channel_axis)
                for layout_node in chain:
                    value = layout_node.target(value, *layout_node.args[1:], **layout_node.kwargs)
                self.fused_results[q] = value
            out = exact_readout(transposed, float(sx), weights, float(alpha), moved_bias, float(beta))
            return out.movedim(-1, channel_axis)

    runner = _IntegerInterpreter(model)
    with torch.no_grad():
        output = runner.run(*tuple(inputs))
    result = IntegerReferenceResult(output, runner.conv2d_count, runner.linear_count,
                                    runner.matmul_count, tuple(runner.partial_sum_evidence))
    if expected_contractions is not None and result.contraction_count != expected_contractions:
        raise ValueError(
            f"integer reference executed {result.contraction_count} contractions, "
            f"expected {expected_contractions}"
        )
    return result
