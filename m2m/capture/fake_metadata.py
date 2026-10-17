"""Metadata-only corrections for discrepancies in upstream FakeTensor kernels."""

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.node import map_arg


def repair_native_dropout_metadata(exported):
    """Preserve native CPU allocation and shape semantics without sampling data.

    Fresh copy/scatter preserve the source offset (an overlapping source gets
    a dense empty_like layout at offset 0); resizing mutations retain
    the original offset; BF16 embedding sum has the same fast auxiliary shape
    as FP32. Corrections describe storage, never evaluate sample values.

    The upstream dropout fake kernel returns its input in inference mode, while the
    native kernel clones nonempty inputs. Empty inputs instead return the input
    and an empty mask of the input dtype. Replay dependent metadata operations
    so views of either result inherit the corrected storage and descriptors.
    """
    repaired = set()
    nodes = {node.name: node for node in exported.graph.nodes}
    mutations = {
        spec.arg.name: nodes[spec.target].meta.get("val")
        for spec in getattr(getattr(exported, "graph_signature", None), "output_specs", ())
        if spec.kind.name == "USER_INPUT_MUTATION"
    }

    def fresh_layout(value, offset):
        # Allocate metadata storage only. The compiler creates the actual
        # independent allocation and copies logical values into this layout.
        shape, strides = list(value.shape), list(value.stride())
        span = offset + 1 + sum((n - 1) * s for n, s in zip(shape, strides))
        if 0 in shape:
            span = offset
        with value.fake_mode:
            base = torch.empty((span,), dtype=value.dtype, device=value.device)
            return base.as_strided(shape, strides, offset)

    for node in exported.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target == torch.ops.aten.native_dropout.default:
            source = node.args[0].meta.get("val")
            if not isinstance(source, FakeTensor) or not all(type(d) is int for d in source.shape):
                continue
            with source.fake_mode:
                if source.numel() == 0:
                    value = source, torch.empty_like(source)
                else:
                    value = source.clone(), torch.empty_like(source, dtype=torch.bool)
            node.meta["val"] = value
            repaired.add(node)
        elif node.target in (
            torch.ops.aten.copy.default,
            torch.ops.aten.select_scatter.default,
            torch.ops.aten.slice_scatter.default,
            torch.ops.aten.diagonal_scatter.default,
        ):
            source = node.args[0].meta.get("val")
            if (
                isinstance(source, FakeTensor)
                and source.device.type == "cpu"
                and all(type(d) is int for d in (*source.shape, *source.stride(), source.storage_offset()))
            ):
                if torch._debug_has_internal_overlap(source) != 1:
                    node.meta["val"] = fresh_layout(source, source.storage_offset())
                else:
                    # An overlapping (expanded) source cannot be cloned in place
                    # of its strides: CPU allocates empty_like(source), whose
                    # suggested format is dense, at offset 0. The upstream fake
                    # scatter kernels instead keep the source's zero strides.
                    with source.fake_mode:
                        node.meta["val"] = torch.empty_like(source)
                repaired.add(node)
        elif node.target == torch.ops.aten.native_group_norm.default:
            source = node.args[0].meta.get("val")
            parameters = [
                value.meta.get("val") if isinstance(value, torch.fx.Node) else value for value in node.args[1:3]
            ]
            if (
                isinstance(source, FakeTensor)
                and source.device.type == "cpu"
                and source.dtype in (torch.float16, torch.bfloat16)
                and any(getattr(value, "dtype", None) == torch.float32 for value in parameters)
            ):
                # The mixed CPU kernel saves opmath statistics in the affine
                # parameter dtype; the upstream fake kernel uses input dtype.
                values = node.meta["val"]
                with source.fake_mode:
                    node.meta["val"] = (
                        values[0],
                        *(torch.empty_like(value, dtype=torch.float32) for value in values[1:]),
                    )
                repaired.add(node)
        elif node.target == torch.ops.aten._embedding_bag.default:
            source = node.args[0].meta.get("val")
            mode = node.args[4] if len(node.args) > 4 else node.kwargs.get("mode", 0)
            padding = node.args[8] if len(node.args) > 8 else node.kwargs.get("padding_idx", -1)
            sample = node.args[6] if len(node.args) > 6 else node.kwargs.get("per_sample_weights")
            sample_value = sample.meta.get("val") if isinstance(sample, torch.fx.Node) else sample
            from m2m.ir.embedding_bag_cpu import fast_path_eligible

            if (
                isinstance(source, FakeTensor)
                and source.device.type == "cpu"
                and source.dtype == torch.bfloat16
                and mode == 0
                and fast_path_eligible(source, sample_value, padding_idx=padding)
            ):
                values = list(node.meta["val"])
                with source.fake_mode:
                    values[1] = torch.empty((0,), dtype=values[1].dtype, device=source.device)
                node.meta["val"] = tuple(values)
                repaired.add(node)
        elif node.name in mutations:
            source, value = mutations[node.name], node.meta.get("val")
            if (
                isinstance(source, FakeTensor)
                and isinstance(value, FakeTensor)
                and source.device.type == "cpu"
                and source.shape != value.shape
                and source.dtype == value.dtype
                and all(type(d) is int for d in (*value.shape, *value.stride(), source.storage_offset()))
            ):
                # CPU resize and out= shape changes retain the Tensor's offset,
                # even when its backing storage must grow. Functionalization
                # represents only the new values and otherwise drops that fact.
                node.meta["val"] = fresh_layout(value, source.storage_offset())
                repaired.add(node)
        elif any(parent in repaired for parent in node.all_input_nodes):
            # Scalars do not carry storage identity. Keep their original
            # symbolic metadata rather than creating new data-dependent
            # symbols while reconstructing tensor descriptors.
            if not any(
                isinstance(value, FakeTensor) for value in torch.utils._pytree.tree_leaves(node.meta.get("val"))
            ):
                continue
            args = map_arg(node.args, lambda parent: parent.meta.get("val"))
            kwargs = map_arg(node.kwargs, lambda parent: parent.meta.get("val"))
            source = next(
                value
                for parent in node.all_input_nodes
                for value in torch.utils._pytree.tree_leaves(parent.meta.get("val"))
                if isinstance(value, FakeTensor)
            )
            with source.fake_mode:
                node.meta["val"] = node.target(*args, **kwargs)
            repaired.add(node)
    return exported
