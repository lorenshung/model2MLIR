"""Metadata-only corrections for discrepancies in upstream FakeTensor kernels."""

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.node import map_arg


def repair_native_dropout_metadata(exported):
    """Preserve native CPU dropout allocation semantics without sampling data.

    The upstream fake kernel returns its input in inference mode, while the
    native kernel clones nonempty inputs. Empty inputs instead return the input
    and an empty mask of the input dtype. Replay dependent metadata operations
    so views of either result inherit the corrected storage and descriptors.
    """
    repaired = set()
    for node in exported.graph.nodes:
        if node.op != 'call_function':
            continue
        if node.target == torch.ops.aten.native_dropout.default:
            source = node.args[0].meta.get('val')
            if not isinstance(source, FakeTensor) or not all(type(d) is int for d in source.shape):
                continue
            with source.fake_mode:
                if source.numel() == 0:
                    value = source, torch.empty_like(source)
                else:
                    value = source.clone(), torch.empty_like(source, dtype=torch.bool)
            node.meta['val'] = value
            repaired.add(node)
        elif any(parent in repaired for parent in node.all_input_nodes):
            # Scalars do not carry storage identity. Keep their original
            # symbolic metadata rather than creating new data-dependent
            # symbols while reconstructing tensor descriptors.
            if not any(isinstance(value, FakeTensor)
                       for value in torch.utils._pytree.tree_leaves(node.meta.get('val'))):
                continue
            args = map_arg(node.args, lambda parent: parent.meta.get('val'))
            kwargs = map_arg(node.kwargs, lambda parent: parent.meta.get('val'))
            source = next(
                value for parent in node.all_input_nodes
                for value in torch.utils._pytree.tree_leaves(parent.meta.get('val'))
                if isinstance(value, FakeTensor)
            )
            with source.fake_mode:
                node.meta['val'] = node.target(*args, **kwargs)
            repaired.add(node)
    return exported
