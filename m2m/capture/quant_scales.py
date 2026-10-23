"""Canonicalize frozen PT2E scales once, before exact integerization/reference."""

from __future__ import annotations


def canonicalize_pt2e_scales(model):
    """Round Q/DQ scale factors to fp32 and record every semantic change.

    New buffers isolate scale uses from unrelated consumers of the original
    tensor. Runtime scales stay SSA operands and grant no static admission.
    The operation is idempotent; the model carries its original change ledger.
    """
    import math
    import torch
    from torch.fx import Node
    from m2m.ir.quant_attrs import fp32

    ledger = list(model.meta.get("m2m_scale_canonicalization", []))
    replacements = {}
    changed = False
    for node in list(model.graph.nodes):
        target = str(node.target)
        if node.op != "call_function" or not target.startswith("quantized_decomposed."):
            continue
        if not any(name in target for name in ("quantize_per_tensor.", "quantize_per_channel.")):
            continue
        scale = node.args[1]
        if isinstance(scale, (float, int)):
            canonical = fp32(scale)
            if not math.isfinite(canonical) or canonical <= 0:
                raise ValueError(f"{node.name}: scale cannot be represented as finite positive fp32")
            if scale != canonical:
                node.args = (node.args[0], canonical, *node.args[2:])
                ledger.append({"node": node.name, "source": scale, "canonical": canonical})
                changed = True
        elif isinstance(scale, Node) and scale.op == "get_attr":
            value = model
            for part in str(scale.target).split("."):
                value = getattr(value, part)
            canonical = value.detach().to(torch.float32)
            if not bool(torch.isfinite(canonical).all() and (canonical > 0).all()):
                raise ValueError(f"{node.name}: scale cannot be represented as finite positive fp32")
            if not torch.equal(value.to(torch.float64), canonical.to(torch.float64)):
                if scale not in replacements:
                    name = f"_m2m_fp32_scale_{len(ledger)}"
                    while hasattr(model, name):
                        name += "_"
                    model.register_buffer(name, canonical)
                    with model.graph.inserting_before(node):
                        replacements[scale] = model.graph.get_attr(name)
                node.args = (node.args[0], replacements[scale], *node.args[2:])
                ledger.append({"node": node.name, "source": value.flatten().tolist(),
                               "canonical": canonical.flatten().tolist()})
                changed = True
    model.meta["m2m_scale_canonicalization"] = ledger
    if changed:
        model.graph.lint()
        model.recompile()
    return ledger
