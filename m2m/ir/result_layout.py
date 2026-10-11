"""Result storage contract derived only from the graph being lowered.

Tensor IR has value semantics. This optional module attribute preserves frontend
storage semantics for consumers with a descriptor result ABI. Constants specialize
on captured input layouts; symbolic layout expressions refer to result dimensions.
No eager execution, sample values or operator-name table is consulted.
"""

from __future__ import annotations

import json

ATTRIBUTE = "m2m.result_layout"


def exported_layout(program):
    signature = getattr(program, "graph_signature", None)
    if signature is None:
        return None
    nodes = {node.name: node for node in program.graph.nodes}
    placeholders = [node for node in program.graph.nodes if node.op == "placeholder"]
    sources = [node.meta.get("val") for node in placeholders]

    def storage(value):
        return value.untyped_storage()._cdata

    def metadata(value):
        def expression(v):
            import sympy

            expr = getattr(getattr(v, "node", None), "expr", v)
            if isinstance(expr, (int, sympy.Integer)):
                return int(expr)
            axes = []
            rest = expr
            for axis, dim in enumerate(value.shape):
                symbol = getattr(getattr(dim, "node", None), "expr", None)
                if symbol is not None and symbol in rest.free_symbols:
                    while rest % symbol == 0:
                        axes.append(axis)
                        rest = sympy.cancel(rest / symbol)
            if axes and rest == 1:
                return {"shape_axes": axes}
            raise ValueError("unsupported symbolic result layout")

        return dict(
            dtype=str(value.dtype).removeprefix("torch."),
            shape=[int(v) if not getattr(v, "node", None) else -1 for v in value.shape],
            stride=[expression(v) for v in value.stride()],
            storage_offset=expression(value.storage_offset()),
            requires_grad=bool(value.requires_grad),
        )

    if any(not hasattr(value, "untyped_storage") for value in sources):
        return None
    mutations = {}
    for spec in signature.output_specs:
        if spec.kind.name == "USER_INPUT_MUTATION":
            mutations[spec.target] = nodes[spec.arg.name].meta.get("val")
    records = []
    try:
        for spec in signature.output_specs:
            value = nodes.get(getattr(spec.arg, "name", None))
            value = value.meta.get("val") if value else None
            role = spec.kind.name.lower()
            if not hasattr(value, "untyped_storage") or role not in ("user_output", "user_input_mutation"):
                return None
            entry = dict(
                **metadata(value),
                role=role,
                alias_inputs=[
                    i
                    for i, original in enumerate(sources)
                    if storage(original) == storage(value)
                    or (
                        placeholders[i].name in mutations and storage(mutations[placeholders[i].name]) == storage(value)
                    )
                ],
            )
            if role == "user_input_mutation":
                entry["input_index"] = next(i for i, node in enumerate(placeholders) if node.name == spec.target)
            # Result-to-result aliasing also matters for views of fresh intermediates.
            entry["alias_results"] = [
                i
                for i, previous in enumerate(signature.output_specs[: len(records)])
                if storage(nodes[previous.arg.name].meta["val"]) == storage(value)
            ]
            records.append(entry)
        return dict(
            schema_version=1,
            authority="same_conversion_exported_program",
            inputs=[metadata(value) for value in sources],
            results=records,
        )
    except (ValueError, TypeError, RuntimeError, AttributeError):
        return None  # Unrepresentable layouts remain ordinary value-semantic tensors.


def layout_attribute(program, input_types=(), result_types=()):
    from xdsl.dialects.builtin import StringAttr

    layout = exported_layout(program)
    if layout is None:
        return None
    # The importer represents a complex tensor as real components on a trailing
    # axis. Layout units must follow that representation, just like tensor types.
    for records, types in ((layout["inputs"], input_types), (layout["results"], result_types)):
        if len(records) != len(types):
            return None
        for record, typ in zip(records, types):
            if record["dtype"].startswith("complex") and len(typ.get_shape()) == len(record["shape"]) + 1:

                def components(expr):
                    return expr * 2 if type(expr) is int else {**expr, "scale": 2}

                record["shape"].append(2)
                record["stride"] = [components(s) for s in record["stride"]] + [1]
                record["storage_offset"] = components(record["storage_offset"])
                record["dtype"] = str(typ.element_type)
    return StringAttr(json.dumps(layout, sort_keys=True))
