"""Result storage contract derived only from the graph being lowered.

Tensor IR has value semantics. This optional module attribute preserves frontend
storage semantics for consumers with a descriptor result ABI. Constants specialize
on captured input layouts; symbolic layout expressions refer to result dimensions.
Descriptor consumers must accept the captured input storage ABI before rebuilding
input-relative views. No eager execution, sample values or operator-name table is
consulted.
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
    # Functionalization replaces in-place writes with fresh value tensors. For
    # shape-preserving mutations, their public storage remains the original input.
    # Replay only dependent metadata operations with that FakeTensor storage so
    # views of a mutation inherit caller strides rather than normalized strides.
    from torch._subclasses.fake_tensor import FakeTensor
    from torch.fx.node import map_arg

    repaired = {}
    mutation_sources = {
        spec.arg.name: next(value for node, value in zip(placeholders, sources) if node.name == spec.target)
        for spec in signature.output_specs
        if spec.kind.name == "USER_INPUT_MUTATION"
    }
    for node in program.graph.nodes:
        value = node.meta.get("val")
        original = mutation_sources.get(node.name)
        if (
            original is not None
            and value.shape == original.shape
            and value.dtype == original.dtype
            and storage(value) != storage(original)
        ):
            if not isinstance(original, FakeTensor):
                raise ValueError("mutation layout reconstruction requires exported FakeTensor metadata")
            repaired[node.name] = original
        elif node.op == "call_function" and any(n.name in repaired for n in node.all_input_nodes):

            def resolve(n):
                return repaired.get(n.name, n.meta.get("val"))

            args, kwargs = map_arg(node.args, resolve), map_arg(node.kwargs, resolve)
            mode = next(value.fake_mode for value in sources if isinstance(value, FakeTensor))
            with mode:
                repaired[node.name] = node.target(*args, **kwargs)

    def result_value(name):
        return repaired.get(name, nodes[name].meta.get("val"))

    mutations = {}
    for spec in signature.output_specs:
        if spec.kind.name == "USER_INPUT_MUTATION":
            mutations[spec.target] = result_value(spec.arg.name)
    records = []
    try:
        for spec in signature.output_specs:
            value = nodes.get(getattr(spec.arg, "name", None))
            value = result_value(value.name) if value else None
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
                if storage(result_value(previous.arg.name)) == storage(value)
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
