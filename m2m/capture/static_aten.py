"""Static ATen-only capture for bridges unsupported by torch's fake kernels.

Trace operators without evaluating input data. Neutral scalar placeholders
are restricted to consumers where they cannot affect shape or control flow.
"""
import copy
import operator
import torch
from torch.fx.passes.fake_tensor_prop import FakeTensorProp


def capture_static_bridges(model, inputs):
    # Give FX fixed arity even when a caller's adapter exposes forward(*args).
    names = [f'input_{i}' for i in range(len(inputs))]
    scope = {}
    exec(f"def forward(self, {', '.join(names)}):\n    return self.inner({', '.join(names)})", scope)
    wrapper = type('StaticBridge', (torch.nn.Module,), {'forward': scope['forward']})()
    wrapper.inner = model
    class SchemaProxy(torch.fx.Proxy):
        def __len__(self):
            schema = getattr(self.node.target, '_schema', None)
            if schema is not None and len(schema.returns) > 1:
                return len(schema.returns)
            return super().__len__()

    class SchemaTracer(torch.fx.Tracer):
        def proxy(self, node):
            return SchemaProxy(node, self)
        def iter(self, proxy):
            schema = getattr(proxy.node.target, '_schema', None)
            if schema is not None and len(schema.returns) > 1:
                return iter(proxy[i] for i in range(len(schema.returns)))
            return super().iter(proxy)

    gm = torch.fx.GraphModule(wrapper, SchemaTracer().trace(wrapper))
    nodes = list(gm.graph.nodes)
    item = torch.ops.aten._local_scalar_dense.default
    scalar = torch.ops.aten.scalar_tensor.default
    dropout = torch.ops.aten.native_dropout.default
    if not any(n.op == 'call_function' and n.target in (item, dropout) for n in nodes):
        raise ValueError('no static scalar/dropout bridge')
    for node in nodes:
        if node.op not in ('placeholder', 'call_function', 'output'):
            raise ValueError('static bridge capture requires a stateless ATen-only graph')
        if node.op == 'call_function':
            schema = getattr(node.target, '_schema', None)
            if node.target is not operator.getitem and (schema is None or schema.is_mutable):
                raise ValueError('static bridge capture requires immutable ATen operators')
            if node.target == item and any(user.target != scalar for user in node.users):
                raise ValueError('complex item may only feed scalar_tensor in static capture')

    class Metadata(FakeTensorProp):
        def call_function(self, target, args, kwargs):
            if target == item and args[0].dtype == torch.complex64:
                if args[0].numel() != 1:
                    raise ValueError('item requires one element')
                # Only type propagation; the runtime pair stays on the FX edge.
                return 0.
            if target == dropout and args[0].dtype == torch.complex64:
                source = args[0]
                p = args[1] if len(args) > 1 else kwargs['p']
                if not 0 <= p <= 1:
                    raise ValueError('dropout probability is outside [0,1]')
                with self._mode:
                    return torch.empty_like(source), torch.empty_like(source, dtype=torch.bool)
            return super().call_function(target, args, kwargs)

    propagated = Metadata(gm).propagate(*inputs)
    flat, out_spec = torch.utils._pytree.tree_flatten(propagated)
    if not all(isinstance(v, torch.Tensor) for v in flat):
        raise ValueError('static bridge capture requires tensor outputs')
    # Give the scalar bridge explicit dtype evidence without fabricating values.
    from types import SimpleNamespace
    for node in nodes:
        if node.op == 'call_function' and node.target == item:
            source = node.args[0].meta.get('val')
            if getattr(source, 'dtype', None) == torch.complex64:
                node.meta['val'] = SimpleNamespace(dtype=torch.complex64)
    output = next(n for n in nodes if n.op == 'output')
    flat_nodes, _ = torch.utils._pytree.tree_flatten(output.args[0])
    output.args = (tuple(flat_nodes),)
    gm.meta['m2m.static_bridge_graph'] = True
    gm.recompile()

    class Skeleton(torch.nn.Module):
        def forward(self, *args):
            return args[0]
    exported = torch.export.export(Skeleton(), inputs)
    from torch.export.graph_signature import InputKind, InputSpec, OutputKind, OutputSpec, TensorArgument
    exported._graph_module = gm
    exported._graph_signature = copy.deepcopy(exported.graph_signature)
    placeholders = [n for n in nodes if n.op == 'placeholder']
    exported.graph_signature.input_specs[:] = [InputSpec(InputKind.USER_INPUT, TensorArgument(n.name), None) for n in placeholders]
    exported.graph_signature.output_specs[:] = [OutputSpec(OutputKind.USER_OUTPUT, TensorArgument(n.name), None) for n in flat_nodes]
    exported.module_call_graph[0].signature.out_spec = out_spec
    return exported
