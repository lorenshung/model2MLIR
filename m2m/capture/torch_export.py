"""Frontend capture paths for CompGen.

Captures a PyTorch nn.Module into either a ``torch.export`` ExportedProgram or
TorchDynamo FX graph partitions and builds the canonical capture artifact for
the CompGen frontend boundary.

Invariants:
    - ``capture_model()`` preserves the legacy behaviour and returns the raw
      ``ExportedProgram`` for existing call sites.
    - ``capture_frontend_artifact()`` is the canonical frontend boundary:
      export, optional export decompositions, diagnostics, and unsupported-op
      preparation are all recorded before CompGen takes ownership of the IR.
    - Dynamic shapes, guards, and decomposition provenance are serialized into
      stable dataclasses.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from m2m.capture.dynamo_baseline import DynamoReport, collect_diagnostics
from m2m.capture.external_quantization import ExternalQuantizationConfig
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization
from m2m.capture.unsupported import UnsupportedOpResolution, recover_unsupported_operators
from m2m.capture.unsupported.introspect import runtime_versions
from m2m.ir.decompositions import DECOMPOSITION_TABLE
from m2m.models import CaptureMode


@dataclass(frozen=True)
class ExportValidation:
    """Result of validating an exported program.

    Attributes:
        valid: Whether the export passed all checks.
        round_trip_ok: Whether re-export produces the same graph.
        num_ops: Number of ops in the exported graph.
        graph_breaks: List of graph break descriptions (should be empty).
        warnings: Non-fatal validation warnings.
    """

    valid: bool
    round_trip_ok: bool
    num_ops: int
    graph_breaks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RangeConstraint:
    """Serializable representation of an export dynamic-shape constraint."""

    symbol: str
    minimum: int | None = None
    maximum: int | None = None


@dataclass
class CaptureArtifact:
    """Canonical frontend boundary for CompGen.

    Attributes:
        original_exported_program: Raw ``torch.export.export`` output, if available.
        exported_program: Prepared export after ``run_decompositions()``, if available.
        validation: Validation for the prepared export.
        diagnostics: TorchDynamo diagnostics collected before export.
        decomposition_targets: Export decomposition table keys applied.
        unsupported_resolutions: Unsupported-op recovery results.
        synthesized_payload_translations: Dynamic Payload translations keyed by target.
        explicit_blackboxes: Targets explicitly approved for opaque lowering.
        capture_mode: Which frontend path produced the artifact.
        graphs: Captured FX graph partitions for Dynamo capture.
    """

    original_exported_program: Any | None
    exported_program: Any | None
    validation: ExportValidation
    diagnostics: DynamoReport
    graph_signature: str
    module_call_graph: tuple[str, ...]
    capture_mode: CaptureMode = CaptureMode.TORCH_EXPORT
    graphs: tuple[torch.fx.GraphModule, ...] = ()
    graph_break_count: int = 0
    analysis_success: bool = False
    range_constraints: tuple[RangeConstraint, ...] = ()
    decomposition_targets: tuple[str, ...] = ()
    quantization_config: QuantizationConfig | ExternalQuantizationConfig | None = None
    runtime_versions: dict[str, str] = field(default_factory=dict)
    unsupported_resolutions: list[UnsupportedOpResolution] = field(default_factory=list)
    synthesized_payload_translations: dict[str, Any] = field(default_factory=dict)
    importer_decompositions: dict[str, Any] = field(default_factory=dict)
    explicit_blackboxes: tuple[str, ...] = ()

    def strict_import_options(self) -> dict[str, Any]:
        """Options used by strict Payload import on the prepared graph."""

        return {
            "allow_opaque_fallback": False,
            "explicit_blackboxes": set(self.explicit_blackboxes),
            "dynamic_decompositions": {**self.importer_decompositions,
                                       **self.synthesized_payload_translations},
        }

    @property
    def graph_count(self) -> int:
        """Number of captured graphs represented by this artifact."""

        if self.graphs:
            return len(self.graphs)
        return 1 if self.exported_program is not None else 0


def _load_model_from_path(model_path: str | Path) -> Any:
    """Load an nn.Module from a Python file.

    Expects the file to define a function ``get_model_and_inputs()``
    or a class with a known name.
    """
    path = Path(model_path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    if hasattr(module, "get_model_and_inputs"):
        model, _ = module.get_model_and_inputs()
        return model

    # Try common class names
    for name in ("SimpleMLP", "TransformerBlock", "Model"):
        if hasattr(module, name):
            cls = getattr(module, name)
            return cls()

    raise ImportError(f"No model found in {path}. Define get_model_and_inputs() or a known model class.")


def _resolve_model(model_path: str | Path | nn.Module) -> nn.Module:
    if isinstance(model_path, nn.Module):
        return model_path
    return _load_model_from_path(model_path)


def _count_call_ops(graph: Any) -> int:
    """Count call_function nodes in an FX graph (the actual operations)."""
    return sum(1 for node in graph.nodes if node.op == "call_function")


def _count_graph_partitions(graphs: tuple[torch.fx.GraphModule, ...]) -> int:
    return sum(_count_call_ops(graph.graph) for graph in graphs)


def _serialize_range_constraints(exported_program: Any) -> tuple[RangeConstraint, ...]:
    import math

    def finite_bound(raw: Any) -> int | None:
        if raw is None:
            return None
        try:
            if not math.isfinite(float(raw)):
                return None
            return int(raw)
        except (OverflowError, TypeError, ValueError):
            return None

    constraints = []
    raw_constraints = getattr(exported_program, "range_constraints", {}) or {}
    for symbol, value in raw_constraints.items():
        minimum = finite_bound(getattr(value, "lower", None))
        maximum = finite_bound(getattr(value, "upper", None))
        constraints.append(RangeConstraint(symbol=str(symbol), minimum=minimum, maximum=maximum))
    return tuple(constraints)


def _build_decomposition_table(
    *,
    run_default_decompositions: bool,
    export_decomposition_table: dict[Any, Any] | None,
    preserve_supported_ops: bool = False,
    importer_decompositions: dict[str, Any] | None = None,
) -> tuple[dict[Any, Any], tuple[str, ...]]:
    table: dict[Any, Any] = {}
    if run_default_decompositions:
        default_factory = getattr(torch.export, "default_decompositions", None)
        if callable(default_factory):
            table.update(default_factory())
    if export_decomposition_table:
        table.update(export_decomposition_table)
    if preserve_supported_ops:
        from m2m.ir.core_aten import register_core_decompositions
        local = register_core_decompositions(table)
        if importer_decompositions is not None:
            importer_decompositions.update(local)
        supported = set(DECOMPOSITION_TABLE) | set(local)
        # Functionalization remains in export; only mathematical decomposition
        # of owned source overloads moves to the importer boundary.
        table = {k:v for k,v in table.items()
                 if str(k) not in supported or torch.Tag.core not in getattr(k,"tags",())}
    targets = tuple(sorted(str(key) for key in table.keys()))
    return table, targets


def _retain_static_queries(model,sample_inputs,exported_program):
    """Recover pure metadata queries folded by export before m2m sees them.

    FX traces the source operators; FakeTensor propagation supplies shapes and
    strides only. Restrict this route to stateless tensor programs with no
    attributes or mutable operators, so the existing export ABI remains valid.
    """
    queries={torch.ops.aten.sym_numel.default,torch.ops.aten.sym_size.int,
             torch.ops.aten.sym_stride.int,torch.ops.aten.sym_storage_offset.default,
             torch.ops.aten.sym_is_contiguous.default}
    calls=[node for node in exported_program.graph.nodes if node.op=="call_function"]
    if not calls or any(node.target not in queries and node.target != torch.ops.aten.scalar_tensor.default for node in calls):
        return exported_program
    try:
        import copy
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp
        graph_module=torch.fx.symbolic_trace(model)
        nodes=list(graph_module.graph.nodes)
        queries={torch.ops.aten.sym_numel.default,torch.ops.aten.sym_size.int,
                 torch.ops.aten.sym_stride.int,torch.ops.aten.sym_storage_offset.default,
                 torch.ops.aten.sym_is_contiguous.default}
        if not any(node.target in queries for node in nodes): return exported_program
        if any(node.op in ("get_attr","call_module") for node in nodes): return exported_program
        if any(getattr(getattr(node.target,"_schema",None),"is_mutable",False) for node in nodes):
            return exported_program
        placeholders=[node for node in nodes if node.op=="placeholder"]
        if len(placeholders)!=len(exported_program.graph_signature.input_specs): return exported_program
        FakeTensorProp(graph_module).propagate(*sample_inputs)
        output=next(node for node in nodes if node.op=="output")
        value=output.args[0]
        if not isinstance(value,torch.fx.Node) or not isinstance(value.meta.get("val"),torch.Tensor):
            return exported_program
        output.args=((value,),)
        graph_module.meta["m2m.static_query_graph"]=True
        graph_module.recompile()
        recovered=copy.copy(exported_program)
        recovered._graph_module=graph_module
        recovered._graph_signature=copy.deepcopy(exported_program.graph_signature)
        from torch.export.graph_signature import TensorArgument
        for spec,node in zip(recovered.graph_signature.input_specs,placeholders,strict=True):
            spec.arg=TensorArgument(node.name)
        recovered.graph_signature.output_specs[0].arg=TensorArgument(value.name)
        return recovered
    except (RuntimeError,TypeError,ValueError,AttributeError):
        return exported_program


def _functionalize_metadata_views(exported_program):
    """Represent storage-preserving resize as a view plus an input descriptor update."""
    import math
    graph = exported_program.graph
    mutations = []
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.aten.resize_.default:
            continue
        source = node.args[0]
        value = source.meta.get("val")
        size = list(node.args[1])
        if (value is None or node.kwargs.get("memory_format") not in (None,torch.contiguous_format)):
            continue
        if not value.is_contiguous() or math.prod(size) > value.numel():
            # Read the runtime backing buffer, including holes, rather than
            # reshaping a logical contiguous copy. Growth leaves new elements
            # uninitialized; the importer emits only the preserved storage prefix.
            node.meta['_m2m_storage_resize'] = True
        elif math.prod(size) == value.numel():
            node.target = torch.ops.aten.view.default
        else:
            strides = []
            span = 1
            for extent in reversed(size):
                strides.insert(0, span)
                span *= extent
            node.target = torch.ops.aten.as_strided.default
            node.args = (source, size, strides, value.storage_offset())
        node.kwargs = {}
        node.meta["_m2m_source_aten"] = "aten.resize_.default"
        if source.op == "placeholder":
            mutations.append((node.name,source.name))
        # References to the resized Tensor object after the mutation observe
        # its new descriptor; earlier values and other views retain their shape.
        later = False
        for user in graph.nodes:
            if user is node:
                later = True
            elif later:
                user.replace_input_with(source,node)
    if mutations:
        exported_program.graph_module.recompile()
    return mutations


def _prepare_exported_program(
    exported_program: Any,
    *,
    run_default_decompositions: bool,
    export_decomposition_table: dict[Any, Any] | None,
    capture_trace: bool = False,
    preserve_supported_ops: bool = False,
    importer_decompositions: dict[str, Any] | None = None,
) -> tuple[Any, tuple[str, ...]]:
    table, targets = _build_decomposition_table(
        run_default_decompositions=run_default_decompositions,
        export_decomposition_table=export_decomposition_table,
        preserve_supported_ops=preserve_supported_ops,
        importer_decompositions=importer_decompositions,
    )
    if not table:
        return exported_program, targets
    eliminations: list[dict[str, Any]] = list(
        getattr(exported_program.graph_module, "_m2m_trace_eliminations", ())
    ) if capture_trace else []
    if capture_trace:
        from m2m.capture.trace import tuple_selection_trace_program

        exported_program = tuple_selection_trace_program(exported_program)
        import functools
        from torch.fx.traceback import get_current_meta

        # Only certify an identity elimination at its exact source operation,
        # not an arbitrary nested decomposition that happens to return an input.
        source_targets = {
            n.meta.get("custom", {}).get("m2m_node_id"): str(n.target)
            for n in exported_program.graph.nodes
        }
        def wrap(target, function):
            @functools.wraps(function)
            def traced(*args, **kwargs):
                custom = get_current_meta().get("custom") or {}
                source_id = custom.get("m2m_node_id")
                result = function(*args, **kwargs)
                if source_targets.get(source_id) == str(target):
                    for index, arg in enumerate(args):
                        if isinstance(arg, torch.Tensor) and result is arg:
                            eliminations.append({"source_ids": list(custom.get("m2m_lineage") or ()),
                                                 "kind": "eliminated", "reason": "identity decomposition",
                                                 "argument_index": index})
                            break
                return result
            return traced
        table = {target: wrap(target, function) for target, function in table.items()}
    metadata_mutations = []
    candidate = exported_program
    if preserve_supported_ops and any(node.target == torch.ops.aten.resize_.default for node in exported_program.graph.nodes):
        import copy
        candidate = copy.copy(exported_program)
        graph = torch.fx.Graph()
        mapping = {}
        for node in exported_program.graph.nodes:
            mapping[node] = graph.node_copy(node, lambda value: mapping[value])
        candidate._graph_module = torch.fx.GraphModule(exported_program.graph_module, graph)
        candidate._graph_signature = copy.deepcopy(exported_program.graph_signature)
        metadata_mutations = _functionalize_metadata_views(candidate)
        if any(node.target == torch.ops.aten.resize_.default and not node.meta.get('_m2m_storage_resize')
               for node in candidate.graph.nodes):
            raise ValueError('resize_ requires backing storage outside logical tensor inputs '
                             '(noncontiguous holes or newly allocated elements)')
    if metadata_mutations:
        # AOT functionalization cannot represent a metadata-only resize in
        # torch 2.13. Its guard replay reads a dead descriptor. Keep the graph
        # with the explicit view; do not hide other mutable source effects.
        if any(getattr(getattr(node.target,"_schema",None),"is_mutable",False)
               and not node.meta.get('_m2m_storage_resize') for node in candidate.graph.nodes):
            raise ValueError("metadata resize combined with data mutation requires functionalization")
        prepared = candidate
    elif (exported_program.graph_module.meta.get("m2m.static_query_graph")
          or exported_program.graph_module.meta.get("m2m.static_bridge_graph")):
        prepared = exported_program
    else:
        prepared = exported_program.run_decompositions(table)
    if metadata_mutations:
        from torch.export.graph_signature import OutputKind, OutputSpec, TensorArgument
        by_name = {node.name:node for node in prepared.graph.nodes}
        output = next(node for node in prepared.graph.nodes if node.op=="output")
        extra = []
        specs = []
        for name, input_name in metadata_mutations:
            value = by_name[name]
            value.meta["_m2m_source_aten"] = "aten.resize_.default"
            value.meta["_m2m_metadata_mutation_input"] = input_name
            extra.append(value)
            specs.append(OutputSpec(OutputKind.USER_INPUT_MUTATION,TensorArgument(value.name),input_name))
        output.args = (tuple(extra)+tuple(output.args[0]),)
        prepared.graph_signature.output_specs[:0] = specs
        prepared.graph_module.recompile()
    if preserve_supported_ops:
        # Functionalization must expose mutations as results. Its torch_fn
        # metadata retains the exact mutable source overload even when the
        # executable node is now a functional variant.
        mutable_sources = {}
        for node in exported_program.graph.nodes:
            schema = getattr(node.target, "_schema", None)
            key = node.meta.get("torch_fn")
            if schema is not None and schema.is_mutable and key is not None:
                mutable_sources[tuple(key)] = str(node.target)
        for node in prepared.graph.nodes:
            key = node.meta.get("torch_fn")
            if key is not None and tuple(key) in mutable_sources:
                node.meta["_m2m_source_aten"] = mutable_sources[tuple(key)]
    if capture_trace:
        prepared._m2m_trace_eliminations = eliminations
    return prepared, targets


def capture_model(
    model_path: str | Path | nn.Module,
    sample_inputs: Any,
    dynamic_shapes: dict[str, Any] | None = None,
) -> Any:
    """Capture a PyTorch model via torch.export.

    Args:
        model_path: Path to a Python file, or a live nn.Module.
        sample_inputs: Sample input tensors (tuple).
        dynamic_shapes: Optional dynamic shape specifications.

    Returns:
        A torch.export.ExportedProgram.
    """
    model = _resolve_model(model_path)

    model.eval()

    # Ensure model and inputs are on CPU to avoid FakeTensor device propagation
    # errors during torch.export tracing. Skip the move when already on CPU:
    # .to() on quantized tensor-subclass weights forces a swap_tensors that trips
    # the weakref guard ("Couldn't swap Linear.weight").
    if any(getattr(p, "device", None) is not None and p.device.type != "cpu" for p in model.parameters()):
        model = model.to("cpu")
    if isinstance(sample_inputs, (tuple, list)):
        sample_inputs = tuple(t.to("cpu") if isinstance(t, torch.Tensor) else t for t in sample_inputs)

    # Export attempts may execute metadata-mutating operators before failing.
    # Give each attempt its own alias-preserving input storage and descriptors.
    import copy
    from torch.utils._pytree import tree_map
    def export_inputs():
        try:
            return copy.deepcopy(sample_inputs)
        except RuntimeError:
            detached = tree_map(lambda value: value.detach() if isinstance(value,torch.Tensor) else value,
                                sample_inputs)
            copied = copy.deepcopy(detached)
            source_leaves = torch.utils._pytree.tree_leaves(sample_inputs)
            copied_leaves = torch.utils._pytree.tree_leaves(copied)
            for source, value in zip(source_leaves,copied_leaves,strict=True):
                if isinstance(source,torch.Tensor): value.requires_grad_(source.requires_grad)
            return copied
    kwargs: dict[str, Any] = {}
    if dynamic_shapes is not None:
        kwargs["dynamic_shapes"] = dynamic_shapes

    try:
        exported = torch.export.export(model, export_inputs(), **kwargs)
    except Exception as first_err:
        msg = str(first_err)
        if dynamic_shapes is None and ("local_scalar_dense/item NYI" in msg
                                      or "result type Float can't be cast" in msg):
            from m2m.capture.static_aten import capture_static_bridges
            try:
                return capture_static_bridges(model, export_inputs())
            except (RuntimeError, TypeError, ValueError, AttributeError):
                raise first_err
        # Retry under non-strict for known recoverable failure modes:
        #  * FakeTensor / device-propagation issues
        #  * GuardOnDataDependentSymNode — strict mode rejects size-like
        #    symbolic shapes that show up in models doing dynamic image
        #    tile counts (e.g. SmolVLA / SmolVLM); non-strict mode
        #    accepts them as runtime guards.
        recoverable = (
            "FakeTensor" in msg
            or "device" in msg.lower()
            or "GuardOnDataDependentSymNode" in msg
            or "data-dependent" in msg
        )
        if "sources must not be empty for symbol" in msg or "num_mutated_inp_runtime_indices" in msg:
            # Non-strict symbolic replay loses metadata-mutation sources on
            # torch 2.13. Strict tracing retains their descriptor dependencies.
            import sys
            import types
            module_name = type(model).__module__
            temporary = module_name not in sys.modules
            if temporary:
                module = types.ModuleType(module_name)
                module.__dict__.update(model.forward.__globals__)
                sys.modules[module_name] = module
            try:
                exported = torch.export.export(model,export_inputs(),**{**kwargs,"strict":True})
            finally:
                if temporary: sys.modules.pop(module_name,None)
        elif recoverable:
            kwargs["strict"] = False
            exported = torch.export.export(model, export_inputs(), **kwargs)
        else:
            raise
    if dynamic_shapes is None and any(node.target==torch.ops.aten.resize_.default for node in exported.graph.nodes):
        # Export can retain the post-resize FakeTensor descriptor on an input
        # placeholder. The function ABI must describe the pre-invocation input.
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.export.graph_signature import InputKind
        nodes={node.name:node for node in exported.graph.nodes}
        leaves=iter(torch.utils._pytree.tree_leaves(sample_inputs))
        mode=FakeTensorMode()
        for spec in exported.graph_signature.input_specs:
            if spec.kind != InputKind.USER_INPUT: continue
            value=next(leaves)
            if isinstance(value,torch.Tensor):
                nodes[spec.arg.name].meta['val']=mode.from_tensor(value)
    return exported


def capture_frontend_artifact(
    model_path: str | Path | nn.Module,
    sample_inputs: Any,
    dynamic_shapes: dict[str, Any] | None = None,
    *,
    quantization_config: QuantizationConfig | ExternalQuantizationConfig | None = None,
    run_default_decompositions: bool = True,
    export_decomposition_table: dict[Any, Any] | None = None,
    preserve_supported_ops: bool = False,
) -> CaptureArtifact:
    """Build the canonical export-boundary artifact for CompGen.

    ``capture_model()`` returns the raw exported program for backward
    compatibility. This function is the strict frontend path used by
    ``compile_model()`` and by agentic analysis.
    """

    model = _resolve_model(model_path)

    model.eval()

    active_model = model
    if quantization_config is not None:
        try:
            inputs = tuple(sample_inputs) if isinstance(sample_inputs, (tuple, list)) else (sample_inputs,)
            active_model = apply_quantization(model, quantization_config, example_inputs=inputs)
        except Exception:
            if isinstance(quantization_config, ExternalQuantizationConfig):
                raise
            active_model = model

    diagnostics = collect_diagnostics(active_model, sample_inputs)
    original_exported = capture_model(active_model, sample_inputs, dynamic_shapes=dynamic_shapes)
    if preserve_supported_ops and dynamic_shapes is None:
        original_exported = _retain_static_queries(active_model,sample_inputs,original_exported)
    importer_decompositions = {}
    prepared_exported, decomposition_targets = _prepare_exported_program(
        original_exported,
        run_default_decompositions=run_default_decompositions,
        export_decomposition_table=export_decomposition_table,
        preserve_supported_ops=preserve_supported_ops,
        importer_decompositions=importer_decompositions,
    )
    validation = validate_export(prepared_exported)
    versions = runtime_versions()

    unsupported_resolutions = recover_unsupported_operators(
        prepared_exported,
        supported_targets=set(DECOMPOSITION_TABLE) | set(importer_decompositions),
        runtime_versions=versions,
    )
    explicit_blackboxes = tuple(
        sorted(resolution.target for resolution in unsupported_resolutions if resolution.approved_blackbox)
    )
    synthesized_payload_translations = {
        resolution.target: resolution.translation.translator
        for resolution in unsupported_resolutions
        if resolution.translation is not None
    }

    return CaptureArtifact(
        original_exported_program=original_exported,
        exported_program=prepared_exported,
        validation=validation,
        diagnostics=diagnostics,
        graph_signature=str(getattr(prepared_exported, "graph_signature", "")),
        module_call_graph=tuple(str(item) for item in (getattr(prepared_exported, "module_call_graph", None) or ())),
        capture_mode=CaptureMode.TORCH_EXPORT,
        graphs=(),
        graph_break_count=len(diagnostics.graph_breaks),
        analysis_success=validation.valid,
        range_constraints=_serialize_range_constraints(prepared_exported),
        decomposition_targets=decomposition_targets,
        quantization_config=quantization_config,
        runtime_versions=versions,
        unsupported_resolutions=unsupported_resolutions,
        synthesized_payload_translations=synthesized_payload_translations,
        importer_decompositions=importer_decompositions,
        explicit_blackboxes=explicit_blackboxes,
    )


def capture_dynamo_partitions(
    model_path: str | Path | nn.Module,
    sample_inputs: Any,
    *,
    fullgraph: bool = False,
) -> CaptureArtifact:
    """Capture FX graph partitions via TorchDynamo/torch.compile."""

    model = _resolve_model(model_path)
    model.eval()
    diagnostics = collect_diagnostics(model, sample_inputs)
    versions = runtime_versions()

    import torch._dynamo as dynamo

    dynamo.reset()
    captured: list[torch.fx.GraphModule] = []

    def capture_backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Any:
        captured.append(gm)
        return gm.forward

    compiled = torch.compile(model, backend=capture_backend, fullgraph=fullgraph)
    with torch.no_grad():
        compiled(*sample_inputs)

    graphs = tuple(captured)
    total_ops = _count_graph_partitions(graphs)
    validation = ExportValidation(
        valid=bool(graphs),
        round_trip_ok=bool(graphs),
        num_ops=total_ops,
        graph_breaks=[reason for _, reason in diagnostics.graph_breaks],
        warnings=[] if graphs else ["TorchDynamo produced no graph partitions"],
    )

    return CaptureArtifact(
        original_exported_program=None,
        exported_program=None,
        validation=validation,
        diagnostics=diagnostics,
        graph_signature="",
        module_call_graph=tuple(),
        capture_mode=CaptureMode.TORCH_DYNAMO_PARTITIONED,
        graphs=graphs,
        graph_break_count=len(diagnostics.graph_breaks),
        analysis_success=bool(graphs),
        range_constraints=(),
        decomposition_targets=(),
        quantization_config=None,
        runtime_versions=versions,
        unsupported_resolutions=[],
        synthesized_payload_translations={},
        explicit_blackboxes=(),
    )


def capture_frontend(
    model_path: str | Path | nn.Module,
    sample_inputs: Any,
    *,
    dynamic_shapes: dict[str, Any] | None = None,
    prefer_dynamo: bool = False,
    fallback_to_dynamo: bool = False,
) -> CaptureArtifact:
    """Capture a model using the preferred frontend mode."""

    if prefer_dynamo:
        return capture_dynamo_partitions(model_path, sample_inputs)
    try:
        return capture_frontend_artifact(model_path, sample_inputs, dynamic_shapes=dynamic_shapes)
    except Exception:
        if not fallback_to_dynamo:
            raise
    return capture_dynamo_partitions(model_path, sample_inputs)


def validate_export(exported_program: Any) -> ExportValidation:
    """Validate an exported program.

    Counts ops, checks for graph breaks, and verifies the graph is well-formed.
    """
    graph = exported_program.graph
    num_ops = _count_call_ops(graph)
    warnings: list[str] = []

    # Check for unsupported patterns
    graph_breaks: list[str] = []
    for node in graph.nodes:
        if node.op == "call_function":
            target_name = str(node.target)
            if "graph_break" in target_name.lower():
                graph_breaks.append(f"Graph break at {node.name}: {target_name}")

    # Simple round-trip check: verify the graph has expected structure
    has_output = any(n.op == "output" for n in graph.nodes)
    has_placeholder = any(n.op == "placeholder" for n in graph.nodes)
    round_trip_ok = has_output and has_placeholder

    if not has_output:
        warnings.append("Graph has no output node")
    if num_ops == 0:
        warnings.append("Graph has no call_function ops")

    valid = round_trip_ok and len(graph_breaks) == 0

    return ExportValidation(
        valid=valid,
        round_trip_ok=round_trip_ok,
        num_ops=num_ops,
        graph_breaks=graph_breaks,
        warnings=warnings,
    )


__all__ = [
    "CaptureArtifact",
    "ExportValidation",
    "RangeConstraint",
    "capture_dynamo_partitions",
    "capture_frontend",
    "capture_frontend_artifact",
    "capture_model",
    "validate_export",
]
