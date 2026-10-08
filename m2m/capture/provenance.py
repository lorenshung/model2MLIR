"""Byte-bound receipt for a captured model bundle.

The receipt records observed sources and outputs, not a claim that an arbitrary
Python loader's transitive code/data closure is known.  A caller with a .pt2
source passes that archive explicitly; it is hashed once, when the bundle is
complete.  The receipt itself is deliberately outside the hash list.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import torch


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _tool_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    sources = (
        "m2m/api.py",
        "m2m/capture/bundle.py",
        "m2m/capture/provenance.py",
        "m2m/capture/torch_export.py",
        "m2m/capture/torchao_pipeline.py",
        "m2m/ir/decompositions.py",
        "m2m/ir/import_fx.py",
        "m2m/ir/torchmlir_decomps.py",
        "workloads/capture.py",
        "workloads/capture_consistent.py",
    )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "commit": commit,
        "source_sha256": {name: _sha256_file(root / name) for name in sources},
    }


def capture_receipt(out: str | Path, *, source_path: str | Path | None = None) -> dict:
    """Describe the bytes a bundle writer actually published.

    ``source_path`` is an observed input artifact, such as a torch.export .pt2
    or a loader file; it is not proof of dependency closure or authorship.
    """
    out = Path(out)
    required = (
        "model.mlir", "weights.safetensors", "weights.safetensors.manifest.json",
        "inputs.npz", "golden.npy", "extra.npz", "input_order.json",
    )
    artifacts = {name: _file_record(out / name) for name in required}
    # Bind every semantic companion actually published by the bundle writer.
    for pattern in ("goldens.npz", "result_abi.json", "semantic_session.yaml", "*session*.npz", "*quality*.npz", "region_goldens.npz"):
        for path in sorted(out.glob(pattern)):
            if path.is_file(): artifacts[path.name] = _file_record(path)
    lifted = {}
    with np.load(out / "extra.npz", allow_pickle=False) as extras:
        for name in sorted(extras.files):
            if not name.startswith("c_lifted_tensor_"):
                continue
            value = np.asarray(extras[name])
            lifted[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
    manifest = json.loads((out / "weights.safetensors.manifest.json").read_text())
    expected_lifted = sorted({
        str(entry.get("name")) for entry in manifest.values()
        if isinstance(entry, dict) and str(entry.get("name", "")).startswith("c_lifted_tensor_")
    })
    if expected_lifted != sorted(lifted):
        raise ValueError(
            "lifted-constant ABI is incomplete: "
            f"manifest requires {expected_lifted}, extra.npz provides {sorted(lifted)}"
        )
    input_order = json.loads((out / "input_order.json").read_text())
    with np.load(out / "inputs.npz", allow_pickle=False) as inputs:
        materialized_inputs = set(inputs.files)
    expected_inputs = {f"in{i}" for i in range(len(input_order))}
    if materialized_inputs != expected_inputs or sorted(input_order.values()) != list(range(len(input_order))):
        raise ValueError(
            "input ABI is incomplete: input_order must bind every positional in<i> once"
        )
    result_abi_path = out / "result_abi.json"
    if result_abi_path.is_file():
        from m2m.capture.torch_mlir_bridge import _parse_mlir_text_to_xdsl
        from m2m.ir.import_fx import _tensor_type_from_meta
        from xdsl.dialects.func import FuncOp
        abi = json.loads(result_abi_path.read_text())
        records = [record for record in abi["results"] if "key" in record]
        if [record["result_index"] for record in records] != list(range(len(records))):
            raise ValueError("result ABI must bind every function result once")
        expected_types = []
        with np.load(out / "goldens.npz", allow_pickle=False) as goldens:
            if set(goldens.files) != {record["key"] for record in records}:
                raise ValueError("result ABI and golden tensors differ")
            for record in records:
                array = goldens[record["key"]]
                dtype = getattr(torch,record["dtype"])
                if list(array.shape) != record["shape"]:
                    raise ValueError("result golden shape differs from its ABI")
                nominal = torch.empty(record["shape"],dtype=dtype,device="meta")
                expected_types.append(_tensor_type_from_meta(nominal))
                transport = "float32" if record["dtype"] in ("bfloat16", "float8_e4m3fn", "float8_e5m2") else record["dtype"]
                if str(array.dtype) != transport:
                    raise ValueError("result golden dtype differs from its ABI")
        module = _parse_mlir_text_to_xdsl((out / "model.mlir").read_text())
        if module is None:
            raise ValueError("cannot verify result ABI against emitted MLIR")
        functions = [op for op in module.body.block.ops if isinstance(op,FuncOp) and op.sym_name.data=="forward"]
        if len(functions) != 1 or list(functions[0].function_type.outputs) != expected_types:
            raise ValueError("result ABI differs from emitted function outputs")
    source = None
    if source_path is not None:
        path = Path(source_path).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"capture source is not a file: {path}")
        source = {"path": str(path), **_file_record(path)}
    try:
        torchao_version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        torchao_version = None
    return {
        "schema": "m2m.capture-receipt.v1",
        "tool": _tool_identity(),
        "framework": {"torch": torch.__version__, "torchao": torchao_version},
        "source": source,
        "source_closure_verified": False,
        "artifacts": artifacts,
        "materialized_abi": {"complete": True, "inputs": len(input_order),
                             "lifted_constants": expected_lifted},
        "lifted_constants": lifted,
    }


def write_capture_receipt(out: str | Path, *, source_path: str | Path | None = None) -> dict:
    """Write the receipt last, only after all required bundle outputs exist."""
    receipt = capture_receipt(out, source_path=source_path)
    (Path(out) / "capture_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return receipt
