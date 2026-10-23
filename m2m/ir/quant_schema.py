"""Versioned contract for ``prov.quant.*`` attributes, owned by ``quant_schema.json``.

Changes to names, types, requiredness or semantics require a schema version
change. Consumers must reject unknown keys.
"""

from __future__ import annotations

import json
from pathlib import Path

_TEXT = Path(__file__).with_suffix(".json").read_text()
_SCHEMA = json.loads(_TEXT)
SCHEMA_VERSION = _SCHEMA["schema_version"]
ATTRIBUTES = {key.removeprefix(_SCHEMA["namespace"]): spec for key, spec in _SCHEMA["attributes"].items()}


def schema() -> dict:
    """Return a detached JSON-compatible schema, using fully qualified keys."""
    return json.loads(_TEXT)
