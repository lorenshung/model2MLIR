"""Explicit native CI configuration must fail when the toolchain is broken."""
import os
from pathlib import Path
import shutil
import subprocess
import pytest


def pytest_sessionstart(session):
    if not any(os.environ.get(k) for k in ('MODEL2MLIR_MLIR_TRANSLATE', 'MODEL2MLIR_CLANG')):
        return
    translate = os.environ.get('MODEL2MLIR_MLIR_TRANSLATE') or shutil.which('mlir-translate')
    clang = os.environ.get('MODEL2MLIR_CLANG') or shutil.which('clang')
    paths = [translate, clang, str(Path(translate).with_name('mlir-opt')) if translate else None]
    for path in paths:
        if not path or not Path(path).is_file() or not os.access(path, os.X_OK):
            raise pytest.UsageError(f'configured native toolchain is missing an executable: {path}')
        try:
            subprocess.run([path, '--version'], check=True, capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            raise pytest.UsageError(f'configured native toolchain cannot run {path}: {exc}') from exc
