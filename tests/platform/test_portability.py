###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Native entrypoints and large CSV fields must work without GPU runtimes."""

import ast
from pathlib import Path
import subprocess
import sys

import pytest

# Constants
ROOT = Path(__file__).resolve().parents[2]
SETUP = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
ENTRYPOINTS = next(
    ast.literal_eval(keyword.value)["console_scripts"]
    for node in ast.walk(SETUP)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "setup"
    for keyword in node.keywords
    if keyword.arg == "entry_points"
)


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_native_entrypoint_help(entrypoint):
    name, target = (part.strip() for part in entrypoint.split("="))
    module, function = target.split(":")
    script = (
        "import importlib,sys; sys.argv=[sys.argv[1],'--help']; "
        f"getattr(importlib.import_module({module!r}), {function!r})()"
    )
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script, name],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower(), result.stdout


@pytest.mark.parametrize("platform_limit", [2**31 - 1, sys.maxsize])
def test_large_csv_fields_with_platform_integer_limits(platform_limit):
    script = """
import csv, io, sys
limit = int(sys.argv[1])
original = csv.field_size_limit
def bounded(value=None):
    if value is None:
        return original()
    if value > limit:
        raise OverflowError('simulated platform C long bound')
    return original(value)
csv.field_size_limit = bounded
from TraceLens.Agent.Analysis.utils import deterministic_fallback
assert 131072 < csv.field_size_limit() <= limit
assert next(csv.reader(io.StringIO('x' * 200000)))[0] == 'x' * 200000
"""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script, str(platform_limit)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
