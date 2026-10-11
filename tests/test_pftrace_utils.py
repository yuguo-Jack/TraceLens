###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for TraceLens.Reporting.pftrace_utils."""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from TraceLens.Reporting.pftrace_utils import (
    acquire_traceconv,
    ensure_trace_json,
    run,
)


def test_run_success():
    run([sys.executable, "-c", "print('ok')"])


def test_run_failure_raises():
    with pytest.raises(RuntimeError, match="Command failed"):
        run([sys.executable, "-c", "import sys; sys.exit(1)"])


def test_ensure_trace_json_returns_json_path(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": []}))
    assert ensure_trace_json(str(trace)) == str(trace.resolve())


def test_ensure_trace_json_returns_json_gz_path(tmp_path):
    trace = tmp_path / "trace.json.gz"
    trace.write_bytes(b"")
    result = ensure_trace_json(str(trace))
    assert result.endswith(".json.gz")


def test_ensure_trace_json_unsupported_format(tmp_path):
    bad = tmp_path / "trace.bin"
    bad.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="Unsupported trace format"):
        ensure_trace_json(str(bad))


def test_acquire_traceconv_uses_preferred_path(tmp_path):
    preferred = tmp_path / "traceconv"
    preferred.write_text("#!/bin/sh\n")
    preferred.chmod(0o755)
    result = acquire_traceconv(preferred, tmp_path / "out")
    assert result == preferred.resolve()


def test_acquire_traceconv_uses_path(tmp_path):
    with mock.patch("shutil.which", return_value="/usr/bin/traceconv"):
        result = acquire_traceconv(None, tmp_path)
    assert result == Path("/usr/bin/traceconv")


def test_ensure_trace_json_pftrace_converts(tmp_path):
    pftrace = tmp_path / "capture.pftrace"
    pftrace.write_bytes(b"PFTRACE")
    fake_traceconv = tmp_path / "traceconv"
    fake_traceconv.write_text("#!/usr/bin/env python3\n")
    fake_traceconv.chmod(0o755)

    def fake_run(cmd, cwd=None):
        out_json = Path(cmd[-1])
        out_json.write_text(json.dumps({"traceEvents": []}))

    with mock.patch("TraceLens.Reporting.pftrace_utils.run", side_effect=fake_run):
        result = ensure_trace_json(str(pftrace), traceconv_path=str(fake_traceconv))

    assert Path(result).parent == tmp_path / ".tracelens-cache"
    assert not pftrace.with_suffix(".json").exists()
    assert Path(result).exists()


def test_ensure_trace_json_missing_traceconv_raises(tmp_path):
    pftrace = tmp_path / "capture.pftrace"
    pftrace.write_bytes(b"PFTRACE")
    with pytest.raises(FileNotFoundError, match="traceconv not found"):
        ensure_trace_json(str(pftrace), traceconv_path=str(tmp_path / "missing"))


def test_conversion_preserves_existing_sibling_json_and_handles_gzip(tmp_path):
    import gzip

    pftrace = tmp_path / "capture.pftrace.gz"
    with gzip.open(pftrace, "wb") as stream:
        stream.write(b"PFTRACE")
    original = tmp_path / "capture.json"
    original.write_text("immutable source")
    converter = tmp_path / "traceconv.exe"
    converter.write_bytes(b"EXE")

    def fake_run(argv, **kwargs):
        assert Path(argv[-2]).read_bytes() == b"PFTRACE"
        Path(argv[-1]).write_text('{"traceEvents": []}')

    with mock.patch("TraceLens.Reporting.pftrace_utils.run", fake_run):
        result = ensure_trace_json(
            str(pftrace), str(converter), str(tmp_path / "derived")
        )
    assert original.read_text() == "immutable source"
    assert Path(result).parent == tmp_path / "derived"
    assert json.loads(Path(result).read_text()) == {"traceEvents": []}


def test_shell_converter_is_not_forced_through_python(tmp_path):
    import os

    pftrace = tmp_path / "source.pftrace"
    pftrace.write_bytes(b"PFTRACE")
    converter = tmp_path / "traceconv"
    converter.write_text('#!/bin/sh\nexec some-converter "$@"\n')

    def fake_run(argv, **kwargs):
        assert argv[0] == str(converter)
        Path(argv[-1]).write_text("{}")

    with mock.patch("TraceLens.Reporting.pftrace_utils.run", fake_run):
        if os.name == "nt":
            with pytest.raises(ValueError, match="Non-Python"):
                ensure_trace_json(str(pftrace), str(converter))
        else:
            ensure_trace_json(str(pftrace), str(converter))
