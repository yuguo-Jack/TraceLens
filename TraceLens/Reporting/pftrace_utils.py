###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Shared utilities for Perfetto-style trace handling: traceconv resolution
and .pftrace → JSON conversion. Used by generate_perf_report_pftrace_hip_api
and generate_perf_report_pftrace_hip_activity.
"""

import logging
import gzip
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import uuid
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    """Run command; raise on non-zero exit."""
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr[-2000:]}")


def acquire_traceconv(preferred: Optional[Path], out_dir: Path) -> Path:
    """
    Resolve a usable traceconv executable:
      1) use preferred path if provided and exists
      2) find traceconv on PATH
      3) download into out_dir (curl, then chmod; fallback to Python urllib)
    """
    if preferred:
        p = preferred.resolve()
        if p.exists():
            return p
        logger.warning("--traceconv provided but not found at %s; continuing.", p)

    on_path = shutil.which("traceconv")
    if on_path:
        return Path(on_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "traceconv"
    logger.info("traceconv not found; downloading to %s", target)
    try:
        run(["curl", "-LO", "https://get.perfetto.dev/traceconv"], cwd=out_dir)
        run(["chmod", "+x", "traceconv"], cwd=out_dir)
        if not target.exists():
            raise RuntimeError("curl reported success but traceconv not found.")
        return target
    except Exception as e:
        logger.warning("curl flow failed (%s). Falling back to Python download...", e)
        urllib.request.urlretrieve("https://get.perfetto.dev/traceconv", target)
        target.chmod(0o755)
        return target


def ensure_trace_json(
    trace_path: str,
    traceconv_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    Return path to JSON trace. If trace_path is .pftrace, use traceconv_path if
    provided, otherwise resolve or download traceconv (see acquire_traceconv),
    then convert to JSON. For .json/.json.gz return path as-is.
    """
    path = Path(trace_path).resolve()
    suffix = path.suffix.lower()
    if suffix == ".pftrace" or path.name.lower().endswith(".pftrace.gz"):
        cache = Path(
            output_dir
            or os.environ.get("TRACELENS_CACHE_DIR")
            or path.parent / ".tracelens-cache"
        ).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        if traceconv_path:
            traceconv = Path(traceconv_path).resolve()
            if not traceconv.exists():
                raise FileNotFoundError(f"traceconv not found: {traceconv}")
        else:
            traceconv = acquire_traceconv(None, cache / "tools")
        # Conversion is derived state. Never overwrite a separately collected
        # JSON file beside its pftrace source, even when the base names match.
        digest = hashlib.sha256()
        for source in (path, traceconv):
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        key = digest.hexdigest()
        out_json = cache / (key + ".json")
        raw = path
        if path.name.lower().endswith(".pftrace.gz"):
            raw = cache / (key + ".pftrace")
            with gzip.open(path, "rb") as src, raw.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        temporary = cache / (key + "." + uuid.uuid4().hex + ".json")
        logger.info("Converting .pftrace to JSON: %s -> %s", path, out_json)
        prefix = [str(traceconv)]
        with traceconv.open("rb") as stream:
            shebang = stream.readline(256).decode("utf-8", errors="replace")
        python_script = traceconv.suffix == ".py" or (
            shebang.startswith("#!") and "python" in shebang.lower()
        )
        if python_script:
            prefix.insert(0, sys.executable)
        elif shebang.startswith("#!") and os.name == "nt":
            raise ValueError(
                "Non-Python traceconv script is not executable on Windows; provide the native binary or Python launcher"
            )
        # The current official binary also accepts the traceconv alias. A
        # trace_processor-named executable needs its explicit convert command.
        if traceconv.stem.startswith("trace_processor"):
            prefix.append("convert")
        try:
            run([*prefix, "json", str(raw), str(temporary)])
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("Converter produced no JSON output")
            os.replace(temporary, out_json)
        finally:
            temporary.unlink(missing_ok=True)
        return str(out_json)
    if suffix == ".json" or path.name.lower().endswith(".json.gz"):
        return str(path)
    raise ValueError(
        f"Unsupported trace format: {trace_path}. Use .json, .json.gz, or .pftrace."
    )


def derive_pftrace_output_path(trace_path: str, report_suffix: str) -> str:
    """Derive a default output xlsx path from a pftrace input path.

    Strips ``.pftrace``, ``.json.gz``, or other extensions from
    *trace_path* and appends *report_suffix* (e.g.
    ``"_pftrace_activity_report.xlsx"``).
    """
    base = Path(trace_path).resolve()
    if base.suffix.lower() == ".pftrace":
        base = base.with_suffix("")
    elif base.suffix.lower() == ".gz" and base.name.endswith(".json.gz"):
        base = base.parent / base.name.replace(".json.gz", "")
    else:
        base = base.with_suffix("")
    return str(base) + report_suffix
