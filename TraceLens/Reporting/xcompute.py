###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Read native XComputeLight counter and SQTT CSVs without a trace conversion."""

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path
import re

# Constants
DISPATCH_COLUMNS = ["Dispatch id", "Kernel", "XCD", "Duration [ms]"]
SQTT_COLUMNS = {
    "sqtt_dispatch_summary": [
        "Dispatch id",
        "Kernel",
        "XCDs",
        "SEs",
        "CUs",
        "Waves",
        "Complete Waves",
        "Incomplete Waves",
        "Inst Issues",
        "Duration",
        "No-wave Idle",
        "Idle %",
        "Max Active Waves",
        "Avg Active Waves",
    ],
    "sqtt_hot_inst": [
        "Dispatch id",
        "Kernel",
        "Group",
        "Opcode",
        "Hits",
        "Latency",
        "Latency %",
    ],
    "sqtt_bubble_summary": [
        "Dispatch id",
        "Kernel",
        "Type",
        "Before",
        "After",
        "Count",
        "Duration",
        "Duration %",
        "Max Duration",
        "Example XCD",
        "Example SE",
        "Example SH",
        "Example CU",
        "Example SIMD",
        "Example WaveSlot",
        "Example Source",
    ],
}
METRIC_FILENAME = re.compile(r"^(\d+)_(.+)\.csv$")
SHA256 = re.compile(r"[0-9a-fA-F]{64}")
SQTT_TEXT_COLUMNS = {
    "Kernel",
    "Group",
    "Opcode",
    "Type",
    "Before",
    "After",
    "Example Source",
}
LIMITS = [
    "Counter replay and SQTT collection may change cache, concurrency and dispatch time",
    "Dispatch Duration [ms] is not profiler-off operator or training-step latency",
    "SQTT Duration/Latency units and sampled coverage are not inferred",
    "Metrics do not establish a physical bottleneck or a matched hardware roofline",
    "No tensor shape, training phase, rank, source event or cross-capture join is inferred",
]


def parse_xcompute_csv(path):
    """Return exact sections or native table columns, with an input fingerprint.

    Counter values are finite numbers or None for literal N/A. Native table cells
    remain strings or None; SQTT timing columns are never converted to durations.
    """
    path = Path(path).resolve()
    raw = path.read_bytes()
    rows = []
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    for row in reader:
        if row and any(cell.strip() for cell in row):
            rows.append((reader.line_num, row))
    if not rows:
        raise ValueError(f"Empty XCompute CSV: {path}")
    header = rows[0][1]
    table_kind = next(
        (kind for kind, cols in SQTT_COLUMNS.items() if header == cols), None
    )
    if header == DISPATCH_COLUMNS:
        table_kind = "dispatch_summary"
    if table_kind:
        result = _parse_table(rows, table_kind)
    elif len(header) == 1:
        result = _parse_metrics(rows)
    else:
        raise ValueError(f"Unsupported XCompute CSV header: {header!r}")
    result["source"] = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    return result


def generate_perf_report_xcompute(manifest_path, output_dir):
    """Parse capture groups and save report.json; no GPU work or tool invocation.

    The manifest declares capture membership and provenance. Dispatch binding is
    exact and capture-local; kernel names alone never spread counters to calls.
    """
    manifest_path = Path(manifest_path).resolve()
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("XCompute manifest requires schema_version 1")
    captures = manifest.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ValueError("XCompute manifest requires nonempty captures")
    root = manifest_path.parent
    gaps, output, seen = [], [], set()
    for capture in captures:
        if not isinstance(capture, dict):
            raise ValueError("Each XCompute capture must be an object")
        identity = capture.get("id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("Capture ids must be nonempty and unique")
        seen.add(identity)
        provenance = _provenance(capture, root, identity, gaps)
        files = capture.get("csvs")
        if not isinstance(files, list) or not files:
            raise ValueError(f"Capture {identity} requires nonempty csvs")
        parsed, descriptors, inputs = [], [], set()
        for entry in files:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError("Each CSV entry requires a path")
            path = (root / entry["path"]).resolve()
            if path in inputs:
                raise ValueError(f"Duplicate CSV in capture {identity}: {path}")
            inputs.add(path)
            result = parse_xcompute_csv(path)
            if entry.get("kind") is not None and entry["kind"] != result["kind"]:
                raise ValueError(f"CSV kind does not match its native header: {path}")
            expected = _hash_value(entry.get("sha256"), "CSV sha256")
            if expected is not None and expected != result["source"]["sha256"]:
                raise ValueError(f"CSV hash mismatch: {path}")
            result["source"]["expected_sha256"] = expected
            result["source"]["hash_verified"] = expected is not None
            result["manifest_entry"] = entry
            if expected is None:
                gaps.append(f"{identity}: CSV hash was not declared: {path.name}")
            parsed.append(result)
            descriptors.append(entry)
        summaries = [p for p in parsed if p["kind"] == "dispatch_summary"]
        if len(summaries) > 1:
            raise ValueError(f"Capture {identity} has multiple dispatch summaries")
        dispatches = summaries[0]["dispatches"] if summaries else []
        for result, entry in zip(parsed, descriptors):
            if result["kind"] == "metrics":
                result["dispatch_binding"] = _bind_metrics(
                    result, entry, dispatches, identity, provenance["source_perf"], gaps
                )
        output.append(
            {
                "id": identity,
                "provenance": provenance,
                "files": parsed,
                "dispatch_scope": "capture id + source perf fingerprint + dispatch id + kernel + XCD",
            }
        )
    if (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError("XCompute manifest changed during analysis")
    report = {
        "schema_version": 1,
        "provider": "XComputeLight native CSV",
        "status": "incomplete" if gaps else "parsed",
        "manifest": {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "captures": output,
        "gaps": gaps,
        "limits": LIMITS,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "report.json"
    if target.exists():
        raise ValueError(f"Refusing to overwrite XCompute report: {target}")
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = generate_perf_report_xcompute(args.manifest, args.output_dir)
    print(
        json.dumps(
            {
                "status": report["status"],
                "captures": len(report["captures"]),
                "gaps": report["gaps"],
                "report": str(Path(args.output_dir) / "report.json"),
            }
        )
    )


def _number(value, label, integer=False):
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid XCompute {label}: {value!r}") from exc
    if isinstance(value, bool) or not number.is_finite():
        raise ValueError(f"Invalid XCompute {label}: {value!r}")
    if integer:
        if number < 0 or number != number.to_integral_value():
            raise ValueError(f"XCompute {label} must be a nonnegative integer")
        return int(number)
    if number == number.to_integral_value():
        return int(number)
    result = float(number)
    if not math.isfinite(result) or (result == 0 and number != 0):
        raise ValueError(f"XCompute {label} is outside finite numeric range")
    return result


def _parse_metrics(rows):
    metrics, sections, section = [], [], None
    for line, row in rows:
        if len(row) == 1:
            if not row[0].strip():
                raise ValueError(f"Empty XCompute section at line {line}")
            section = row[0]
            sections.append(section)
        elif len(row) == 3 and section is not None and row[0].strip():
            metrics.append(
                {
                    "section": section,
                    "section_index": len(sections) - 1,
                    "name": row[0],
                    "unit": row[1],
                    "raw_value": row[2],
                    "line": line,
                    "value": (
                        None if row[2] == "N/A" else _number(row[2], "metric value")
                    ),
                }
            )
        else:
            raise ValueError(
                f"Expected section or name/unit/value at line {line}: {row!r}"
            )
    if not metrics:
        raise ValueError("XCompute metric CSV contains no metrics")
    return {"kind": "metrics", "sections": sections, "metrics": metrics}


def _parse_table(rows, kind):
    columns = rows[0][1]
    values, identities, dispatches, seen = [], [], [], set()
    for line, row in rows[1:]:
        if len(row) != len(columns):
            raise ValueError(f"XCompute {kind} column count mismatch at line {line}")
        record = dict(zip(columns, (None if x == "N/A" else x for x in row)))
        dispatch_id = _number(record["Dispatch id"], "dispatch id", integer=True)
        kernel = record["Kernel"]
        if not kernel or not kernel.strip():
            raise ValueError(f"Missing XCompute kernel at line {line}")
        identity = {"dispatch_id": dispatch_id, "kernel": kernel}
        if kind != "dispatch_summary":
            for column, value in record.items():
                if column not in SQTT_TEXT_COLUMNS and value is not None:
                    if _number(value, column) < 0:
                        raise ValueError(f"Negative XCompute {column} at line {line}")
        if kind == "dispatch_summary":
            xcd = (
                None
                if record["XCD"] is None
                else _number(record["XCD"], "XCD", integer=True)
            )
            key = (dispatch_id, kernel, xcd)
            if key in seen:
                raise ValueError(f"Duplicate XCompute dispatch/XCD identity: {key}")
            seen.add(key)
            duration = record["Duration [ms]"]
            ns = None
            ns_decimal = None
            if duration is not None:
                _number(duration, "duration")
                parts = Decimal(duration).as_tuple()
                converted = Decimal((parts.sign, parts.digits, parts.exponent + 6))
                ns = _number(converted, "duration ns")
                ns_decimal = str(converted)
            if ns is not None and ns < 0:
                raise ValueError("Negative XCompute dispatch duration")
            identity["xcd"] = xcd
            dispatches.append(
                {
                    **identity,
                    "duration_ns": ns,
                    "duration_ns_decimal": ns_decimal,
                    "source_line": line,
                }
            )
        values.append(record)
        identities.append(identity)
    if not values:
        raise ValueError(f"XCompute {kind} contains no rows")
    result = {
        "kind": kind,
        "columns": columns,
        "rows": values,
        "row_identities": identities,
    }
    if kind == "dispatch_summary":
        result["dispatches"] = dispatches
        result["duration_basis"] = (
            "CSV Duration [ms], converted to ns; instrumented dispatch timing"
        )
    else:
        result["timing_units"] = {
            c: None
            for c in columns
            if c in {"Duration", "Latency", "Max Duration", "No-wave Idle"}
        }
        result["scope"] = (
            "Native SQTT sampled table; no single-kernel scalar or time conversion"
        )
    return result


def _hash_value(value, label):
    if value is None:
        return None
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a SHA256 hex string or null")
    return value.lower()


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(capture, root, identity, gaps):
    source = capture.get("source_perf") or {}
    if not isinstance(source, dict):
        raise ValueError("source_perf must be an object or null")
    expected = _hash_value(source.get("sha256"), "source perf sha256")
    name = source.get("path")
    if name is not None and (not isinstance(name, str) or not name):
        raise ValueError("source_perf.path must be a path or null")
    path = (root / name).resolve() if name else None
    actual = _file_hash(path) if path and path.is_file() else None
    if actual is not None and expected is not None and actual != expected:
        raise ValueError(f"Source perf hash mismatch: {path}")
    perf = {
        "path": name,
        "resolved_path": str(path) if path else None,
        "sha256": expected,
        "observed_sha256": actual,
        "verification": (
            "verified"
            if actual is not None and expected is not None
            else "declared-only" if expected is not None else "unknown"
        ),
    }
    if perf["verification"] != "verified":
        gaps.append(f"{identity}: source perf provenance is {perf['verification']}")
    result = {"source_perf": perf}
    types = {
        "tools": dict,
        "gfx": str,
        "environment": dict,
        "commands": list,
        "metric_basis": dict,
        "sampling": dict,
        "workload": dict,
    }
    for field in [
        "tools",
        "gfx",
        "environment",
        "commands",
        "counter_replay",
        "metric_basis",
        "sampling",
        "workload",
    ]:
        result[field] = capture.get(field)
        if (
            field in types
            and result[field] is not None
            and not isinstance(result[field], types[field])
        ):
            raise ValueError(f"{field} has an invalid provenance type")
        if field in {
            "tools",
            "gfx",
            "commands",
            "counter_replay",
            "metric_basis",
            "sampling",
        } and (
            _has_unknown(result[field])
            or (field != "counter_replay" and not result[field])
        ):
            gaps.append(f"{identity}: {field} is unknown")
    if result["counter_replay"] is not None and not isinstance(
        result["counter_replay"], bool
    ):
        raise ValueError("counter_replay must be true, false or null")
    return result


def _has_unknown(value):
    if isinstance(value, dict):
        return any(_has_unknown(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_unknown(item) for item in value)
    return value is None


def _bind_metrics(result, entry, dispatches, capture_id, source_perf, gaps):
    match = METRIC_FILENAME.fullmatch(Path(result["source"]["path"]).name)
    candidate = {"dispatch_id": int(match[1]), "kernel": match[2]} if match else {}
    for field in ["dispatch_id", "kernel"]:
        if field in entry:
            value = (
                _number(entry[field], field, integer=True)
                if field == "dispatch_id"
                else entry[field]
            )
            if field == "kernel" and (not isinstance(value, str) or not value.strip()):
                raise ValueError("Manifest kernel must be a nonempty string")
            if field in candidate and candidate[field] != value:
                raise ValueError(f"Metric filename and manifest {field} disagree")
            candidate[field] = value
    xcd = (
        _number(entry["xcd"], "XCD", integer=True)
        if entry.get("xcd") is not None
        else None
    )
    matches = [
        d
        for d in dispatches
        if d["dispatch_id"] == candidate.get("dispatch_id")
        and d["kernel"] == candidate.get("kernel")
        and (xcd is None or d["xcd"] == xcd)
    ]
    if len(matches) != 1:
        gaps.append(
            f"{capture_id}: metrics have no unique dispatch binding: {Path(result['source']['path']).name}"
        )
        return {
            "status": "unlinked",
            "candidate": candidate,
            "requested_xcd": xcd,
            "matches": len(matches),
        }
    return {
        "status": "linked",
        "capture_id": capture_id,
        "source_perf_sha256": source_perf["sha256"],
        **matches[0],
        "method": "filename+unique-summary" if match else "manifest+unique-summary",
        "workload_binding": "caller provenance only; shape/phase/source event not inferred",
    }


if __name__ == "__main__":
    main()
