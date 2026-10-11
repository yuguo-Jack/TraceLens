###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DTK hipprof timeline adapter, qualified against DTK 26.10 JSON/pftrace.

This is not a rocprofv3 schema converter. API, dispatch and copy records remain
trace evidence; tensor shapes, PMC counters and training phases are not inferred.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from .pftrace_utils import ensure_trace_json
from .pftrace_hip_api_analysis import PftraceHipApiAnalyzer
from .generate_perf_report_pftrace_memory_copy import extract_memory_copy_rows
from .reporting_utils import write_report_outputs
from .rocprof_analysis import RocprofAnalyzer

_CATEGORIES = {"HIP": "hip_api", "HIPOPS": "kernel_dispatch", "HIPCOPY": "memory_copy"}
_COPY_NAMES = {
    "HostToDevice": "MEMORY_COPY_HOST_TO_DEVICE",
    "DeviceToHost": "MEMORY_COPY_DEVICE_TO_HOST",
    "DeviceToDevice": "MEMORY_COPY_DEVICE_TO_DEVICE",
}


def _integer(value, field):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError()
        return int(number)
    except Exception as exc:
        raise ValueError(f"Invalid hipprof {field}: {value!r}") from exc


def normalize_hipprof(data):
    """Return canonical Perfetto events plus a loss/accounting receipt.

    Correlation is scoped by process and hipprof index, never by name or nearest
    timestamp. Exact nanosecond fields are retained even for epoch timestamps.
    """
    events = data.get("traceEvents") if isinstance(data, dict) else None
    if not isinstance(events, list):
        raise ValueError("hipprof input must have traceEvents")
    # Perfetto's converter emits paired async b/e events for hipprof tracks.
    # Pair explicitly rather than counting both halves or silently losing them.
    selected, pairs = [], defaultdict(lambda: {"b": [], "e": []})
    for i, e in enumerate(events):
        if not isinstance(e, dict) or e.get("cat") not in _CATEGORIES:
            continue
        if e.get("ph") == "X":
            selected.append((i, e))
        elif e.get("ph") in {"b", "e"}:
            identity = e.get("id2", e.get("id"))
            if identity is None:
                raise ValueError("hipprof async event is missing its identity")
            key = (
                e.get("pid"),
                e.get("cat"),
                json.dumps(identity, sort_keys=True),
                e.get("name"),
            )
            pairs[key][e["ph"]].append((i, e))
        else:
            raise ValueError("Unsupported hipprof event phase: " + str(e.get("ph")))
    for pair in pairs.values():
        if len(pair["b"]) != len(pair["e"]):
            raise ValueError("Unpaired hipprof async begin/end events")
        # Converted Chrome times can be rounded to integer microseconds; an
        # end may precede its begin in file order for a sub-microsecond call.
        # Use exact begin-record EndNs for timing, and validate matching end
        # counts/times independently. Never reconstruct duration from rounded ts.
        beginnings = sorted(
            pair["b"],
            key=lambda item: _integer(item[1].get("args", {}).get("EndNs"), "EndNs"),
        )
        endings = sorted(pair["e"], key=lambda item: Decimal(str(item[1]["ts"])))
        for (source_i, start), (_, end) in zip(beginnings, endings):
            duration = _integer(start.get("args", {}).get("DurationNs"), "DurationNs")
            observed = (Decimal(str(end["ts"])) - Decimal(str(start["ts"]))) * 1000
            if abs(observed - duration) > 2000:
                raise ValueError(
                    "hipprof async end timestamp conflicts with exact duration"
                )
            selected.append((source_i, start))
    selected.sort(key=lambda item: item[0])
    if not selected:
        raise ValueError("No DTK hipprof HIP/HIPOPS/HIPCOPY complete events")
    output, counts, gaps, correlations, api_keys = [], Counter(), [], {}, Counter()
    for i, event in selected:
        a = event.get("args") or {}
        begin = _integer(a.get("BeginNs"), "BeginNs")
        end = _integer(a.get("EndNs"), "EndNs")
        duration = _integer(a.get("DurationNs", end - begin), "DurationNs")
        if duration < 0 or end - begin != duration:
            raise ValueError(f"Inconsistent hipprof duration at event {i}")
        # DTK's procId can be an autograd worker TID. pid is the shared process
        # identifier used by its runtime API records and correlation indices.
        pid = _integer(a.get("pid"), "pid")
        index = _integer(a.get("index"), "index")
        key = (pid, index)
        # A deterministic surrogate; the original process/index are retained.
        corr = correlations.setdefault(key, len(correlations) + 1)
        cat = event["cat"]
        args = {
            "begin_ns": begin,
            "end_ns": end,
            "delta_ns": duration,
            "corr_id": corr,
            "source_index": i,
            "hipprof_index": index,
            "source_pid": pid,
            "source_category": cat,
            "tracelens_source": "dtk-hipprof",
            "original_args": deepcopy(a),
        }
        name = str(event.get("name", ""))
        if not name:
            raise ValueError(f"Unnamed hipprof event {i}")
        if cat == "HIP":
            tid = _integer(a.get("tid"), "tid")
            api_keys[key] += 1
        else:
            device = _integer(a.get("devId"), "devId")
            tid = _integer(a.get("queueId", a.get("tid")), "queueId")
            args.update(
                agent=f"HCU:{device}", device_id=device, stream_ID=tid, queue=tid
            )
            if cat == "HIPOPS":
                for label, field in [("Grid", "grid"), ("Block", "block")]:
                    match = re.search(
                        label + r"=\{x=(\d+),y=(\d+),z=(\d+)\}", str(a.get("args", ""))
                    )
                    if match:
                        args[field] = [int(x) for x in match.groups()]
                        args["grid_size" if field == "grid" else "workgroup_size"] = (
                            int(match[1]) * int(match[2]) * int(match[3])
                        )
                compact = re.fullmatch(
                    r"\((\d+),(\d+),(\d+)\),\((\d+),(\d+),(\d+)\)",
                    str(a.get("args", "")),
                )
                if compact:
                    args["grid"] = [int(x) for x in compact.groups()[:3]]
                    args["block"] = [int(x) for x in compact.groups()[3:]]
            else:
                args["copy_bytes"] = _integer(a.get("Bytes"), "Bytes")
                if args["copy_bytes"] < 0:
                    raise ValueError("Negative copy byte count")
                args["original_name"] = name
                if name == "Memset":
                    args["memory_operation"] = "set"
                elif name in _COPY_NAMES:
                    args["memory_operation"] = "copy"
                    if name == "HostToDevice":
                        args["dst_agent"] = device
                    elif name == "DeviceToHost":
                        args["src_agent"] = device
                    # hipprof records a device, not necessarily both peer ends.
                    name = _COPY_NAMES[name]
                else:
                    gaps.append(f"Unknown memory direction at source event {i}")
        normalized_cat = (
            "memory_set" if cat == "HIPCOPY" and name == "Memset" else _CATEGORIES[cat]
        )
        output.append(
            {
                "ph": "X",
                "cat": normalized_cat,
                "name": name,
                "pid": pid,
                "tid": tid,
                "ts": begin,
                "dur": duration,
                "args": args,
            }
        )
        counts[normalized_cat] += 1
    if any(n != 1 for n in api_keys.values()):
        raise ValueError(
            "Duplicate hipprof API index within a process; correlation is ambiguous"
        )
    origin = min(e["ts"] for e in output)
    unlinked = sum(
        e["cat"] != "hip_api" and (e["pid"], e["args"]["hipprof_index"]) not in api_keys
        for e in output
    )
    if unlinked:
        gaps.append(
            f"{unlinked} device events lack an API record with the same process/index"
        )
    for e in output:
        # Canonical Chrome fields are relative microseconds. Exact ns above
        # avoid truncating sub-microsecond events and epoch precision loss.
        e["ts"] = (e["ts"] - origin) / 1000
        e["dur"] /= 1000
    unhandled = Counter(
        e.get("cat", "<none>")
        for e in events
        if isinstance(e, dict)
        and e.get("ph") == "X"
        and e.get("cat") not in _CATEGORIES
    )
    if unhandled:
        gaps.append(
            "Additional complete-event categories are not modeled: "
            + ", ".join(unhandled)
        )
    receipt = {
        "schema_version": 1,
        "provider": "DTK hipprof",
        "recognized_events": len(selected),
        "output_events": len(output),
        "counts": dict(counts),
        "unhandled_categories": dict(unhandled),
        "unlinked_device_events": unlinked,
        "origin_ns": str(origin),
        "status": "incomplete" if gaps else "converted",
        "gaps": gaps,
        "limits": [
            "No CPU tensor shape or training phase inference",
            "Observed event span is not a selected steady training window",
            "Timeline durations are not PMC or instruction counters",
            "No cross-process clock alignment is inferred",
        ],
    }
    return {"traceEvents": output, "hipprof_conversion": receipt}, receipt


def load_hipprof(path, traceconv_path=None):
    converted = ensure_trace_json(str(path), traceconv_path)
    opener = gzip.open if str(converted).endswith(".gz") else open
    with opener(converted, "rt", encoding="utf-8") as stream:
        return normalize_hipprof(json.load(stream))


def generate_perf_report_hipprof(trace_path, output_csvs_dir, traceconv_path=None):
    path = Path(trace_path).resolve()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    data, receipt = load_hipprof(path, traceconv_path)
    events = data["traceEvents"]
    tables = {}
    device_rows = []
    devices = sorted(
        {e["args"]["device_id"] for e in events if "device_id" in e["args"]}
    )
    kernel_rows = []
    for device in devices:
        selected = [e for e in events if e["args"].get("device_id") == device]

        def native(e):
            a = e["args"]
            return {
                "name": e["name"],
                "ts": a["begin_ns"] - int(receipt["origin_ns"]),
                "dur": a["delta_ns"],
                "grid": a.get("grid"),
                "block": a.get("block"),
                "stream": a.get("stream_ID"),
                "agent_id": device,
            }

        kernels = [native(e) for e in selected if e["cat"] == "kernel_dispatch"]
        memory = [
            native(e) for e in selected if e["cat"] in {"memory_copy", "memory_set"}
        ]
        all_native = kernels + memory
        if not all_native:
            continue
        metadata = {
            "init_time": min(e["ts"] for e in all_native),
            "fini_time": max(e["ts"] + e["dur"] for e in all_native),
        }
        analyzer = RocprofAnalyzer(kernels, memory, [], metadata)
        timeline = analyzer.get_df_gpu_timeline()
        if len(timeline):
            timeline.insert(0, "device", device)
            device_rows.append(timeline)
        summary = analyzer.get_df_kernel_summary()
        summary.insert(0, "device", device)
        kernel_rows.append(summary)
    tables["gpu_timeline"] = (
        pd.concat(device_rows, ignore_index=True) if device_rows else pd.DataFrame()
    )
    tables["kernel_summary"] = (
        pd.concat(kernel_rows, ignore_index=True) if kernel_rows else pd.DataFrame()
    )
    analyzer = PftraceHipApiAnalyzer(events, allow_multi_kernel_per_api=True)
    tables["api_kernel_summary"] = analyzer.get_df_api_kernel_summary()
    copies = extract_memory_copy_rows(events)
    tables["memory_copy_summary"] = pd.DataFrame(
        [
            {"copy_bytes": size, "direction": direction, "count": count}
            for (size, direction), count in Counter(copies).items()
        ]
    )
    tables["memory_set_summary"] = pd.DataFrame(
        [
            {
                "device": e["args"]["device_id"],
                "bytes": e["args"]["copy_bytes"],
                "duration_ns": e["args"]["delta_ns"],
            }
            for e in events
            if e["cat"] == "memory_set"
        ]
    )
    tables["hip_api_summary"] = pd.DataFrame(
        [
            {"name": e["name"], "pid": e["pid"], "duration_ns": e["args"]["delta_ns"]}
            for e in events
            if e["cat"] == "hip_api"
        ]
    )
    if hashlib.sha256(path.read_bytes()).hexdigest() != before:
        raise ValueError("hipprof input changed during analysis")
    output = Path(output_csvs_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_report_outputs(tables, csvs_dir=str(output))
    receipt.update(
        source=str(path),
        source_sha256=before,
        table_rows={key: len(df) for key, df in tables.items()},
    )
    (output / "qualification.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    (output / "normalized.json").write_text(json.dumps(data), encoding="utf-8")
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-path", required=True)
    parser.add_argument("--output-csvs-dir", required=True)
    parser.add_argument("--traceconv-path")
    args = parser.parse_args()
    generate_perf_report_hipprof(
        args.trace_path, args.output_csvs_dir, args.traceconv_path
    )


if __name__ == "__main__":
    main()
