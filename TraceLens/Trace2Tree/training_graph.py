###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Strict capture-to-replay attribution for explicitly annotated training graphs.

This maps captured host structure to measured replay GPU events. It does not
reconstruct training state, captured tensor values, or replay CPU timings.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

from ..util import TraceEventUtils

SCHEMA = "tracelens.training_graph.v1"
ANNOTATION = re.compile(
    r"^tracelens\.train_graph\.(capture|replay):([A-Za-z0-9_-]+):([A-Za-z0-9_-]+)$"
)
UID = TraceEventUtils.TraceKeys.UID
GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
RUNTIME_CATEGORIES = TraceEventUtils.RuntimeCategories


class TrainingGraphContractError(ValueError):
    """An incomplete or ambiguous capture/replay mapping; never a partial pass."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_annotations(events, kind="replay"):
    result = []
    for event in events:
        name = event.get("name", "")
        # GPU annotations duplicate the host labels and have different clocks
        # and ownership. They cannot define a host graph-launch contract.
        if event.get("cat") not in {"cpu_op", "user_annotation"}:
            continue
        match = ANNOTATION.fullmatch(name)
        if name.startswith(f"tracelens.train_graph.{kind}:") and match is None:
            raise TrainingGraphContractError(
                f"Malformed training graph annotation: {name}"
            )
        if match and match[1] == kind:
            if event.get("ph", "X") != "X" or event.get("dur", 0) <= 0:
                raise TrainingGraphContractError(
                    "Training graph annotations must be nonempty complete host ranges"
                )
            result.append((event, (match[2], match[3])))
    return sorted(result, key=lambda pair: pair[0]["ts"])


def _checked_file(folder, filename, digest):
    if not isinstance(filename, str) or not filename or Path(filename).is_absolute():
        raise TrainingGraphContractError(
            "Manifest filenames must be relative to their bundle directory"
        )
    root = Path(folder).resolve()
    path = (root / filename).resolve()
    if root not in path.parents:
        raise TrainingGraphContractError("Manifest file escapes its bundle directory")
    if (
        not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or sha256_file(path) != digest
    ):
        raise TrainingGraphContractError(
            f"Trace digest mismatch or missing SHA256: {filename}"
        )
    return path


def _signature(signature):
    if not isinstance(signature, list) or not signature:
        raise TrainingGraphContractError(
            "An explicit nonempty runtime input_signature is required"
        )
    names = set()
    for item in signature:
        if not isinstance(item, dict) or set(item) != {
            "name",
            "shape",
            "dtype",
            "stride",
        }:
            raise TrainingGraphContractError(
                "Signature tensors need exactly name, shape, dtype and stride"
            )
        if (
            not isinstance(item["name"], str)
            or not item["name"]
            or item["name"] in names
        ):
            raise TrainingGraphContractError(
                "Signature tensor names must be unique nonempty strings"
            )
        names.add(item["name"])
        if not isinstance(item["dtype"], str) or not item["dtype"].startswith("torch."):
            raise TrainingGraphContractError(
                "Signature dtypes must be explicit torch dtype names"
            )
        if (
            not isinstance(item["shape"], list)
            or not isinstance(item["stride"], list)
            or len(item["shape"]) != len(item["stride"])
        ):
            raise TrainingGraphContractError("Signature shape/stride ranks must agree")
        if any(
            not isinstance(x, int) or isinstance(x, bool) or x < 0
            for x in item["shape"] + item["stride"]
        ):
            raise TrainingGraphContractError(
                "Signature shapes/strides must be nonnegative integers"
            )
    return signature


def _inside(event, root, same_thread=False):
    return (
        event.get("pid") == root.get("pid")
        and (not same_thread or event.get("tid") == root.get("tid"))
        and root["ts"] <= event.get("ts", -1)
        and event.get("ts", -1) + event.get("dur", 0) <= root["ts"] + root["dur"]
    )


def _capture_segment(tree, annotation):
    begins = [
        e
        for e in tree.events
        if e.get("name") in {"hipStreamBeginCapture", "cudaStreamBeginCapture"}
        and e.get("cat") in RUNTIME_CATEGORIES
        and _inside(e, annotation, True)
    ]
    ends = [
        e
        for e in tree.events
        if e.get("name") in {"hipStreamEndCapture", "cudaStreamEndCapture"}
        and e.get("cat") in RUNTIME_CATEGORIES
        and _inside(e, annotation, True)
    ]
    if len(begins) != 1 or len(ends) != 1:
        raise TrainingGraphContractError(
            "Each capture annotation must enclose exactly one StreamBeginCapture/StreamEndCapture pair"
        )
    begin, end = begins[0], ends[0]
    start, stop = begin["ts"] + begin["dur"], end["ts"]
    if start >= stop:
        raise TrainingGraphContractError("Capture begin/end are not ordered")
    capture_stream = begin.get("args", {}).get("stream")
    if (
        capture_stream is not None
        and end.get("args", {}).get("stream") != capture_stream
    ):
        raise TrainingGraphContractError("Capture begin/end stream identities differ")
    window = {"ts": start, "dur": stop - start, "pid": annotation["pid"]}
    host_categories = {"cpu_op", "python_function"} | set(RUNTIME_CATEGORIES)
    events = [
        copy.deepcopy(e)
        for e in tree.events
        if e.get("cat") in host_categories and _inside(e, window)
    ]
    dispatches = [
        e
        for e in events
        if e.get("cat") in RUNTIME_CATEGORIES
        and any(
            word in e.get("name", "") for word in ("LaunchKernel", "Memcpy", "Memset")
        )
    ]
    dispatches.sort(key=lambda e: e["ts"])
    if not dispatches:
        raise TrainingGraphContractError("Capture region has no recorded dispatches")
    # A capture window spans autograd worker threads. A host annotation's
    # children alone would silently omit these backward operations.
    selected = {e[UID] for e in events}
    roots = [e for e in events if e.get("parent") not in selected]
    for e in events:
        e["children"] = [u for u in e.get("children", []) if u in selected]
        e.pop("gpu_events", None)
    root_uid = max(tree.events_by_uid, default=-1) + 1
    root = {
        UID: root_uid,
        "ph": "X",
        "cat": "cpu_op",
        "name": annotation["name"],
        "ts": start,
        "dur": stop - start,
        "pid": annotation["pid"],
        "tid": annotation["tid"],
        "args": {},
        "children": [e[UID] for e in roots],
        "capture_annotation_event_index": annotation[UID],
    }
    for e in roots:
        e["parent"] = root_uid
    events.sort(key=lambda e: e["ts"])
    return [root] + events, dispatches


def _identity(event, capture=False):
    name = event.get("name", "")
    if capture:
        if "Memcpy" in name:
            return "gpu_memcpy", None
        if "Memset" in name:
            return "gpu_memset", None
        name = event.get("args", {}).get("kernel")
        if not isinstance(name, str) or not name:
            raise TrainingGraphContractError(
                "capture_kernel_identity_missing: every compute dispatch needs a recorded args.kernel"
            )
        return "kernel", name
    return event["cat"], name if event["cat"] == "kernel" else None


def _verify_mapping(dispatches, kernels):
    if len(dispatches) != len(kernels) or not kernels:
        raise TrainingGraphContractError(
            f"Capture/replay dispatch counts differ: {len(dispatches)} versus {len(kernels)}"
        )
    streams = {e.get("args", {}).get("stream") for e in kernels}
    if len(streams) != 1 or None in streams:
        raise TrainingGraphContractError(
            "Only proven serialized_single_stream replay is supported; multistream mapping is ambiguous"
        )
    for events, label in ((dispatches, "capture dispatch"), (kernels, "replay GPU")):
        for previous, current in zip(events, events[1:]):
            if previous["ts"] + previous.get("dur", 0) > current["ts"] + 0.001:
                raise TrainingGraphContractError(
                    f"Overlapping {label} events violate serialized_single_stream"
                )
    for capture, replay in zip(dispatches, kernels):
        if _identity(capture, True) != _identity(replay):
            raise TrainingGraphContractError(
                "Capture/replay ordered kernel identity or memory-operation class differs; no events may be skipped or reordered"
            )


def merge_training_capture_trace(capture_folder, manifest_path, replay_path=None):
    """Return an augmented tree only after *all* replay instances validate.

    The sidecar binds exact trace bytes, capture IDs, runtime shape/layout
    signatures and one explicit serialized capture contract. Validation never
    rounds batch sizes or selects capture files by approximate kernel names.
    """
    from .trace_capture_merge_experimental import (
        _load_trace_tree_from_file,
        make_connections,
        finalize_non_gpu_paths,
    )

    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise TrainingGraphContractError(f"Expected manifest schema {SCHEMA}")
    if manifest.get("mapping_contract") != "serialized_single_stream":
        raise TrainingGraphContractError(
            "An explicit serialized_single_stream mapping_contract is required"
        )
    provenance = manifest.get("provenance", {})
    if not isinstance(provenance, dict):
        raise TrainingGraphContractError("Manifest provenance must be an object")
    for key in ("torch_version", "hip_version", "device_type", "signature_source"):
        if not isinstance(provenance.get(key), str) or not provenance[key]:
            raise TrainingGraphContractError(f"Missing provenance field: {key}")
    if (
        provenance["device_type"] != "hcu"
        or provenance["signature_source"] != "runtime_tensor_metadata"
    ):
        raise TrainingGraphContractError(
            "HCU and runtime_tensor_metadata provenance must be explicit"
        )
    replay_file = _checked_file(
        capture_folder, manifest.get("replay_file"), manifest.get("replay_sha256")
    )
    if replay_path is not None and Path(replay_path).resolve() != replay_file:
        raise TrainingGraphContractError(
            "Requested replay trace differs from the manifest"
        )
    captures = manifest.get("captures")
    if not isinstance(captures, list) or not captures:
        raise TrainingGraphContractError("Manifest has no capture contracts")
    capture_map, loaded = {}, {}
    for record in captures:
        if not isinstance(record, dict):
            raise TrainingGraphContractError("Each capture contract must be an object")
        key = (record.get("capture_id"), record.get("segment_id"))
        if (
            any(
                not isinstance(s, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", s)
                for s in key
            )
            or key in capture_map
        ):
            raise TrainingGraphContractError(
                "Capture IDs must be explicit, valid and unique"
            )
        if record.get("phase") not in {"forward", "backward", "train_step"}:
            raise TrainingGraphContractError(
                "Capture phase must be forward, backward or train_step"
            )
        if _signature(record.get("input_signature")) != _signature(
            record.get("replay_input_signature")
        ):
            raise TrainingGraphContractError(
                "Capture and replay input signatures differ"
            )
        file = _checked_file(capture_folder, record.get("file"), record.get("sha256"))
        if file not in loaded:
            loaded[file] = _load_trace_tree_from_file(str(file), add_python_func=True)
        tree = loaded[file]
        matching = [
            a for a, k in training_annotations(tree.events, "capture") if k == key
        ]
        if len(matching) != 1:
            raise TrainingGraphContractError(
                f"Capture annotation is missing or ambiguous: {key}"
            )
        events, dispatches = _capture_segment(tree, matching[0])
        capture_map[key] = (record, events, dispatches)
    graph = _load_trace_tree_from_file(str(replay_file), add_python_func=True)
    if graph.trace_metadata.get("tracelens_training_graph_merge"):
        raise TrainingGraphContractError("Replay trace is already augmented")
    annotations = training_annotations(graph.events)
    if not annotations:
        raise TrainingGraphContractError(
            "Replay trace contains no explicit training graph annotations"
        )
    launches = [
        e
        for e in graph.events
        if e.get("cat") in RUNTIME_CATEGORIES
        and e.get("name") in {"cudaGraphLaunch", "hipGraphLaunch"}
    ]
    covered, plans, mapping_report = set(), [], []
    for annotation, key in annotations:
        if key not in capture_map:
            raise TrainingGraphContractError(
                f"No exact capture contract for replay annotation: {key}"
            )
        owned = [e for e in launches if _inside(e, annotation, True)]
        if len(owned) != 1 or owned[0][UID] in covered:
            raise TrainingGraphContractError(
                "Each replay annotation must own exactly one distinct graph launch on its host thread"
            )
        launch = owned[0]
        covered.add(launch[UID])
        record, events, dispatches = capture_map[key]
        kernels = sorted(graph.get_gpu_events(launch), key=lambda e: e["ts"])
        _verify_mapping(dispatches, kernels)
        plans.append((record, events, dispatches, launch, kernels))
        mapping_report.append(
            {
                "capture_id": key[0],
                "segment_id": key[1],
                "phase": record["phase"],
                "replay_annotation_uid": annotation[UID],
                "graph_launch_uid": launch[UID],
                "dispatch_count": len(dispatches),
                "kernel_count": sum(e["cat"] == "kernel" for e in kernels),
                "gpu_duration_sum_us": sum(e["dur"] for e in kernels),
            }
        )
    if covered != {e[UID] for e in launches}:
        raise TrainingGraphContractError(
            "Unannotated graph launches exist; the mapping would be partial"
        )
    if {k for _, k in annotations} != set(capture_map):
        raise TrainingGraphContractError(
            "Unused capture contracts exist; the bundle is not an exact mapping"
        )
    # Nothing has been attached before this point. Rejecting any instance cannot
    # expose a half-augmented tree to callers.
    for record, original_events, original_dispatches, launch, kernels in plans:
        events = copy.deepcopy(original_events)
        by_old_uid = {e[UID]: e for e in events}
        dispatches = [by_old_uid[e[UID]] for e in original_dispatches]
        uid_map = {e[UID]: len(graph.events) + i for i, e in enumerate(events)}
        for e in events:
            origin_uid = e[UID]
            e["capture_origin"] = {
                "file": record["file"],
                "sha256": record["sha256"],
                "uid": None if "capture_annotation_event_index" in e else origin_uid,
                "annotation_event_index": e.get("capture_annotation_event_index"),
                "ts": e["ts"],
                "dur": e.get("dur", 0),
                "capture_id": record["capture_id"],
                "segment_id": record["segment_id"],
            }
            e[UID] = uid_map[origin_uid]
            e["children"] = [uid_map[u] for u in e.get("children", [])]
            e["parent"] = uid_map.get(e.get("parent"), launch[UID])
            if "bwd_events" in e:
                outside = [u for u in e["bwd_events"] if u not in uid_map]
                e["bwd_events"] = [uid_map[u] for u in e["bwd_events"] if u in uid_map]
                if outside:
                    e["capture_relationships_outside_segment"] = {"bwd_events": outside}
            if "fwd_event" in e:
                if e["fwd_event"] in uid_map:
                    e["fwd_event"] = uid_map[e["fwd_event"]]
                else:
                    e.setdefault("capture_relationships_outside_segment", {})[
                        "fwd_event"
                    ] = e.pop("fwd_event")
            e["ts"], e["dur"] = launch["ts"], 0
            e[TraceEventUtils.TraceKeys.TimeEnd] = e["ts"]
            e["timing_origin"] = "capture_structure_only"
            graph.events.append(e)
            graph.events_by_uid[e[UID]] = e
            graph.name2event_uids[e["name"]].append(e[UID])
            if e["cat"] == "cpu_op":
                (
                    graph.cpu_root_nodes.append(e[UID])
                    if e["parent"] == launch[UID]
                    else None
                )
        launch["children"] = [events[0][UID]]
        # The raw tree already propagated these kernels to graph-launch
        # ancestors. Rebuild that ancestry once through the captured structure
        # so inclusive GPU accounting cannot count every event twice.
        kernel_uids = {e[UID] for e in kernels}
        ancestor = launch
        while ancestor is not None:
            ancestor["gpu_events"] = [
                u for u in ancestor.get("gpu_events", []) if u not in kernel_uids
            ]
            ancestor = graph.get_parent_event(ancestor)
        make_connections(graph, kernels, dispatches)
        for dispatch, kernel in zip(dispatches, kernels):
            owner = graph.get_parent_event(dispatch)
            while owner is not None and owner.get("cat") != "cpu_op":
                owner = graph.get_parent_event(owner)
            if owner is None or not owner.get("name", "").startswith("aten::"):
                raise TrainingGraphContractError(
                    "A replay GPU event lacks an explicit captured ATen CPU owner"
                )
            if kernel["cat"] == "kernel":
                args = owner.get("args", {})
                vectors = [
                    args.get(k) for k in ("Input Dims", "Input type", "Input Strides")
                ]
                if (
                    any(not isinstance(v, list) or not v for v in vectors)
                    or len({len(v) for v in vectors}) != 1
                ):
                    raise TrainingGraphContractError(
                        "Captured compute owner lacks complete shape/dtype/stride vectors"
                    )
            kernel["graph_attribution"] = {
                "replay_event_index": kernel[UID],
                "graph_launch_event_index": launch[UID],
                "capture_file": record["file"],
                "capture_sha256": record["sha256"],
                "capture_id": record["capture_id"],
                "segment_id": record["segment_id"],
                "phase": record["phase"],
                "capture_dispatch_event_index": dispatch["capture_origin"]["uid"],
                "capture_operator_event_index": owner["capture_origin"]["uid"],
                "capture_operator_name": owner["name"],
                "capture_operator_tid": owner.get("tid"),
                "Input Dims": copy.deepcopy(owner.get("args", {}).get("Input Dims")),
                "Input type": copy.deepcopy(owner.get("args", {}).get("Input type")),
                "Input Strides": copy.deepcopy(
                    owner.get("args", {}).get("Input Strides")
                ),
                "shape_source": "capture_profiler_metadata",
            }
    finalize_non_gpu_paths(graph)
    graph.trace_metadata["tracelens_training_graph_merge"] = {
        "schema": SCHEMA,
        "status": "complete",
        "manifest_sha256": sha256_file(manifest_path),
        "mapping_contract": manifest["mapping_contract"],
        "provenance": provenance,
        "cpu_timing_semantics": "capture_structure_only",
        "training_state_equivalence": "not_evaluated",
        "instances": mapping_report,
    }
    return graph


def generate_training_graph_report(
    capture_folder,
    manifest_path,
    *,
    profile_json_path=None,
    output_xlsx_path=None,
    output_csvs_dir=None,
    **report_options,
):
    """Generate native per-op/PerfModel tables plus traceable graph attribution.

    Hardware ceilings must be supplied through the native ``gpu_arch`` or
    ``gpu_arch_json_path`` options; a graph mapping never fabricates them.
    """
    import pandas as pd
    from ..Reporting.generate_perf_report_pytorch_inference import (
        generate_perf_report_pytorch,
    )
    from ..Reporting.reporting_utils import write_report_outputs

    tree = merge_training_capture_trace(
        capture_folder, manifest_path, profile_json_path
    )
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    replay_path = str(Path(capture_folder) / manifest["replay_file"])
    tables = generate_perf_report_pytorch(
        profile_json_path=replay_path,
        augmented_tree=tree,
        write_outputs=False,
        **report_options,
    )
    tables["graph_attribution"] = pd.DataFrame(
        [
            {
                **e["graph_attribution"],
                "kernel_name": e["name"],
                "gpu_category": e["cat"],
                "gpu_duration_us": e["dur"],
                "gpu_stream": e.get("args", {}).get("stream"),
            }
            for e in tree.events
            if "graph_attribution" in e
        ]
    )
    contract = tree.trace_metadata["tracelens_training_graph_merge"]
    tables["graph_contract"] = pd.DataFrame(contract["instances"])
    if output_xlsx_path is not None or output_csvs_dir is not None:
        write_report_outputs(
            tables, xlsx_path=output_xlsx_path, csvs_dir=output_csvs_dir
        )
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--capture-folder", required=True)
    parser.add_argument("--replay")
    parser.add_argument(
        "--report",
        required=True,
        help="Write mapping evidence or the explicit rejection reason",
    )
    parser.add_argument(
        "--csvs-dir", help="Also generate native performance report CSVs"
    )
    parser.add_argument("--xlsx", help="Also generate a native performance workbook")
    parser.add_argument(
        "--gpu-arch-json", help="Explicit hardware ceilings for PerfModel"
    )
    args = parser.parse_args(argv)
    try:
        tree = merge_training_capture_trace(
            args.capture_folder, args.manifest, args.replay
        )
        report = tree.trace_metadata["tracelens_training_graph_merge"]
        if args.csvs_dir or args.xlsx:
            tables = generate_training_graph_report(
                args.capture_folder,
                args.manifest,
                profile_json_path=args.replay,
                output_csvs_dir=args.csvs_dir,
                output_xlsx_path=args.xlsx,
                gpu_arch_json_path=args.gpu_arch_json,
                collective_analysis=False,
            )
            report["report_tables"] = {
                name: len(table) for name, table in tables.items()
            }
    except (TrainingGraphContractError, OSError, ValueError) as exc:
        report = {"schema": SCHEMA, "status": "rejected", "reason": str(exc)}
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
