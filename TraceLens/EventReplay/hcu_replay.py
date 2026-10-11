###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""SHA-bound same-shape operator replay and optional captured-output validation.

HIP devices use PyTorch's ``cuda`` namespace. ``--device hcu`` additionally
requires a HIP build; the report records actual hardware properties. This is
an isolated functional operator experiment, not a training-step replay.
"""

import argparse
import json
from pathlib import Path
import statistics
import time

from .event_replay import EventReplayer
from .safety import ReplayContractError
from .utils import _get_torch_or_raise, TensorCfg
from ..Trace2Tree.training_graph import _checked_file, sha256_file
from ..util import DataLoader

SCHEMA = "tracelens.event_replay.v1"


def _validate_provenance(provenance, captured=False):
    if not isinstance(provenance, dict):
        raise ReplayContractError("Source provenance must be an object")
    for field in ("torch_version", "hip_version", "device_type", "signature_source"):
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            raise ReplayContractError(f"Missing source provenance: {field}")
    if provenance["signature_source"] != "runtime_tensor_metadata":
        raise ReplayContractError(
            "Only runtime_tensor_metadata signatures are supported"
        )
    if captured and provenance.get("values_source") != "captured_runtime":
        raise ReplayContractError(
            "Captured values require explicit captured_runtime provenance"
        )


def _source_event(trace_path, event_index):
    data = DataLoader.load_data(str(trace_path))
    events = data.get("traceEvents")
    if (
        not isinstance(events, list)
        or not isinstance(event_index, int)
        or isinstance(event_index, bool)
        or not 0 <= event_index < len(events)
    ):
        raise ReplayContractError("event_index must identify one raw traceEvents entry")
    event = events[event_index]
    if event.get("cat") != "cpu_op" or not event.get("name", "").startswith("aten::"):
        raise ReplayContractError(
            "Replay source must be a recorded ATen CPU operator event"
        )
    return event


def _cpu_storage_copy(value):
    """Preserve noncontiguous strides without exporting unrelated base storage."""
    torch = _get_torch_or_raise()
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ReplayContractError("Only strided tensor values can be exported")
        count = (
            0
            if 0 in value.shape
            else 1 + sum((d - 1) * s for d, s in zip(value.shape, value.stride()))
        )
        if count > 16_777_216:
            raise ReplayContractError(
                "Captured backing storage exceeds the export bound"
            )
        storage = (
            value.detach()
            .as_strided((count,), (1,), value.storage_offset())
            .cpu()
            .clone()
        )
        return storage.as_strided(value.shape, value.stride())
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_storage_copy(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ReplayContractError("Unsupported captured output structure")


def write_replay_bundle(
    trace_path,
    event_index,
    bundle_path,
    *,
    provenance,
    input_values=None,
    expected_output=None,
):
    """Export a source-bound replay bundle from explicitly captured tensors.

    Input tensor names are the installed schema argument names (e.g. ``self``
    and ``mat2``). Callers must capture values and the reference at the same
    invocation; matching shapes alone do not establish value provenance.
    The trace must reside inside the bundle directory (or a subdirectory).
    """
    torch = _get_torch_or_raise()
    bundle_path = Path(bundle_path).resolve()
    trace_path = Path(trace_path).resolve()
    if bundle_path.suffix != ".json" or bundle_path == trace_path:
        raise ReplayContractError(
            "Replay bundle must be a separate .json file; source traces cannot be overwritten"
        )
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        trace_file = str(trace_path.relative_to(bundle_path.parent))
    except ValueError as exc:
        raise ReplayContractError(
            "Source trace must be inside the bundle directory; choose a bundle next to the trace"
        ) from exc
    semantics = "captured" if input_values is not None else "synthetic"
    _validate_provenance(provenance, captured=semantics == "captured")
    if expected_output is not None and semantics != "captured":
        raise ReplayContractError(
            "A captured output reference requires captured inputs"
        )
    event = _source_event(trace_path, event_index)
    # Validate before exporting or running an operator. Use the source device
    # to detect aliasing before any transfer creates independent storages.
    device = str(next(iter(input_values.values())).device) if input_values else "cpu"
    replayer = EventReplayer(
        event,
        device=device,
        input_semantics=semantics,
        input_values=input_values,
        lazy=False if input_values is not None else True,
    )
    bundle = {
        "schema": SCHEMA,
        "trace_file": trace_file,
        "trace_sha256": sha256_file(trace_path),
        "event_index": event_index,
        "operator_schema": str(replayer.matched_schema),
        "input_semantics": semantics,
        "provenance": provenance,
    }
    if input_values is not None:
        values_path = bundle_path.with_suffix(".pt")
        torch.save(
            {
                "inputs": {k: _cpu_storage_copy(v) for k, v in input_values.items()},
                "expected_output": _cpu_storage_copy(expected_output),
                "has_expected_output": expected_output is not None,
            },
            values_path,
        )
        bundle.update(
            values_file=values_path.name, values_sha256=sha256_file(values_path)
        )
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle


def benchmark_replayer(replayer, *, warmup=3, steps=10, max_seconds=30):
    """Bound iteration count and wall time between completed device calls.

    A running kernel or device synchronization cannot be preempted by Python.
    Use an external process deadline when qualifying an untrusted runtime.
    """
    torch = _get_torch_or_raise()
    if (
        any(not isinstance(x, int) or isinstance(x, bool) for x in (warmup, steps))
        or not 0 <= warmup <= 100
        or not 1 <= steps <= 1000
    ):
        raise ReplayContractError("warmup must be 0..100 and steps 1..1000")
    if not isinstance(max_seconds, (int, float)) or not 0 < max_seconds <= 300:
        raise ReplayContractError("max_seconds must be positive and at most 300")
    if replayer.lazy:
        raise ReplayContractError(
            "Benchmarking requires prepared inputs; lazy replay would include input allocation"
        )
    device = torch.device(replayer.device)
    cuda = device.type == "cuda"

    def synchronize():
        if cuda:
            torch.cuda.synchronize(device)

    start = time.perf_counter()
    deadline = start + max_seconds
    synchronize()
    samples = []
    for index in range(warmup + steps):
        if time.perf_counter() >= deadline:
            raise ReplayContractError(
                "Replay exceeded its wall-time bound before completing all requested samples"
            )
        if cuda:
            with torch.cuda.device(device):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                output = replayer.replay()
                end.record()
                end.synchronize()
                elapsed_us = begin.elapsed_time(end) * 1000
        else:
            before = time.perf_counter()
            output = replayer.replay()
            elapsed_us = (time.perf_counter() - before) * 1e6
        del output
        if time.perf_counter() >= deadline:
            raise ReplayContractError(
                "Replay exceeded its wall-time bound after a completed sample"
            )
        if index >= warmup:
            samples.append(elapsed_us)
    return {
        "warmup": warmup,
        "steps": steps,
        "samples_us": samples,
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "timing_method": (
            "device_events_synchronized_each_call" if cuda else "cpu_wall_clock"
        ),
        "wall_seconds": time.perf_counter() - start,
        "max_seconds": max_seconds,
    }


def run_bundle(
    bundle_path,
    *,
    device="hcu",
    warmup=3,
    steps=10,
    max_seconds=30,
    rtol=1e-5,
    atol=1e-6,
):
    torch = _get_torch_or_raise()
    bundle_path = Path(bundle_path).resolve()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(bundle, dict) or bundle.get("schema") != SCHEMA:
        raise ReplayContractError(f"Expected bundle schema {SCHEMA}")
    provenance = bundle.get("provenance", {})
    _validate_provenance(
        provenance, captured=bundle.get("input_semantics") == "captured"
    )
    if (
        not isinstance(bundle.get("operator_schema"), str)
        or not bundle["operator_schema"]
    ):
        raise ReplayContractError("Bundle needs an exact recorded operator_schema")
    if device == "hcu" or device.startswith("hcu:"):
        if torch.version.hip is None:
            raise ReplayContractError(
                "HCU replay requires an installed HIP PyTorch build"
            )
        device = "cuda" + device[3:]
    target = torch.device(device)
    if target.type not in {"cuda", "cpu"}:
        raise ReplayContractError("Replay devices must be CPU or CUDA/HIP")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ReplayContractError(
            "Requested CUDA/HIP device is unavailable; no CPU fallback"
        )
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    trace_path = _checked_file(
        bundle_path.parent, bundle.get("trace_file"), bundle.get("trace_sha256")
    )
    event = _source_event(trace_path, bundle.get("event_index"))
    semantics = bundle.get("input_semantics")
    values, expected = None, None
    has_expected = False
    if semantics == "captured":
        values_path = _checked_file(
            bundle_path.parent, bundle.get("values_file"), bundle.get("values_sha256")
        )
        data = torch.load(values_path, map_location="cpu", weights_only=True)
        if (
            not isinstance(data, dict)
            or set(data) != {"inputs", "expected_output", "has_expected_output"}
            or not isinstance(data["has_expected_output"], bool)
        ):
            raise ReplayContractError("Malformed captured tensor-value bundle")
        values, expected, has_expected = (
            data["inputs"],
            data["expected_output"],
            data["has_expected_output"],
        )
    elif semantics != "synthetic":
        raise ReplayContractError(
            "Bundle must declare synthetic or captured input_semantics"
        )
    replayer = EventReplayer(
        event,
        device=str(target),
        input_semantics=semantics,
        input_values=values,
        operator_schema=bundle.get("operator_schema"),
    )
    verification = "not_requested"
    if has_expected:
        replayer.verify(expected, rtol=rtol, atol=atol)
        verification = "passed"
    timing = benchmark_replayer(
        replayer, warmup=warmup, steps=steps, max_seconds=max_seconds
    )
    runtime = {
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "device": str(target),
    }
    if target.type == "cuda":
        props = torch.cuda.get_device_properties(target)
        runtime.update(
            device_name=props.name, gcn_arch=getattr(props, "gcnArchName", None)
        )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "source": provenance,
        "trace_sha256": bundle["trace_sha256"],
        "event_index": bundle["event_index"],
        "values_sha256": bundle.get("values_sha256"),
        "operator_schema": str(replayer.matched_schema),
        "input_semantics": semantics,
        "numerical_verification": verification,
        "verification_rtol": rtol if has_expected else None,
        "verification_atol": atol if has_expected else None,
        "training_state_equivalence": "not_evaluated",
        "runtime": runtime,
        "timing": timing,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--device", default="hcu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    try:
        report = run_bundle(
            args.bundle,
            device=args.device,
            warmup=args.warmup,
            steps=args.steps,
            max_seconds=args.max_seconds,
            rtol=args.rtol,
            atol=args.atol,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        report = {"schema": SCHEMA, "status": "rejected", "reason": str(exc)}
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
