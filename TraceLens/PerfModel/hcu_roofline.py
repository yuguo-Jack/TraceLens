###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Qualified HCU roofline diagnostics, independent of AMD hardware presets.

The native architecture keys describe a *measured envelope* on HCU.  Dividing
work by that envelope is a conditional reference time, not a guaranteed lower
bound.  Optional vendor ceilings have their own provenance and output fields.
Neither quantity establishes achievable kernel or end-to-end speedup.
"""

import argparse
import ast
import copy
import json
import math
from collections import Counter
from pathlib import Path

SCHEMA_VERSION = 1


def _number(value, positive=False):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        return None
    return value


def _gfx(value):
    return str(value or "").split(":", 1)[0]


def is_hcu_arch(arch):
    """Recognize an explicit HCU contract or a Hygon device identity."""
    name = str(arch.get("name", "")).upper()
    return (
        "hcu_roofline" in arch
        or name.startswith(("BW", "HCU", "DCU"))
        or str(arch.get("manufacturer", "")).lower() == "hygon"
    )


def _provenance_issues(provenance, arch, allowed_kinds, require_stack):
    issues = []
    if not isinstance(provenance, dict):
        return ["provenance"]
    for key in ("source", "device_name", "device_arch"):
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            issues.append("provenance." + key)
    if provenance.get("kind") not in allowed_kinds:
        issues.append("provenance.kind")
    if provenance.get("device_name") != arch.get("name"):
        issues.append("provenance.device_name_mismatch")
    actual_gfx = _gfx(arch.get("hcu_roofline", {}).get("device_arch"))
    if not actual_gfx or _gfx(provenance.get("device_arch")) != actual_gfx:
        issues.append("provenance.device_arch_mismatch")
    if require_stack:
        if not provenance.get("timestamp"):
            issues.append("provenance.timestamp")
        stack = provenance.get("stack")
        if (
            not isinstance(stack, dict)
            or not stack
            or not all(
                isinstance(value, str) and value.strip() for value in stack.values()
            )
        ):
            issues.append("provenance.stack")
        if (
            not isinstance(provenance.get("conditions"), dict)
            or not provenance["conditions"]
        ):
            issues.append("provenance.conditions")
    return issues


def qualify_hcu_arch(arch, expected_name=None, expected_device_arch=None):
    """Return an architecture with only provenance-qualified native inputs.

    Missing calibration is a supported unknown state.  A device identity or
    schema mismatch raises ``ValueError`` so a wrong device cannot be selected
    silently.  Non-HCU architectures are returned unchanged for compatibility.
    """
    if not isinstance(arch, dict):
        raise ValueError("Architecture JSON must contain an object")
    if not is_hcu_arch(arch):
        return arch
    result = copy.deepcopy(arch)
    contract = result.setdefault("hcu_roofline", {})
    if not isinstance(contract, dict):
        raise ValueError("hcu_roofline must contain an object")
    if contract.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError("Unsupported hcu_roofline schema_version")
    contract["schema_version"] = SCHEMA_VERSION
    if expected_name and result.get("name") != expected_name:
        raise ValueError("Architecture name does not match the selected platform")
    if expected_device_arch and _gfx(contract.get("device_arch")) != _gfx(
        expected_device_arch
    ):
        raise ValueError("Architecture does not match the observed HCU ISA")
    issues = _provenance_issues(
        contract.get("provenance"), result, {"measured", "calibrated"}, True
    )
    contract["qualification"] = {
        "sustained_status": "unknown" if issues else "qualified",
        "sustained_missing_fields": issues,
        "reference_kind": "measured_envelope",
        "guaranteed_upper_bound": False,
    }
    # Retain the supplied values for audit, but do not feed unqualified numbers
    # to existing native models or category metadata.
    contract.setdefault(
        "supplied_sustained",
        {
            "mem_bw_gbps": result.get("mem_bw_gbps"),
            "max_achievable_tflops": copy.deepcopy(
                result.get("max_achievable_tflops", {})
            ),
        },
    )
    result["mem_bw_gbps"] = (
        None if issues else _number(result.get("mem_bw_gbps"), positive=True)
    )
    peaks = result.get("max_achievable_tflops", {})
    result["max_achievable_tflops"] = {
        key: value
        for key, raw in (peaks.items() if isinstance(peaks, dict) else [])
        if not issues and (value := _number(raw, positive=True)) is not None
    }
    theoretical = contract.get("theoretical", {})
    theory_issues = _provenance_issues(
        theoretical.get("provenance") if isinstance(theoretical, dict) else None,
        result,
        {"vendor"},
        False,
    )
    contract["qualification"]["theoretical_status"] = (
        "unknown" if theory_issues else "qualified"
    )
    contract["qualification"]["theoretical_missing_fields"] = theory_issues
    return result


def _literal(value):
    if isinstance(value, (list, tuple, dict)):
        return value
    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None


def _qualified_mapping(row, arch):
    contract = arch.get("hcu_roofline", {})
    mappings = contract.get("workload_mappings", [])
    if not isinstance(mappings, list):
        return None, "workload_mapping"
    matches = []

    def sequence(value):
        return (
            [sequence(item) for item in value]
            if isinstance(value, (list, tuple))
            else value
        )

    for mapping in mappings:
        if not isinstance(mapping, dict) or mapping.get("name") != row.get("name"):
            continue
        if mapping.get("compute_spec") != row.get("Compute Spec"):
            continue
        if any(
            sequence(_literal(row.get(column))) != sequence(mapping[key])
            for key, column in (
                ("input_dims", "Input Dims"),
                ("input_type", "Input type"),
            )
            if key in mapping
        ):
            continue
        if (
            mapping.get("status") == "qualified"
            and isinstance(mapping.get("evidence"), str)
            and mapping["evidence"].strip()
            and isinstance(mapping.get("model"), str)
            and mapping["model"].strip()
        ):
            matches.append(mapping)
    if not matches:
        return None, "qualified_workload_mapping"
    if len(matches) != 1:
        return None, "ambiguous_workload_mapping"
    return matches[0], None


def _validate_work(row, mapping, flops, bytes_moved):
    """Recompute the registered native model from the recorded shape and dtype."""
    from .torch_op_mapping import op_to_perf_model_class_map

    model_class = op_to_perf_model_class_map.get(row.get("name"))
    if model_class is None:
        return "registered_perf_model"
    model_name = model_class.__module__ + "." + model_class.__name__
    if model_name != mapping.get("model"):
        return "workload_model_mismatch"
    dims, types = _literal(row.get("Input Dims")), _literal(row.get("Input type"))
    if not isinstance(dims, (list, tuple)) or not isinstance(types, (list, tuple)):
        return "recorded_shape_and_dtype"
    if row.get("name") == "aten::mm":
        # Upstream aten_mm historically trusts input metadata and warns rather
        # than rejects mixed inputs. HCU admission requires a valid dense 2D
        # homogeneous matrix contract before trusting its work arithmetic.
        if (
            len(dims) != 2
            or any(
                not isinstance(shape, (list, tuple)) or len(shape) != 2
                for shape in dims
            )
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for shape in dims
                for size in shape
            )
            or dims[0][1] != dims[1][0]
        ):
            return "dense_2d_mm_shape_contract"
        if len(types) != 2 or types[0] != types[1]:
            return "homogeneous_mm_input_dtype"
    event = {"name": row.get("name"), "args": {"Input Dims": dims, "Input type": types}}
    params = _literal(row.get("perf_params"))
    if isinstance(params, dict):
        if params.get("stride_A") is not None and params.get("stride_B") is not None:
            event["args"]["Input Strides"] = [params["stride_A"], params["stride_B"]]
    try:
        model = model_class(event, arch=None)
        work_flops, work_bytes = float(model.flops()), float(model.bytes())
        compute_spec = (
            str(model.get_maf_type()) + "_" + str(model.get_compute_precision())
        )
    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        AttributeError,
        NotImplementedError,
        AssertionError,
    ):
        return "workload_model_reconstruction"
    if compute_spec != row.get("Compute Spec"):
        return "workload_compute_spec_mismatch"
    if not math.isclose(flops, work_flops, rel_tol=1e-6, abs_tol=1):
        return "workload_flops_mismatch"
    if not math.isclose(bytes_moved, work_bytes, rel_tol=1e-6, abs_tol=1):
        return "workload_bytes_mismatch"
    return None


def _reference(flops, bytes_moved, compute_spec, peaks, mem_bw):
    compute_peak = (
        _number(peaks.get(compute_spec), positive=True)
        if isinstance(peaks, dict)
        else None
    )
    mem_bw = _number(mem_bw, positive=True)
    missing = []
    if flops > 0 and compute_peak is None:
        missing.append("compute_reference." + str(compute_spec))
    if bytes_moved > 0 and mem_bw is None:
        missing.append("memory_reference.mem_bw_gbps")
    if missing:
        return None, missing
    compute_us = flops / (compute_peak * 1e6) if flops > 0 else 0.0
    memory_us = bytes_moved / (mem_bw * 1e3) if bytes_moved > 0 else 0.0
    return {
        "time_us": max(compute_us, memory_us),
        "compute_time_us": compute_us,
        "memory_time_us": memory_us,
        "bound": "compute" if compute_us >= memory_us else "memory",
        "compute_tflops": compute_peak,
        "memory_gbps": mem_bw,
    }, []


def evaluate_hcu_roofline_row(row, arch):
    """Export one modeled row, with unknowns and conditional headroom explicit.

    GFLOPS/Data Moved describe one invocation, kernel time is the summed duration
    of all invocations.  Arithmetic uses ``sum / count``, avoiding an unweighted
    mean of efficiencies.  Modeled bytes are compulsory traffic, not HBM counters.
    """
    arch = qualify_hcu_arch(arch)
    if not is_hcu_arch(arch):
        raise ValueError("HCU diagnostics require an explicit HCU architecture")
    contract = arch["hcu_roofline"]
    provenance = contract.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    result = {
        "name": str(row.get("name", "Unknown")),
        "compute_spec": (
            row.get("Compute Spec")
            if isinstance(row.get("Compute Spec"), str)
            else None
        ),
        "input_dims": _literal(row.get("Input Dims")),
        "input_type": _literal(row.get("Input type")),
        "status": "unknown",
        "missing_fields": [],
        "warnings": [],
        "model": None,
        "mapping_evidence": None,
        "count": None,
        "modeled_flops_per_invocation": None,
        "modeled_bytes_per_invocation": None,
        "observed_kernel_time_sum_us": _number(row.get("Kernel Time (µs)_sum")),
        "observed_kernel_time_mean_us": None,
        "measured_reference_time_us": None,
        "reference_bound": None,
        "efficiency_vs_measured_pct": None,
        "conditional_headroom_ratio": None,
        "conditional_kernel_time_headroom_us": None,
        "theoretical_lower_bound_us": None,
        "efficiency_vs_theoretical_pct": None,
        "theoretical_kernel_time_savings_ceiling_us": None,
        "achievable_kernel_speedup": None,
        "achievable_e2e_savings_pct": None,
        "reference_kind": "measured_envelope",
        "guaranteed_upper_bound": False,
        "time_basis": "summed_kernel_duration_not_wallclock",
        "bytes_basis": "native_model_compulsory_bytes_not_measured_HBM_traffic",
        "architecture_provenance": contract.get("provenance"),
        "calibration_scope": provenance.get("conditions"),
        "calibration_shape_match": None,
    }
    count = _number(row.get("operation_count", row.get("count")), positive=True)
    if count is None or not count.is_integer():
        result["missing_fields"].append("positive_integer_operation_count")
    else:
        result["count"] = int(count)
    duration = _number(row.get("Kernel Time (µs)_sum"), positive=True)
    if duration is None:
        result["missing_fields"].append("positive_kernel_time_sum")
    if result["count"] is not None and duration is not None:
        result["observed_kernel_time_mean_us"] = duration / count
    flops = _number(row.get("GFLOPS"))
    byte_mb = _number(row.get("Data Moved (MB)"))
    if flops is None:
        result["missing_fields"].append("modeled_GFLOPS_per_invocation")
    if byte_mb is None:
        result["missing_fields"].append("modeled_Data_Moved_MB_per_invocation")
    mapping, mapping_issue = _qualified_mapping(row, arch)
    if mapping_issue:
        result["missing_fields"].append(mapping_issue)
    if result["missing_fields"]:
        return result
    flops *= 1e9
    bytes_moved = byte_mb * 1024 * 1024
    result["modeled_flops_per_invocation"] = flops
    result["modeled_bytes_per_invocation"] = bytes_moved
    result["model"], result["mapping_evidence"] = mapping["model"], mapping["evidence"]
    params = _literal(row.get("perf_params"))
    conditions = provenance.get("conditions", {})
    conditions = conditions if isinstance(conditions, dict) else {}
    compute_conditions = conditions.get("compute", {})
    calibration_entry = (
        compute_conditions.get(result["compute_spec"], {})
        if isinstance(compute_conditions, dict)
        else {}
    )
    calibration_shape = (
        calibration_entry.get("shape") if isinstance(calibration_entry, dict) else None
    )
    if (
        isinstance(params, dict)
        and calibration_shape is not None
        and all(key in params for key in ("M", "N", "K"))
    ):
        result["calibration_shape_match"] = [
            params["M"],
            params["N"],
            params["K"],
        ] == calibration_shape
        if not result["calibration_shape_match"]:
            result["warnings"].append(
                "Workload shape differs from the GEMM calibration shape; the ratio is a conditional screening diagnostic."
            )
    work_issue = _validate_work(row, mapping, flops, bytes_moved)
    if work_issue:
        result["missing_fields"].append(work_issue)
        return result
    if flops == 0 and bytes_moved == 0:
        result["missing_fields"].append("nonzero_modeled_work")
        return result
    result["workload_status"] = "qualified"
    reference, missing = _reference(
        flops,
        bytes_moved,
        result["compute_spec"],
        arch.get("max_achievable_tflops"),
        arch.get("mem_bw_gbps"),
    )
    result["missing_fields"].extend(missing)
    if contract["qualification"]["sustained_status"] != "qualified":
        result["missing_fields"].extend(
            contract["qualification"]["sustained_missing_fields"]
        )
    if reference:
        result["status"] = "supported"
        result["measured_reference_time_us"] = reference["time_us"]
        result["reference_bound"] = reference["bound"]
        result["resolved_compute_tflops"] = reference["compute_tflops"]
        result["resolved_memory_gbps"] = reference["memory_gbps"]
        result["reference_compute_time_us"] = reference["compute_time_us"]
        result["reference_memory_time_us"] = reference["memory_time_us"]
        ratio = duration / count / reference["time_us"]
        result["efficiency_vs_measured_pct"] = 100 / ratio
        result["conditional_headroom_ratio"] = ratio
        result["conditional_kernel_time_headroom_us"] = max(
            0, duration - count * reference["time_us"]
        )
        result["warnings"].append(
            "Measured envelope is shape/stack dependent; headroom requires workload calibration and intervention validation."
        )
        if ratio < 1:
            result["warnings"].append(
                "Observed throughput exceeds this measured envelope; it is not an architectural ceiling."
            )
    theoretical = contract.get("theoretical", {})
    if contract["qualification"]["theoretical_status"] == "qualified":
        ceiling, theory_missing = _reference(
            flops,
            bytes_moved,
            result["compute_spec"],
            theoretical.get("max_achievable_tflops"),
            theoretical.get("mem_bw_gbps"),
        )
        result["theoretical_missing_fields"] = theory_missing
        result["theoretical_provenance"] = theoretical.get("provenance")
        if ceiling:
            result["theoretical_lower_bound_us"] = ceiling["time_us"]
            result["efficiency_vs_theoretical_pct"] = (
                ceiling["time_us"] / (duration / count) * 100
            )
            if duration / count >= ceiling["time_us"]:
                result["theoretical_kernel_time_savings_ceiling_us"] = (
                    duration - count * ceiling["time_us"]
                )
            else:
                result["warnings"].append(
                    "Observed time violates the supplied theoretical bound; verify model and vendor ceiling."
                )
    return result


def export_hcu_roofline(frame, arch, wallclock_time_us=None):
    """Return a JSON-safe per-op audit and a coverage summary, without E2E projections."""
    arch = qualify_hcu_arch(arch)
    rows = [evaluate_hcu_roofline_row(row, arch) for _, row in frame.iterrows()]
    known_time_sum = sum(
        row["observed_kernel_time_sum_us"]
        for row in rows
        if row["observed_kernel_time_sum_us"] is not None
    )
    missing_time_rows = sum(row["observed_kernel_time_sum_us"] is None for row in rows)
    total = known_time_sum if not missing_time_rows else None
    supported = [row for row in rows if row["status"] == "supported"]
    for row in rows:
        observed = row["observed_kernel_time_sum_us"]
        row["share_of_summed_kernel_time_pct"] = (
            observed / total * 100 if total and observed is not None else None
        )
        headroom = row["conditional_kernel_time_headroom_us"]
        row["conditional_screening_score"] = (
            headroom / total * 100 if total and headroom is not None else None
        )
        row["screening_score_basis"] = (
            "conditional_kernel_gap_over_summed_kernel_time_not_E2E_savings"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "platform": arch.get("name"),
        "device_arch": arch["hcu_roofline"].get("device_arch"),
        "architecture_qualification": arch["hcu_roofline"]["qualification"],
        "operations": rows,
        "summary": {
            "operation_rows": len(rows),
            "supported_rows": len(supported),
            "unknown_rows": len(rows) - len(supported),
            "observed_kernel_time_sum_us": total,
            "observed_known_kernel_time_sum_us": known_time_sum,
            "missing_observed_time_rows": missing_time_rows,
            "supported_kernel_time_sum_us": sum(
                row["observed_kernel_time_sum_us"] for row in supported
            ),
            "missing_field_counts": dict(
                Counter(field for row in rows for field in row["missing_fields"])
            ),
            "wallclock_time_us": _number(wallclock_time_us, positive=True),
            "kernel_time_sum_is_wallclock": False,
            "achievable_e2e_savings_pct": None,
            "aggregate_e2e_speedup": None,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export provenance-qualified HCU roofline/headroom diagnostics"
    )
    parser.add_argument(
        "--perf-csv", required=True, help="Native unified_perf_summary.csv"
    )
    parser.add_argument(
        "--arch-json",
        required=True,
        help="HCU identity, measured envelope and qualified workload mappings",
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--wallclock-time-us",
        type=float,
        help="Measured trace window span; never inferred from summed kernels",
    )
    args = parser.parse_args(argv)
    import pandas as pd

    with open(args.arch_json, encoding="utf-8") as stream:
        arch = json.load(stream)
    result = export_hcu_roofline(
        pd.read_csv(args.perf_csv), arch, args.wallclock_time_us
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result["summary"], allow_nan=False))


if __name__ == "__main__":
    main()
