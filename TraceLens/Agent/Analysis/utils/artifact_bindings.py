###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""Content bindings for prepared HCU evidence and native analysis outputs."""

import hashlib
import json
from pathlib import Path


def file_fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": path.as_posix(), "sha256": digest.hexdigest()}


def prepared_source_bindings(trace_path, arch_json_path=None):
    result = {"trace": file_fingerprint(trace_path)}
    if arch_json_path is not None:
        result["site_architecture"] = file_fingerprint(arch_json_path)
    return result


def analysis_source_bindings(output_dir, manifest):
    """Fingerprint current inputs, rejecting sources changed since preparation."""
    output_dir = Path(output_dir)
    prepared = manifest.get("prepared_sources")
    if not isinstance(prepared, dict) or "trace" not in prepared:
        raise ValueError(
            "HCU prepared source fingerprints are missing; rerun orchestrator_prepare"
        )
    result = {}
    for name, binding in prepared.items():
        if (
            not isinstance(binding, dict)
            or not binding.get("path")
            or not binding.get("sha256")
        ):
            raise ValueError("Invalid prepared source binding: " + name)
        current = file_fingerprint(binding["path"])
        if current != binding:
            raise ValueError(
                "Source changed since preparation: "
                + name
                + "; rerun orchestrator_prepare"
            )
        result["prepared_source:" + name] = current
    if file_fingerprint(manifest["trace_path"]) != prepared["trace"]:
        raise ValueError("Manifest trace does not match prepared source binding")
    result["manifest"] = file_fingerprint(
        output_dir / "category_data" / "category_manifest.json"
    )
    result["roofline_audit"] = file_fingerprint(
        output_dir / "category_data" / "hcu_roofline.json"
    )
    comparative = manifest.get("comparison_scope") == "comparative"
    csv_dirs = (
        ("perf_report_trace1_csvs", "perf_report_trace2_csvs")
        if comparative
        else ("perf_report_csvs",)
    )
    for directory in csv_dirs:
        for path in sorted((output_dir / directory).glob("*.csv")):
            result["perf_csv:" + directory + "/" + path.name] = file_fingerprint(path)
    if comparative:
        result["comparison_gpu_utilization"] = file_fingerprint(
            output_dir / "metadata" / "trace2_gpu_utilization.json"
        )
    utils_dir = Path(__file__).parent
    for path in (
        utils_dir / "artifact_bindings.py",
        utils_dir / "hcu_analysis.py",
        utils_dir.parent / "category_analyses" / "analysis_utils.py",
        utils_dir.parents[2] / "PerfModel" / "hcu_roofline.py",
        utils_dir.parents[2] / "PerfModel" / "perf_model.py",
        utils_dir.parents[2] / "PerfModel" / "torch_op_mapping.py",
    ):
        result["analysis_source:" + path.name] = file_fingerprint(path)
    return result


def category_source_bindings(output_dir, category, command, common):
    """Include category CSV/metadata/data and the actual native analyzer code."""
    output_dir = Path(output_dir)
    result = dict(common)
    name = category["name"]
    for field, fallback in (
        ("csv_file", output_dir / "category_data" / (name + "_ops.csv")),
        ("metadata_file", output_dir / "metadata" / (name + "_metadata.json")),
        ("data_file", None),
    ):
        path = category.get(field)
        if path is not None:
            result[field] = file_fingerprint(path)
        elif fallback is not None and fallback.is_file():
            result[field] = file_fingerprint(fallback)
    module_name = command[2].rsplit(".", 1)[-1]
    result["category_analyzer_source"] = file_fingerprint(
        Path(__file__).parent.parent / "category_analyses" / (module_name + ".py")
    )
    metadata = result.get("metadata_file")
    if metadata:
        data = json.loads(Path(metadata["path"]).read_text(encoding="utf-8"))
        provenance = data.get("hcu_arch", {}).get("hcu_roofline", {}).get("provenance")
        source = provenance.get("source") if isinstance(provenance, dict) else None
        if isinstance(source, str) and Path(source).is_file():
            result["calibration_source"] = file_fingerprint(source)
    return result
