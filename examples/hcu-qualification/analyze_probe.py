###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Exercise native analysis on torch_probe outputs; replay requires opt-in."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from TraceLens.Trace2Tree.training_graph import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute-replay", action="store_true")
    args = parser.parse_args()
    probe, output = args.probe_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((probe / "metadata-torch.json").read_text())
    device_name = metadata.get("device_name")
    if not device_name:
        raise ValueError("Re-run torch_probe.py with explicit device identity")
    calibration = metadata["calibration"]
    arch = {
        "name": device_name,
        "manufacturer": "Hygon",
        "mem_bw_gbps": calibration["copy"]["measured_gbps"],
        "max_achievable_tflops": {
            "matrix_fp32": calibration["torch.float32"]["measured_tflops"],
            "matrix_bf16": calibration["torch.bfloat16"]["measured_tflops"],
        },
        "hcu_roofline": {
            "schema_version": 1,
            "device_arch": metadata["gfx"],
            "provenance": {
                "kind": "measured",
                "source": str(probe / "metadata-torch.json"),
                "device_name": device_name,
                "device_arch": metadata["gfx"],
                "timestamp": str(metadata["timestamp"]),
                "stack": {"torch": metadata["torch"], "hip": metadata["hip"]},
                "conditions": {
                    "compute": {"matrix_fp32": {"shape": [4096, 4096, 4096]}},
                    "memory": "256 MiB per tensor, device copy, algorithmic read+write bytes",
                    "limitation": "Measured envelopes, not hardware peaks or E2E savings",
                },
            },
            "workload_mappings": [
                {
                    "name": "aten::mm",
                    "compute_spec": "matrix_fp32",
                    "model": "TraceLens.PerfModel.perf_model.aten_mm",
                    "status": "qualified",
                    "evidence": str(Path(__file__).resolve()),
                }
            ],
        },
    }
    arch_path = output / "arch.json"
    arch_path.write_text(json.dumps(arch, indent=2))
    manifest = {
        "schema": "tracelens.training_graph.v1",
        "mapping_contract": "serialized_single_stream",
        "replay_file": "replay.json",
        "replay_sha256": sha256_file(probe / "replay.json"),
        "provenance": {
            "torch_version": metadata["torch"],
            "hip_version": metadata["hip"],
            "device_type": "hcu",
            "signature_source": "runtime_tensor_metadata",
        },
        "captures": [
            {
                "capture_id": "qualification",
                "segment_id": "trainstep",
                "phase": "train_step",
                "file": "capture.json",
                "sha256": sha256_file(probe / "capture.json"),
                "input_signature": metadata["input_signature"],
                "replay_input_signature": metadata["input_signature"],
            }
        ],
    }
    manifest_path = output / "graph-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    results = []

    def run(name, module, *options):
        command = [sys.executable, "-m", module, *map(str, options)]
        with (output / (name + ".log")).open("w") as log:
            try:
                code = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=300
                ).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        results.append({"name": name, "returncode": code, "command": command})
        (output / "execution.json").write_text(json.dumps(results, indent=2))
        if code != 0:
            raise RuntimeError(f"{name} failed; see {output / (name + '.log')}")

    report = output / "agent" / "perf_report_csvs"
    run(
        "report",
        "TraceLens.Reporting.generate_perf_report_pytorch",
        "--profile_json_path",
        probe / "eager.json",
        "--output_csvs_dir",
        report,
        "--gpu_arch_json_path",
        arch_path,
        "--enable_kernel_summary",
        "--include_overlap_info",
        "--include_unlinked_kernels",
        "--include_call_stack",
    )
    run(
        "roofline",
        "TraceLens.PerfModel.hcu_roofline",
        "--perf-csv",
        report / "unified_perf_summary.csv",
        "--arch-json",
        arch_path,
        "--output",
        output / "roofline.json",
    )
    run(
        "graph",
        "TraceLens.Trace2Tree.training_graph",
        "--manifest",
        manifest_path,
        "--capture-folder",
        probe,
        "--report",
        output / "graph.json",
        "--csvs-dir",
        output / "graph-tables",
        "--gpu-arch-json",
        arch_path,
    )
    run(
        "self-comparison",
        "TraceLens.Reporting.compare_perf_reports_pytorch",
        report,
        report,
        "--names",
        "original",
        "same-input",
        "-o",
        output / "comparison.xlsx",
        "--output_csvs_dir",
        output / "comparison",
    )
    run(
        "tracediff",
        "TraceLens.Reporting.generate_perf_report_pytorch",
        "--profile_json_path",
        probe / "eager.json",
        "--comparison_json_path",
        probe / "eager.json",
        "--output_csvs_dir",
        output / "diff",
        "--gpu_arch_json_path",
        arch_path,
    )
    run(
        "split",
        "TraceLens.TraceUtils.split_trace.main",
        probe / "eager.json",
        "-o",
        output / "split",
        "--iterations",
        "1:3",
    )
    run(
        "index",
        "TraceLens.TraceIndex.cli",
        "--db",
        output / "index.sqlite",
        "append",
        "--trace-path",
        probe / "eager.json",
        "--report-dir",
        report,
    )
    run(
        "search",
        "TraceLens.TraceIndex.cli",
        "--db",
        output / "index.sqlite",
        "search",
        "mm",
    )
    run(
        "source",
        "TraceLens.TraceUtils.kernel_source.cli",
        "--kernel",
        "qualification_add",
        "--search-path",
        Path(__file__).parent,
    )
    if not json.loads((output / "source.log").read_text())["patchable"]:
        raise RuntimeError("Qualification kernel source was not resolved")
    run(
        "agent-prepare",
        "TraceLens.Agent.Analysis.utils.orchestrator_prepare",
        "--trace-path",
        probe / "eager.json",
        "--platform",
        device_name,
        "--gpu-arch-json",
        arch_path,
        "--output-dir",
        output / "agent",
    )
    run(
        "agent-metrics",
        "TraceLens.Agent.Analysis.utils.hcu_analysis",
        "run",
        "--output-dir",
        output / "agent",
    )
    if args.execute_replay:
        import torch
        from TraceLens.EventReplay.hcu_replay import write_replay_bundle
        from TraceLens.util import DataLoader

        events = DataLoader.load_data(str(probe / "eager.json"))["traceEvents"]
        index = next(
            i
            for i, e in enumerate(events)
            if e.get("name") == "aten::mm"
            and e.get("args", {}).get("Input Dims") == [[512, 768], [768, 256]]
        )
        values = torch.load(probe / "mm-values.pt", weights_only=True)
        bundle = probe / "mm-replay.json"
        write_replay_bundle(
            str(probe / "eager.json"),
            index,
            str(bundle),
            provenance={**manifest["provenance"], "values_source": "captured_runtime"},
            input_values={"self": values["args"][0], "mat2": values["args"][1]},
            expected_output=values["expected"],
        )
        run(
            "replay",
            "TraceLens.EventReplay.hcu_replay",
            "--bundle",
            bundle,
            "--device",
            "hcu",
            "--steps",
            10,
            "--report",
            output / "replay.json",
        )
    print(
        f"Completed {len(results)} bounded commands. Inspect outputs; Agent findings still require review."
    )


if __name__ == "__main__":
    main()
