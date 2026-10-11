###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Authored native CSV dialect fixtures; these are not hardware measurements."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from TraceLens.Reporting.xcompute import (
    generate_perf_report_xcompute,
    parse_xcompute_csv,
)

METRICS = (
    "Instruction Statistics\nIPC(Executed),inst/cycle,0.25\n"
    "Issued instruction,inst,9007199254740993\n"
    "GPU Speed Of Light Throughput\nVALU FLOPs,GFLOPs/s,1.23e2\n"
    "MMOP FLOPs(BF16),GFLOPs/s,N/A\n"
)
SUMMARY = (
    "Dispatch id,Kernel,XCD,Duration [ms]\n"
    "3,vector_add,0,0.000321\n4,vector_add,0,0.001234\n"
)
SQTT = {
    "dispatch_summary": (
        "Dispatch id,Kernel,XCDs,SEs,CUs,Waves,Complete Waves,Incomplete Waves,"
        "Inst Issues,Duration,No-wave Idle,Idle %,Max Active Waves,Avg Active Waves\n"
        "3,vector_add,1,1,1,16,16,0,32,100,0,0.00,4,2.50\n"
    ),
    "hot_inst": (
        "Dispatch id,Kernel,Group,Opcode,Hits,Latency,Latency %\n"
        "3,vector_add,Execution & Sync,s_waitcnt,8,800,25.00\n"
    ),
    "bubble_summary": (
        "Dispatch id,Kernel,Type,Before,After,Count,Duration,Duration %,Max Duration,"
        "Example XCD,Example SE,Example SH,Example CU,Example SIMD,Example WaveSlot,Example Source\n"
        "3,vector_add,issue_gap,global_load_dword,s_waitcnt,4,700,20.00,300,0,0,0,1,2,3,N/A\n"
    ),
}


def write_file(root, name, data):
    path = root / name
    path.write_text(data, encoding="utf-8")
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_manifest(root, captures):
    path = root / "manifest.json"
    path.write_text(
        json.dumps({"schema_version": 1, "captures": captures}), encoding="utf-8"
    )
    return path


def capture(root, identity="counter", summary=SUMMARY):
    perf = root / (identity + ".perf")
    perf.write_bytes(b"authored fixture; not a hardware perf file")
    files = [
        write_file(root, identity + "_dispatch_results_summary_info.csv", summary),
        write_file(root, "3_vector_add.csv", METRICS),
    ]
    return {
        "id": identity,
        "source_perf": {
            "path": perf.name,
            "sha256": hashlib.sha256(perf.read_bytes()).hexdigest(),
        },
        "tools": {
            "XProf": "fixture",
            "hsa-aqlprofile": "fixture",
            "XComputeLight": "fixture",
        },
        "gfx": "fixture",
        "environment": {"qualification": "authored"},
        "commands": [["fixture-only"]],
        "counter_replay": True,
        "metric_basis": {"dispatch": "native summary ms", "metrics": "reported units"},
        "sampling": {"selection": "fixture"},
        "workload": None,
        "csvs": files,
    }


def test_sections_na_units_and_large_integer_are_not_collapsed(tmp_path):
    write_file(tmp_path, "3_vector_add.csv", METRICS)
    result = parse_xcompute_csv(tmp_path / "3_vector_add.csv")
    assert result["sections"] == [
        "Instruction Statistics",
        "GPU Speed Of Light Throughput",
    ]
    assert result["metrics"][1]["value"] == 9007199254740993
    assert result["metrics"][2]["value"] == 123
    assert result["metrics"][2]["unit"] == "GFLOPs/s"
    assert result["metrics"][3]["value"] is None
    assert result["metrics"][3]["raw_value"] == "N/A"
    assert result["metrics"][3]["section_index"] == 1


def test_csv_quoting_bom_and_repeated_metric_names_preserve_rows(tmp_path):
    write_file(
        tmp_path,
        "metrics.csv",
        '\ufeffSection\n"Metric, name",%,1\nOther\n"Metric, name",cycles,2\n',
    )
    result = parse_xcompute_csv(tmp_path / "metrics.csv")
    assert [m["name"] for m in result["metrics"]] == ["Metric, name"] * 2
    assert [m["unit"] for m in result["metrics"]] == ["%", "cycles"]


@pytest.mark.parametrize(
    "content",
    [
        "",
        "Section\n",
        "Section\nmetric,%,nan\n",
        "Section\nmetric,%,Infinity\n",
        "Section\nmetric,%,\n",
        "Section\nmetric,%,1,unexpected\n",
        "Wrong,Header\n1,2\n",
        "Dispatch id,Kernel,XCD,Duration [ms]\n3,vector_add,0\n",
    ],
)
def test_invalid_or_unrecognized_csv_is_rejected(tmp_path, content):
    write_file(tmp_path, "input.csv", content)
    with pytest.raises(ValueError):
        parse_xcompute_csv(tmp_path / "input.csv")


def test_dispatch_duration_uses_declared_ms_without_float_intermediate(tmp_path):
    write_file(
        tmp_path, "summary.csv", SUMMARY.replace("0.000321", "9007199254.740993")
    )
    result = parse_xcompute_csv(tmp_path / "summary.csv")
    assert result["dispatches"][0]["duration_ns"] == 9007199254740993
    assert result["rows"][0]["Duration [ms]"] == "9007199254.740993"


@pytest.mark.parametrize(
    "summary",
    [
        SUMMARY + "3,vector_add,0,0.001\n",
        SUMMARY.replace("0.000321", "-1"),
        SUMMARY.replace("3,vector_add", "3.5,vector_add"),
    ],
)
def test_invalid_summary_identity_or_timing_is_rejected(tmp_path, summary):
    write_file(tmp_path, "summary.csv", summary)
    with pytest.raises(ValueError):
        parse_xcompute_csv(tmp_path / "summary.csv")


def test_metrics_bind_only_one_actual_dispatch_not_every_same_name(tmp_path):
    c = capture(tmp_path)
    c["csvs"][1]["workload"] = {
        "operator_key": "declared-call",
        "shape": [8],
        "phase": "forward",
    }
    report = generate_perf_report_xcompute(
        write_manifest(tmp_path, [c]), tmp_path / "out"
    )
    assert report["status"] == "parsed"
    binding = report["captures"][0]["files"][1]["dispatch_binding"]
    assert binding["dispatch_id"] == 3 and binding["duration_ns"] == 321
    assert "shape" not in binding
    assert (
        report["captures"][0]["files"][1]["manifest_entry"]["workload"]["operator_key"]
        == "declared-call"
    )
    assert any("profiler-off" in limit for limit in report["limits"])


def test_multi_xcd_metrics_remain_unlinked_until_scope_is_explicit(tmp_path):
    c = capture(tmp_path, summary=SUMMARY + "3,vector_add,1,0.000400\n")
    manifest = write_manifest(tmp_path, [c])
    report = generate_perf_report_xcompute(manifest, tmp_path / "ambiguous")
    binding = report["captures"][0]["files"][1]["dispatch_binding"]
    assert binding["status"] == "unlinked" and binding["matches"] == 2
    assert report["status"] == "incomplete"
    c["csvs"][1]["xcd"] = 1
    report = generate_perf_report_xcompute(
        write_manifest(tmp_path, [c]), tmp_path / "explicit"
    )
    assert report["captures"][0]["files"][1]["dispatch_binding"]["duration_ns"] == 400


def test_same_id_and_name_in_other_captures_do_not_share_measurements(tmp_path):
    a = capture(tmp_path, identity="a")
    b = capture(tmp_path, identity="b", summary=SUMMARY.replace("0.000321", "0.000700"))
    report = generate_perf_report_xcompute(
        write_manifest(tmp_path, [a, b]), tmp_path / "out"
    )
    bindings = [c["files"][1]["dispatch_binding"] for c in report["captures"]]
    assert [(b["capture_id"], b["duration_ns"]) for b in bindings] == [
        ("a", 321),
        ("b", 700),
    ]


def test_filename_manifest_disagreement_is_not_overridden(tmp_path):
    c = capture(tmp_path)
    c["csvs"][1]["dispatch_id"] = 4
    with pytest.raises(ValueError, match="disagree"):
        generate_perf_report_xcompute(write_manifest(tmp_path, [c]), tmp_path / "out")


def test_no_summary_keeps_the_metric_identity_unlinked(tmp_path):
    c = capture(tmp_path)
    c["csvs"] = c["csvs"][1:]
    report = generate_perf_report_xcompute(
        write_manifest(tmp_path, [c]), tmp_path / "out"
    )
    assert report["captures"][0]["files"][0]["dispatch_binding"]["status"] == "unlinked"


@pytest.mark.parametrize("kind", ["dispatch_summary", "hot_inst", "bubble_summary"])
def test_sqtt_columns_are_raw_tables_with_no_invented_scalar_or_units(tmp_path, kind):
    write_file(tmp_path, "sqtt.csv", SQTT[kind])
    result = parse_xcompute_csv(tmp_path / "sqtt.csv")
    assert result["kind"] == "sqtt_" + kind
    assert "dispatches" not in result and "metrics" not in result
    assert result["row_identities"] == [{"dispatch_id": 3, "kernel": "vector_add"}]
    if kind == "bubble_summary":
        assert result["rows"][0]["Example Source"] is None
        assert result["rows"][0]["Duration"] == "700"
        assert result["timing_units"]["Duration"] is None


@pytest.mark.parametrize("target", ["csv", "perf"])
def test_provenance_hash_mismatches_are_rejected(tmp_path, target):
    c = capture(tmp_path)
    selected = c["csvs"][0] if target == "csv" else c["source_perf"]
    selected["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        generate_perf_report_xcompute(write_manifest(tmp_path, [c]), tmp_path / "out")


@pytest.mark.parametrize("value", ["NaN", "-1", "invalid"])
def test_malformed_sqtt_numeric_fields_are_not_silent_raw_success(tmp_path, value):
    write_file(
        tmp_path, "sqtt.csv", SQTT["hot_inst"].replace(",800,", "," + value + ",")
    )
    with pytest.raises(ValueError):
        parse_xcompute_csv(tmp_path / "sqtt.csv")


def test_missing_perf_and_nested_unknowns_are_auditable_not_zero_or_verified(tmp_path):
    c = capture(tmp_path)
    (tmp_path / c["source_perf"]["path"]).unlink()
    c["tools"]["XComputeLight"] = None
    c["sampling"] = {"CU": None}
    c["counter_replay"] = None
    report = generate_perf_report_xcompute(
        write_manifest(tmp_path, [c]), tmp_path / "out"
    )
    assert report["status"] == "incomplete"
    provenance = report["captures"][0]["provenance"]
    assert provenance["source_perf"]["verification"] == "declared-only"
    assert provenance["source_perf"]["path"] == c["source_perf"]["path"]
    assert provenance["tools"]["XComputeLight"] is None
    assert provenance["counter_replay"] is None
    assert any("tools is unknown" in gap for gap in report["gaps"])


def test_cli_runs_offline_and_saves_receipt(tmp_path):
    manifest = write_manifest(tmp_path, [capture(tmp_path)])
    output = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "TraceLens.Reporting.xcompute",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["status"] == "parsed"
    receipt = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert (
        receipt["manifest"]["sha256"]
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="overwrite"):
        generate_perf_report_xcompute(manifest, output)
