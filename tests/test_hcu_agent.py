###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Native HCU prepare/category/aggregate contracts without hardware tests."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from TraceLens.Agent.Analysis.category_analyses.analysis_utils import (
    build_operation_metrics,
    compute_impact_estimates,
)
from TraceLens.Agent.Analysis.utils.arch_utils import load_arch
from TraceLens.Agent.Analysis.utils.hcu_analysis import (
    aggregate_hcu_analysis,
    analyze_prepared_categories,
    category_analysis_command,
    record_hcu_agent_review,
)
from TraceLens.Agent.Analysis.utils.artifact_bindings import (
    file_fingerprint,
    prepared_source_bindings,
)
from TraceLens.Agent.Analysis.utils.orchestrator_prepare import (
    _architecture_metadata,
    _extract_standalone_fusion_candidates,
    _hcu_category_time_metrics,
)
from TraceLens.PerfModel.hcu_roofline import export_hcu_roofline, qualify_hcu_arch
from TraceLens.Agent.Analysis.category_analyses.cpu_idle_analysis import (
    analyze_kernel_patterns,
)
from TraceLens.Agent.Analysis.utils.validation_utils import (
    validate_findings_file,
    _validate_report_comparison_scope_diffs,
    _check_time_sanity,
)


def _arch():
    return qualify_hcu_arch(
        {
            "name": "BW1000_H",
            "hcu_roofline": {"schema_version": 1, "device_arch": "gfx936"},
        }
    )


def _prepared(tmp_path):
    (tmp_path / "category_data").mkdir()
    (tmp_path / "metadata").mkdir()
    (tmp_path / "category_findings").mkdir()
    (tmp_path / "system_findings").mkdir()
    arch = _arch()
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": []}))
    site = tmp_path / "site.json"
    site.write_text(json.dumps(arch))
    frame = pd.DataFrame(
        [{"name": "custom::op", "operation_count": 1, "Kernel Time (µs)_sum": 20}]
    )
    manifest = {
        "platform": "BW1000_H",
        "hcu_roofline": {"reference_kind": "measured_envelope"},
        "trace_path": str(trace),
        "prepared_sources": prepared_source_bindings(trace, site),
        "comparison_scope": "standalone",
        "gpu_utilization": {"total_time_ms": 0.1, "computation_time_percent": 20},
        "categories": [
            {
                "name": "custom",
                "skill": "generic-op-analyzer",
                "tier": "compute_kernel",
                "gpu_kernel_time_ms": 0.02,
            }
        ],
    }
    (tmp_path / "category_data" / "category_manifest.json").write_text(
        json.dumps(manifest)
    )
    (tmp_path / "category_data" / "hcu_roofline.json").write_text(
        json.dumps(export_hcu_roofline(frame, arch, 100))
    )
    frame.to_csv(tmp_path / "category_data" / "custom_ops.csv", index=False)
    metadata = {
        "platform": "BW1000_H",
        "output_dir": str(tmp_path),
        "category_name": "custom",
        "gpu_utilization": manifest["gpu_utilization"],
        **_architecture_metadata(arch),
    }
    (tmp_path / "metadata" / "custom_metadata.json").write_text(json.dumps(metadata))
    return tmp_path


def test_arch_loader_accepts_site_file_and_preserves_unknown(tmp_path):
    path = tmp_path / "hcu.json"
    path.write_text(json.dumps(_arch()))
    arch = load_arch("BW1000_H", path)
    assert arch["mem_bw_gbps"] is None
    assert _architecture_metadata(arch)["peak_mem_bw_tbs"] is None
    assert _architecture_metadata(arch)["max_achievable_tflops"] == {}
    with pytest.raises(ValueError):
        load_arch("MI300X", path)


@pytest.mark.parametrize("runtime_category", ["hip_runtime", "hip_driver"])
def test_hip_runtime_wrappers_are_not_fusion_containers(tmp_path, runtime_category):
    kernels = [
        {"UID": 2, "name": "elementwise_one", "cat": "kernel", "dur": 4},
        {"UID": 3, "name": "elementwise_two", "cat": "kernel", "dur": 5},
    ]
    wrapper = {
        "UID": 1,
        "name": "hipRuntimeWrapper",
        "cat": runtime_category,
        "gpu_events": [2, 3],
    }
    events = [wrapper] + kernels
    tree = SimpleNamespace(
        events=events,
        get_UID2event=lambda uid: next(
            event for event in events if event["UID"] == uid
        ),
    )
    analyzer = SimpleNamespace(
        event_to_category=lambda event: event["cat"],
        collect_unified_perf_events=lambda: [],
    )
    assert _extract_standalone_fusion_candidates(analyzer, tree, str(tmp_path)) == []


def test_unknown_hcu_work_never_uses_raw_report_efficiency_or_heuristic():
    frame = pd.DataFrame(
        [
            {
                "name": "custom::op",
                "count": 1,
                "Kernel Time (µs)_sum": 100,
                "Pct Roofline_mean": 10,
                "Compute Spec": "matrix_bf16",
            }
        ]
    )
    metadata = {
        "hcu_arch": _arch(),
        "max_achievable_tflops": {"matrix_bf16": 708},
        "peak_mem_bw_tbs": 5.3,
        "gpu_utilization": {"total_time_ms": 1},
    }
    operations = build_operation_metrics(frame, metadata, {})
    assert operations[0]["efficiency"]["efficiency_percent"] is None
    assert operations[0]["efficiency"]["resolved_peak_maf"] is None
    assert operations[0]["impact_score"] is None
    assert compute_impact_estimates(operations, "other", baseline_ms=1) == []


def test_native_analyzers_are_selected_only_from_observed_categories():
    cmd = category_analysis_command({"name": "gemm", "tier": "compute_kernel"}, "out")
    assert "TraceLens.Agent.Analysis.category_analyses.gemm_analysis" in cmd
    custom = category_analysis_command(
        {"name": "custom", "tier": "compute_kernel"}, "out"
    )
    assert "TraceLens.Agent.Analysis.category_analyses.other_analysis" in custom
    assert custom[-3:] == ["custom", "--comparison_scope", "standalone"]
    with pytest.raises(ValueError):
        category_analysis_command(
            {"name": "../../outside", "tier": "compute_kernel"}, "out"
        )


def test_native_category_and_aggregate_keep_missing_findings_explicit(tmp_path):
    out = _prepared(tmp_path)
    executions = analyze_prepared_categories(out)
    assert len(executions) == 1
    assert executions[0]["exit_code"] == 0
    result = aggregate_hcu_analysis(out)
    assert result["status"] == "metrics_complete"
    assert result["agent_findings_status"] == "incomplete"
    assert result["native_validation"]["coverage_check"]["status"] == "WARN"
    assert result["agent_report_status"] == "not_generated"
    priority = json.loads((out / "priority_data.json").read_text())
    assert priority["recommendations"] == []
    assert priority["hcu_headroom"]["aggregate_e2e_speedup"] is None


def test_missing_metrics_fail_aggregate_instead_of_becoming_empty_pass(tmp_path):
    out = _prepared(tmp_path)
    result = aggregate_hcu_analysis(out)
    assert result["status"] == "failed"
    assert result["failures"]


def test_native_failed_run_cannot_reuse_old_metrics(tmp_path, monkeypatch):
    from TraceLens.Agent.Analysis.utils import hcu_analysis

    out = _prepared(tmp_path)
    stale = out / "category_data" / "custom_metrics.json"
    stale.write_text(json.dumps({"status": "OK", "category": "custom"}))
    monkeypatch.setattr(
        hcu_analysis.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="", stderr="failed", returncode=1
        ),
    )
    executions = analyze_prepared_categories(out)
    assert not stale.exists()
    assert executions[0]["metrics_status"] == "MISSING_OUTPUT"
    assert aggregate_hcu_analysis(out)["status"] == "failed"


def test_absent_category_is_not_a_valid_worker_assignment(tmp_path):
    out = _prepared(tmp_path)
    with pytest.raises(ValueError, match="absent"):
        analyze_prepared_categories(out, categories=["sdpa_fwd"])


def test_timed_out_native_analyzer_is_audited_and_partial_metrics_removed(
    tmp_path, monkeypatch
):
    from TraceLens.Agent.Analysis.utils import hcu_analysis

    out = _prepared(tmp_path)
    partial = out / "category_data" / "custom_metrics.json"

    def timed_out(command, **kwargs):
        assert kwargs["timeout"] == 0.25
        partial.write_text(json.dumps({"status": "OK"}))
        raise hcu_analysis.subprocess.TimeoutExpired(
            command, kwargs["timeout"], output=b"partial"
        )

    monkeypatch.setattr(hcu_analysis.subprocess, "run", timed_out)
    result = analyze_prepared_categories(out, category_timeout_s=0.25)[0]
    assert result["metrics_status"] == "TIMEOUT"
    assert result["exit_code"] is None
    assert not partial.exists()
    assert (out / "native_category_logs" / "custom.execution.json").is_file()
    assert aggregate_hcu_analysis(out)["status"] == "failed"


@pytest.mark.parametrize("timeout", [0, -1, 301])
def test_cpu_category_timeout_is_bounded(tmp_path, timeout):
    with pytest.raises(ValueError, match="at most 300"):
        analyze_prepared_categories(tmp_path, category_timeout_s=timeout)


def test_total_cpu_budget_prevents_late_child_and_stale_success(tmp_path, monkeypatch):
    from TraceLens.Agent.Analysis.utils import hcu_analysis

    out = _prepared(tmp_path)
    stale = out / "category_data" / "custom_metrics.json"
    stale.write_text(json.dumps({"status": "OK"}))
    values = iter([0, 2])
    monkeypatch.setattr(hcu_analysis.time, "monotonic", lambda: next(values))

    def forbidden_child(*args, **kwargs):
        raise AssertionError("A category cannot start after the total analysis budget")

    monkeypatch.setattr(hcu_analysis.subprocess, "run", forbidden_child)
    result = analyze_prepared_categories(out, analysis_timeout_s=1)[0]
    assert result["metrics_status"] == "TIME_BUDGET_EXHAUSTED"
    assert result["started"] is False
    assert not stale.exists()
    assert (out / "native_category_logs" / "custom.execution.json").is_file()


def test_missing_hcu_cpu_times_are_not_gpu_fallback_or_zero():
    times = _hcu_category_time_metrics(pd.DataFrame([{"Kernel Time (µs)_sum": 30}]))
    assert times["gpu_kernel_time_ms"] == 0.03
    assert times["cpu_duration_ms"] is None
    assert times["sync_time_ms"] is None
    assert times["has_sync_bottleneck"] is None
    observed = _hcu_category_time_metrics(
        pd.DataFrame([{"Kernel Time (µs)_sum": 30, "total_duration_us": 900}])
    )
    assert observed["cpu_duration_ms"] == 0.9
    assert observed["sync_time_ms"] is None
    assert (
        _hcu_category_time_metrics(pd.DataFrame([{"name": "unknown"}]))[
            "gpu_kernel_time_ms"
        ]
        is None
    )


def test_missing_hcu_gpu_time_warns_and_cannot_rank_as_zero(tmp_path):
    out = _prepared(tmp_path)
    manifest_path = out / "category_data" / "category_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["categories"][0]["gpu_kernel_time_ms"] = None
    manifest_path.write_text(json.dumps(manifest))
    assert _check_time_sanity(manifest)["status"] == "WARN"
    assert (
        aggregate_hcu_analysis(out)["status"] == "failed"
    )  # absent actual metrics remains failure
    assert json.loads((out / "priority_data.json").read_text())["priorities"] == []


def _reviewed_custom(out):
    assert analyze_prepared_categories(out)[0]["exit_code"] == 0
    finding = out / "category_findings" / "custom_findings.md"
    finding.write_text(
        "# Reviewed actual custom metrics\n\n## Recommendations\n\n## Detailed Analysis\n"
    )
    record_hcu_agent_review(out)
    assert aggregate_hcu_analysis(out)["agent_findings_status"] == "complete"
    return finding


@pytest.mark.parametrize(
    "relative",
    [
        "category_data/custom_ops.csv",
        "metadata/custom_metadata.json",
        "category_data/hcu_roofline.json",
        "category_data/category_manifest.json",
        "trace.json",
        "site.json",
    ],
)
def test_current_receipts_reject_changed_inputs_at_same_paths(tmp_path, relative):
    out = _prepared(tmp_path)
    _reviewed_custom(out)
    changed = out / relative
    changed.write_text(changed.read_text() + "\n ")
    result = aggregate_hcu_analysis(out)
    assert result["status"] == "failed"
    assert result["agent_findings_status"] == "incomplete"


def test_status_ok_and_old_exit_zero_cannot_accept_modified_metrics(tmp_path):
    out = _prepared(tmp_path)
    _reviewed_custom(out)
    metrics_path = out / "category_data" / "custom_metrics.json"
    data = json.loads(metrics_path.read_text())
    data["status"] = "OK"
    data["changed_after_execution"] = True
    metrics_path.write_text(json.dumps(data))
    result = aggregate_hcu_analysis(out)
    assert result["status"] == "failed"
    assert result["agent_findings_status"] == "incomplete"


@pytest.mark.parametrize(
    "relative",
    [
        "perf_report_trace1_csvs/unified_perf_summary.csv",
        "perf_report_trace2_csvs/unified_perf_summary.csv",
        "metadata/trace2_gpu_utilization.json",
    ],
)
def test_comparative_receipts_reject_changes_to_either_trace_or_utilization(
    tmp_path, relative
):
    out = _prepared(tmp_path)
    manifest_path = out / "category_data" / "category_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["comparison_scope"] = "comparative"
    manifest_path.write_text(json.dumps(manifest))
    for directory in ("perf_report_trace1_csvs", "perf_report_trace2_csvs"):
        (out / directory).mkdir()
        (out / directory / "unified_perf_summary.csv").write_text(
            (out / "category_data" / "custom_ops.csv").read_text()
        )
    (out / "metadata" / "trace2_gpu_utilization.json").write_text(
        json.dumps(manifest["gpu_utilization"])
    )
    _reviewed_custom(out)
    changed = out / relative
    changed.write_text(changed.read_text() + "\n ")
    result = aggregate_hcu_analysis(out)
    assert result["status"] == "failed"
    assert result["agent_findings_status"] == "incomplete"


def test_missing_execution_receipt_never_accepts_old_success(tmp_path):
    out = _prepared(tmp_path)
    _reviewed_custom(out)
    (out / "native_category_logs" / "custom.execution.json").unlink()
    assert aggregate_hcu_analysis(out)["status"] == "failed"


def test_same_inputs_and_identical_rerun_metrics_allow_review_reuse(tmp_path):
    out = _prepared(tmp_path)
    _reviewed_custom(out)
    assert analyze_prepared_categories(out)[0]["exit_code"] == 0
    assert aggregate_hcu_analysis(out)["agent_findings_status"] == "complete"


def test_reprepare_different_trace_requires_analysis_and_new_agent_review(tmp_path):
    out = _prepared(tmp_path)
    _reviewed_custom(out)
    trace = out / "trace.json"
    trace.write_text(json.dumps({"traceEvents": [{"name": "new_source"}]}))
    manifest_path = out / "category_data" / "category_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["prepared_sources"] = prepared_source_bindings(trace, out / "site.json")
    manifest_path.write_text(json.dumps(manifest))
    before = aggregate_hcu_analysis(out)
    assert before["status"] == "failed"
    assert before["agent_findings_status"] == "incomplete"
    assert analyze_prepared_categories(out)[0]["exit_code"] == 0
    after = aggregate_hcu_analysis(out)
    assert after["status"] == "metrics_complete"
    assert after["agent_findings_status"] == "incomplete"
    record_hcu_agent_review(out)
    assert aggregate_hcu_analysis(out)["agent_findings_status"] == "complete"


def test_changed_findings_need_explicit_agent_review(tmp_path):
    out = _prepared(tmp_path)
    finding = _reviewed_custom(out)
    finding.write_text(finding.read_text() + "\nChanged reasoning.\n")
    assert aggregate_hcu_analysis(out)["agent_findings_status"] == "incomplete"
    record_hcu_agent_review(out)
    assert aggregate_hcu_analysis(out)["agent_findings_status"] == "complete"


def test_report_requires_current_explicit_source_and_content_binding(tmp_path):
    out = _prepared(tmp_path)
    _reviewed_custom(out)
    report = out / "analysis.md"
    report.write_text("""# HCU reviewed test report

<!-- report-begin kind=report_mode mode=agentic -->
<!-- report-end -->

## Executive Summary

| Metric | Value |
|---|---|
| Total Time | 0.1 ms |
| Compute % | 20% |
| Idle % | unknown |
| Exposed Communication % | unknown |
| Top Bottleneck Category | custom |

## Compute Kernel Optimizations

<!-- impact-begin kind=top_ops -->
| Rank | Category | Observed time |
|---|---|---|
| 1 | custom | 0.02 ms |
<!-- impact-end -->

## Kernel Fusion Opportunities (Experimental)

No established opportunity in this fixture.

## System-Level Optimizations

No established observation in this fixture.

## Detailed Analysis

Reviewed synthetic unknown-work fixture.

## Appendix

Work and achievable speedup remain unknown.
""")
    assert aggregate_hcu_analysis(out)["agent_report_status"] == "invalid"
    record_hcu_agent_review(out, include_report=True)
    assert aggregate_hcu_analysis(out)["agent_report_status"] == "complete"
    report.write_text(report.read_text() + "\nNew reasoning.\n")
    assert aggregate_hcu_analysis(out)["agent_report_status"] == "invalid"


def test_hcu_unqualified_fusion_report_requires_nonquantified_impact(tmp_path):
    out = _prepared(tmp_path)
    (out / "category_data" / "kernel_fusion_metrics.json").write_text(
        json.dumps({"status": "UNQUALIFIED", "reference_kind": "measured_envelope"})
    )
    content = "## Kernel Fusion Opportunities (Experimental)\n\n**Impact**: Not quantifiable from trace data.\n"
    assert not _validate_report_comparison_scope_diffs(content, str(out), "standalone")
    numeric = content.replace(
        "Not quantifiable from trace data.",
        "impact_score: 40.0 (perf-model coverage 2/2 kernels)",
    )
    assert _validate_report_comparison_scope_diffs(numeric, str(out), "standalone")
    manifest_path = out / "category_data" / "category_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("hcu_roofline")
    manifest_path.write_text(json.dumps(manifest))
    assert _validate_report_comparison_scope_diffs(content, str(out), "standalone")


def test_hcu_cpu_operation_count_is_not_a_kernel_count():
    frame = pd.DataFrame([{"Count": 4, "total_direct_kernel_time_sum": 20}])
    patterns = analyze_kernel_patterns(frame, hcu=True)
    assert patterns["availability"] == "unavailable"
    assert patterns["total_kernel_count"] is None
    assert patterns["short_kernel_count"] is None


def test_reviewed_negative_system_findings_need_no_fabricated_recommendation(tmp_path):
    (tmp_path / "system_findings").mkdir()
    (tmp_path / "category_data").mkdir()
    findings = tmp_path / "system_findings" / "multi_kernel_findings.md"
    findings.write_text(
        "# Reviewed negative result\n\nNo issues were flagged by native metrics.\n\n## Recommendations\n\n## Detailed Analysis\n"
    )
    metrics = {
        "status": "SUCCESS",
        "patterns_detected": [],
        **{
            key: {"flagged": False}
            for key in (
                "memcpy_assessment",
                "nccl_blocking_assessment",
                "overlap_assessment",
            )
        },
    }
    path = tmp_path / "category_data" / "multi_kernel_metrics.json"
    path.write_text(json.dumps(metrics))
    assert validate_findings_file(str(findings), "system")[0]
    path.write_text(json.dumps({"status": "ERROR"}))
    assert not validate_findings_file(str(findings), "system")[0]
