###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Run native category metrics from a prepared HCU manifest and audit outputs.

This executes deterministic native Python analyzers. Agent findings and the
stakeholder report remain separate deliverables; missing findings stay WARN.
Categories come from the actual prepared trace, never a mandatory agent roster.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from TraceLens.Agent.Analysis.utils.report_utils import (
    generate_priority_data,
    load_manifest,
)
from TraceLens.Agent.Analysis.utils.artifact_bindings import (
    analysis_source_bindings,
    category_source_bindings,
    file_fingerprint,
)
from TraceLens.Agent.Analysis.utils.validation_utils import (
    validate_findings_file,
    validate_report,
    validate_subagent_outputs,
)


def category_analysis_command(category, output_dir, comparison_scope="standalone"):
    """Build the compatible native command for one observed category."""
    name = category["name"]
    if not isinstance(name, str) or re.fullmatch(r"[a-z0-9_]+", name) is None:
        raise ValueError("Invalid native category name")
    base = [sys.executable, "-m", "TraceLens.Agent.Analysis.category_analyses."]
    extra = []
    if name in ("cpu_idle", "multi_kernel"):
        module = name + "_analysis"
    elif name == "kernel_fusion":
        module = "kernel_fusion_analysis"
        extra = ["--comparison-scope", comparison_scope]
    else:
        skill = category.get("skill", "generic-op-analyzer")
        modules = {
            "norm-analyzer": "norm_analysis",
            "sdpa-analyzer": "sdpa_analysis",
            "convolution-analyzer": "convolution_analysis",
            "moe-analyzer": "moe_analysis",
        }
        if skill in modules:
            module = modules[skill]
            extra = ["--category", name]
        elif name in ("gemm", "elementwise", "reduce", "triton"):
            module = name + "_analysis"
        elif category.get("tier") == "compute_kernel":
            module = "other_analysis"
            extra = ["--category", name]
        else:
            raise ValueError("No native analyzer for observed system category: " + name)
        extra += ["--comparison_scope", comparison_scope]
    base[-1] += module
    return base + ["--output-dir", str(Path(output_dir).resolve())] + extra


def analyze_prepared_categories(
    output_dir, categories=None, category_timeout_s=60, analysis_timeout_s=240
):
    """Execute selected manifest categories and record actual commands/statuses."""
    if not 0 < category_timeout_s <= 300:
        raise ValueError("category_timeout_s must be greater than 0 and at most 300")
    if not 0 < analysis_timeout_s <= 300:
        raise ValueError("analysis_timeout_s must be greater than 0 and at most 300")
    manifest = load_manifest(str(output_dir))
    if "hcu_roofline" not in manifest:
        raise ValueError(
            "Use HCU preparation with --gpu-arch-json before HCU category analysis"
        )
    observed = manifest.get("categories", [])
    try:
        common_bindings = analysis_source_bindings(output_dir, manifest)
        binding_error = None
    except (OSError, ValueError, KeyError) as error:
        common_bindings = None
        binding_error = str(error)
    requested = set(categories) if categories is not None else None
    if requested is not None:
        absent = requested - {entry["name"] for entry in observed}
        if absent:
            raise ValueError(
                "Categories are absent from the trace manifest: "
                + ", ".join(sorted(absent))
            )
    results = []
    deadline = time.monotonic() + analysis_timeout_s
    for category in observed:
        name = category["name"]
        if not isinstance(name, str) or re.fullmatch(r"[a-z0-9_]+", name) is None:
            raise ValueError("Invalid native category name")
        if requested is not None and name not in requested:
            continue
        log_dir = Path(output_dir) / "native_category_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = Path(output_dir) / "category_data" / (name + "_metrics.json")
        command = None
        input_bindings = None
        started = time.monotonic()
        remaining = deadline - started
        execution_timeout_s = (
            min(category_timeout_s, remaining) if remaining > 0 else None
        )
        try:
            command = category_analysis_command(
                category, output_dir, manifest.get("comparison_scope", "standalone")
            )
            # Remove stale output before execution so an earlier successful
            # file cannot hide a failed run or a script which wrote nothing.
            if metrics_path.is_file():
                metrics_path.unlink()
            if binding_error:
                raise ValueError(binding_error)
            input_bindings = category_source_bindings(
                output_dir, category, command, common_bindings
            )
            if remaining <= 0:
                result = {
                    "category": name,
                    "command": command,
                    "exit_code": None,
                    "metrics_status": "TIME_BUDGET_EXHAUSTED",
                    "metrics_file": str(metrics_path),
                    "error": "Total CPU analysis budget exhausted before this category started",
                    "started": False,
                }
                (log_dir / (name + ".log")).write_text(
                    result["error"], encoding="utf-8"
                )
                result.update(
                    elapsed_s=0,
                    category_timeout_s=category_timeout_s,
                    analysis_timeout_s=analysis_timeout_s,
                    execution_timeout_s=None,
                )
                results.append(result)
                (log_dir / (name + ".execution.json")).write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
                continue
            process_env = dict(os.environ)
            process_env["PYTHONUTF8"] = "1"
            # subprocess.run kills and waits for the direct Python analyzer on
            # timeout. Native CPU analyzers do not launch hardware workers.
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=process_env,
                check=False,
                timeout=execution_timeout_s,
            )
            (log_dir / (name + ".log")).write_text(
                completed.stdout + completed.stderr, encoding="utf-8"
            )
            status = "MISSING_OUTPUT"
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if not isinstance(metrics, dict):
                    raise ValueError("Native metrics JSON must contain an object")
                status = metrics.get("status", "UNKNOWN")
            result = {
                "category": name,
                "command": command,
                "exit_code": completed.returncode,
                "metrics_status": status,
                "metrics_file": str(metrics_path),
            }
        except subprocess.TimeoutExpired as error:

            def decoded(value):
                return (
                    value.decode("utf-8", errors="replace")
                    if isinstance(value, bytes)
                    else (value or "")
                )

            (log_dir / (name + ".log")).write_text(
                decoded(error.stdout) + decoded(error.stderr), encoding="utf-8"
            )
            # A timed-out analyzer may have written a partial success file.
            if metrics_path.is_file():
                metrics_path.unlink()
            result = {
                "category": name,
                "command": command,
                "exit_code": None,
                "metrics_status": "TIMEOUT",
                "metrics_file": str(metrics_path),
                "error": str(error),
                "timed_out": True,
                "direct_child_reaped": True,
            }
        except (OSError, ValueError) as error:
            result = {
                "category": name,
                "command": command,
                "exit_code": None,
                "metrics_status": "ERROR",
                "metrics_file": str(metrics_path),
                "error": str(error),
            }
        result["elapsed_s"] = round(time.monotonic() - started, 3)
        result["category_timeout_s"] = category_timeout_s
        result["analysis_timeout_s"] = analysis_timeout_s
        result["execution_timeout_s"] = execution_timeout_s
        result["input_bindings"] = input_bindings
        result["metrics_binding"] = (
            file_fingerprint(metrics_path) if metrics_path.is_file() else None
        )
        results.append(result)
        (log_dir / (name + ".execution.json")).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    return results


def _verified_execution(output_dir, category, manifest, common):
    """Require current inputs and the exact output of a successful execution."""
    output_dir = Path(output_dir)
    name = category["name"]
    path = output_dir / "native_category_logs" / (name + ".execution.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    command = category_analysis_command(
        category, output_dir, manifest.get("comparison_scope", "standalone")
    )
    current = category_source_bindings(output_dir, category, command, common)
    if (
        not isinstance(receipt, dict)
        or receipt.get("exit_code") != 0
        or receipt.get("command") != command
    ):
        raise ValueError("Successful native execution receipt is missing or invalid")
    if receipt.get("metrics_status") not in {"OK", "SUCCESS", "NO_DATA", "UNQUALIFIED"}:
        raise ValueError(
            "Native execution did not produce qualified successful metrics"
        )
    if receipt.get("input_bindings") != current:
        raise ValueError("Native execution inputs changed; reanalyze this category")
    metrics = file_fingerprint(output_dir / "category_data" / (name + "_metrics.json"))
    if receipt.get("metrics_binding") != metrics:
        raise ValueError(
            "Native metrics changed after execution; reanalyze this category"
        )
    return {"input_bindings": current, "metrics_binding": metrics}


def _finding_path(output_dir, category):
    return (
        Path(output_dir)
        / (
            "category_findings"
            if category.get("tier") == "compute_kernel"
            else "system_findings"
        )
        / (category["name"] + "_findings.md")
    )


def _finding_tier(category):
    return (
        "compute"
        if category.get("tier") == "compute_kernel"
        else ("fusion" if category["name"] == "kernel_fusion" else "system")
    )


def record_hcu_agent_review(output_dir, categories=None, include_report=False):
    """Bind findings after an Agent explicitly accepts the current reasoning.

    This records the caller's review declaration, not an automatic LLM run.
    Analysis/aggregation must never call it implicitly to accept stale prose.
    """
    output_dir = Path(output_dir)
    if include_report and categories is not None:
        raise ValueError("--include-report requires review of all observed categories")
    manifest = load_manifest(str(output_dir))
    common = analysis_source_bindings(output_dir, manifest)
    requested = set(categories) if categories is not None else None
    observed = manifest["categories"]
    if requested is not None and requested - {entry["name"] for entry in observed}:
        raise ValueError("Review categories must be observed in the current manifest")
    bindings = {}
    for category in observed:
        if requested is not None and category["name"] not in requested:
            continue
        execution = _verified_execution(output_dir, category, manifest, common)
        finding = _finding_path(output_dir, category)
        passed, errors = validate_findings_file(
            str(finding),
            _finding_tier(category),
            manifest.get("comparison_scope", "standalone"),
        )
        if not passed:
            raise ValueError("Invalid Agent findings: " + "; ".join(errors))
        binding = {
            **execution,
            "finding_binding": file_fingerprint(finding),
            "review_kind": "explicit_agent_review_declaration",
            "automatic_agent_invocation": False,
        }
        bindings[category["name"]] = binding
        (
            output_dir / "native_category_logs" / (category["name"] + ".findings.json")
        ).write_text(json.dumps(binding, indent=2), encoding="utf-8")
    if include_report:
        generate_priority_data(str(output_dir))
        passed, errors = validate_report(
            str(output_dir), manifest.get("comparison_scope", "standalone")
        )
        if not passed:
            raise ValueError("Invalid Agent report: " + "; ".join(errors))
        report_binding = {
            "category_bindings": bindings,
            "report_binding": file_fingerprint(output_dir / "analysis.md"),
            "review_kind": "explicit_agent_review_declaration",
            "automatic_agent_invocation": False,
        }
        (output_dir / "agent_report_binding.json").write_text(
            json.dumps(report_binding, indent=2), encoding="utf-8"
        )
    return {
        "status": "agent_outputs_bound",
        "categories": list(bindings),
        "report_bound": include_report,
        "automatic_agent_invocation": False,
    }


def aggregate_hcu_analysis(output_dir):
    """Aggregate native priorities, HCU screening diagnostics and real validation."""
    output_dir = Path(output_dir)
    manifest = load_manifest(str(output_dir))
    if "hcu_roofline" not in manifest:
        raise ValueError("Prepared manifest has no HCU qualification contract")
    expected = manifest.get("categories", [])
    metric_statuses = []
    failures = []
    allowed = {"OK", "SUCCESS", "NO_DATA", "UNQUALIFIED"}
    executions = {}
    try:
        common = analysis_source_bindings(output_dir, manifest)
        common_error = None
    except (OSError, ValueError, KeyError) as error:
        common = None
        common_error = str(error)
    for category in expected:
        path = output_dir / "category_data" / (category["name"] + "_metrics.json")
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(metrics, dict):
                raise ValueError("Native metrics JSON must contain an object")
            status = metrics.get("status", "UNKNOWN")
            if status not in allowed:
                failures.append(category["name"] + ": " + str(status))
        except (OSError, ValueError) as error:
            status = "MISSING_OR_INVALID_OUTPUT"
            failures.append(category["name"] + ": " + str(error))
        metric_statuses.append(
            {"category": category["name"], "status": status, "metrics_file": str(path)}
        )
    for category in expected:
        try:
            if common_error:
                raise ValueError(common_error)
            executions[category["name"]] = _verified_execution(
                output_dir, category, manifest, common
            )
        except (OSError, ValueError, KeyError) as error:
            failures.append(category["name"] + ": " + str(error))
    audit_path = output_dir / "category_data" / "hcu_roofline.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    headroom = sorted(
        audit.get("operations", []),
        key=lambda row: (
            row.get("conditional_screening_score") or 0,
            row.get("observed_kernel_time_sum_us") or 0,
        ),
        reverse=True,
    )
    priority_path = Path(generate_priority_data(str(output_dir)))
    priorities = json.loads(priority_path.read_text(encoding="utf-8"))
    priorities["hcu_headroom"] = {
        "reference_kind": "measured_envelope",
        "screening_score_basis": "conditional_kernel_gap_over_summed_kernel_time_not_E2E_savings",
        "operations": headroom,
        "achievable_e2e_savings_pct": None,
        "aggregate_e2e_speedup": None,
    }
    priority_path.write_text(
        json.dumps(priorities, indent=2, allow_nan=False), encoding="utf-8"
    )
    validation = validate_subagent_outputs(str(output_dir))
    findings_validation = []
    current_finding_bindings = {}
    for category in expected:
        name = category["name"]
        tier = _finding_tier(category)
        finding = _finding_path(output_dir, category)
        if finding.is_file():
            valid, errors = validate_findings_file(
                str(finding), tier, manifest.get("comparison_scope", "standalone")
            )
            try:
                current = {
                    **executions[name],
                    "finding_binding": file_fingerprint(finding),
                    "review_kind": "explicit_agent_review_declaration",
                    "automatic_agent_invocation": False,
                }
                declared = json.loads(
                    (
                        output_dir / "native_category_logs" / (name + ".findings.json")
                    ).read_text(encoding="utf-8")
                )
                if current != declared:
                    raise ValueError(
                        "Agent findings binding changed; review current inputs and findings again"
                    )
                current_finding_bindings[name] = current
            except (OSError, ValueError, KeyError) as error:
                valid = False
                errors.append("Missing or stale Agent review binding: " + str(error))
            findings_validation.append(
                {
                    "category": category["name"],
                    "status": "PASS" if valid else "FAIL",
                    "errors": errors,
                }
            )
        else:
            findings_validation.append(
                {
                    "category": category["name"],
                    "status": "MISSING",
                    "errors": ["Agent findings have not been written"],
                }
            )
    report_status = "not_generated"
    report_errors = []
    if (output_dir / "analysis.md").is_file():
        report_valid, report_errors = validate_report(
            str(output_dir), manifest.get("comparison_scope", "standalone")
        )
        try:
            declared = json.loads(
                (output_dir / "agent_report_binding.json").read_text(encoding="utf-8")
            )
            if (
                len(current_finding_bindings) != len(expected)
                or declared.get("category_bindings") != current_finding_bindings
                or declared.get("report_binding")
                != file_fingerprint(output_dir / "analysis.md")
            ):
                raise ValueError(
                    "Agent report sources or content changed; review the current report again"
                )
        except (OSError, ValueError, KeyError, AttributeError) as error:
            report_valid = False
            report_errors.append("Missing or stale Agent report binding: " + str(error))
        report_status = "complete" if report_valid else "invalid"
    result = {
        "status": "failed" if failures else "metrics_complete",
        "native_category_metrics": metric_statuses,
        "failures": failures,
        "native_validation": validation,
        "agent_findings_status": (
            "complete"
            if all(item["status"] == "PASS" for item in findings_validation)
            else "incomplete"
        ),
        "agent_findings_validation": findings_validation,
        "agent_report_status": report_status,
        "agent_report_errors": report_errors,
        "roofline_summary": audit["summary"],
        "priority_file": str(priority_path),
        "achievable_e2e_savings_pct": None,
    }
    (output_dir / "hcu_native_analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Native HCU category analysis and aggregate validation"
    )
    parser.add_argument("stage", choices=("analyze", "aggregate", "run", "review"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--category",
        action="append",
        help="Analyze only an observed category; repeat for independent workers",
    )
    parser.add_argument(
        "--category-timeout-s",
        type=float,
        default=60,
        help="CPU analyzer timeout per category, greater than 0 and at most 300 seconds (default: 60)",
    )
    parser.add_argument(
        "--analysis-timeout-s",
        type=float,
        default=240,
        help="Total CPU category execution budget, greater than 0 and at most 300 seconds (default: 240)",
    )
    parser.add_argument(
        "--include-report",
        action="store_true",
        help="Explicit review also binds analysis.md to all current category findings",
    )
    args = parser.parse_args(argv)
    if args.stage == "aggregate" and args.category:
        parser.error(
            "--category applies to analyze/run; aggregate audits every observed category"
        )
    if args.include_report and args.stage != "review":
        parser.error("--include-report applies only to explicit Agent review")
    if args.stage == "review":
        print(
            json.dumps(
                record_hcu_agent_review(
                    args.output_dir, args.category, args.include_report
                ),
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    if args.stage in ("analyze", "run"):
        executions = analyze_prepared_categories(
            args.output_dir,
            args.category,
            args.category_timeout_s,
            args.analysis_timeout_s,
        )
    if args.stage in ("aggregate", "run"):
        result = aggregate_hcu_analysis(args.output_dir)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if result["status"] == "failed" else 0
    print(json.dumps({"categories": executions}, indent=2, allow_nan=False))
    return (
        0
        if all(
            item.get("exit_code") == 0
            and item["metrics_status"] not in {"ERROR", "MISSING_OUTPUT", "UNKNOWN"}
            for item in executions
        )
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
