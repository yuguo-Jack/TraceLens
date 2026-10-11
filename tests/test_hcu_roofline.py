###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HCU work/peak admission and arithmetic; no accelerator or network required."""

import copy
import json

import pandas as pd
import pytest

from TraceLens.PerfModel.hcu_roofline import (
    evaluate_hcu_roofline_row,
    export_hcu_roofline,
    main,
    qualify_hcu_arch,
)
from TraceLens.Reporting.reporting_utils import resolve_gpu_arch


@pytest.fixture
def hcu_arch():
    return {
        "name": "BW1000_H",
        "mem_bw_gbps": 100,
        "max_achievable_tflops": {"matrix_fp32": 2, "matrix_bf16": 8},
        "hcu_roofline": {
            "schema_version": 1,
            "device_arch": "gfx936:sramecc+:xnack-",
            "provenance": {
                "kind": "measured",
                "source": "qualification/calibration.json",
                "device_name": "BW1000_H",
                "device_arch": "gfx936",
                "timestamp": "2026-10-11T08:00:00Z",
                "stack": {"torch": "2.11", "hip": "7.2"},
                "conditions": {
                    "compute": {"matrix_fp32": {"shape": [4096, 4096, 4096]}}
                },
            },
            "workload_mappings": [
                {
                    "name": "aten::mm",
                    "compute_spec": "matrix_fp32",
                    "model": "TraceLens.PerfModel.perf_model.aten_mm",
                    "status": "qualified",
                    "evidence": "dense 2D mm formula and trace input/output contract review",
                }
            ],
        },
    }


@pytest.fixture
def mm_row():
    # FLOPs = 2*128*64*256; compulsory bytes = 4*(128*256 + 256*64 + 128*64).
    return {
        "name": "aten::mm",
        "Compute Spec": "matrix_fp32",
        "Input Dims": "[[128,256],[256,64]]",
        "Input type": "['float','float']",
        "perf_params": "{'M':128,'N':64,'K':256}",
        "GFLOPS": 4194304 / 1e9,
        "Data Moved (MB)": 229376 / 1024**2,
        "operation_count": 2,
        "Kernel Time (µs)_sum": 20,
        # Deliberately inconsistent precomputed fields must not drive HCU arithmetic.
        "Kernel Time (µs)_mean": 999,
        "Pct Roofline_mean": 99,
    }


def test_shape_dtype_model_contract(mm_row, hcu_arch):
    result = evaluate_hcu_roofline_row(mm_row, hcu_arch)
    assert result["status"] == "supported"
    assert result["observed_kernel_time_mean_us"] == 10
    assert result["measured_reference_time_us"] == pytest.approx(2.29376)
    assert result["reference_compute_time_us"] == pytest.approx(2.097152)
    assert result["reference_bound"] == "memory"
    assert result["efficiency_vs_measured_pct"] == pytest.approx(22.9376)
    assert result["conditional_headroom_ratio"] == pytest.approx(10 / 2.29376)
    assert result["conditional_kernel_time_headroom_us"] == pytest.approx(
        20 - 2 * 2.29376
    )
    assert result["calibration_shape_match"] is False
    assert result["achievable_e2e_savings_pct"] is None
    assert result["achievable_kernel_speedup"] is None
    assert result["theoretical_lower_bound_us"] is None


@pytest.mark.parametrize("field", ["GFLOPS", "Data Moved (MB)"])
def test_rejects_inconsistent_work(field, mm_row, hcu_arch):
    mm_row[field] *= 2
    result = evaluate_hcu_roofline_row(mm_row, hcu_arch)
    assert result["status"] == "unknown"
    assert any("mismatch" in item for item in result["missing_fields"])
    assert result["measured_reference_time_us"] is None


def test_unknown_dtype_never_uses_bf16_reference(mm_row, hcu_arch):
    del hcu_arch["max_achievable_tflops"]["matrix_fp32"]
    result = evaluate_hcu_roofline_row(mm_row, hcu_arch)
    assert result["status"] == "unknown"
    assert "compute_reference.matrix_fp32" in result["missing_fields"]
    assert result["efficiency_vs_measured_pct"] is None


def test_dtype_mapping_requires_local_model_match(mm_row, hcu_arch):
    mm_row["Input type"] = "['c10::BFloat16','c10::BFloat16']"
    assert (
        "workload_compute_spec_mismatch"
        in evaluate_hcu_roofline_row(mm_row, hcu_arch)["missing_fields"]
    )


def test_mapping_evidence_and_shape_scope_required(mm_row, hcu_arch):
    mapping = hcu_arch["hcu_roofline"]["workload_mappings"][0]
    mapping["input_dims"] = [[4096, 4096], [4096, 4096]]
    assert evaluate_hcu_roofline_row(mm_row, hcu_arch)["status"] == "unknown"
    del mapping["input_dims"]
    mapping["evidence"] = ""
    assert evaluate_hcu_roofline_row(mm_row, hcu_arch)["status"] == "unknown"


def test_json_mapping_selectors_match_native_tuple_metadata(mm_row, hcu_arch):
    mapping = hcu_arch["hcu_roofline"]["workload_mappings"][0]
    mapping.update(input_dims=[[128, 256], [256, 64]], input_type=["float", "float"])
    mm_row["Input Dims"] = "((128,256),(256,64))"
    mm_row["Input type"] = "('float','float')"
    assert evaluate_hcu_roofline_row(mm_row, hcu_arch)["status"] == "supported"


@pytest.mark.parametrize(
    "dims,types,issue",
    [
        ("[[128,256],[128,64]]", "['float','float']", "dense_2d_mm_shape_contract"),
        (
            "[[128,256],[256,64]]",
            "['float','c10::BFloat16']",
            "homogeneous_mm_input_dtype",
        ),
    ],
)
def test_dense_mm_contract_rejects_invalid_native_metadata(
    mm_row, hcu_arch, dims, types, issue
):
    mm_row.update({"Input Dims": dims, "Input type": types})
    assert issue in evaluate_hcu_roofline_row(mm_row, hcu_arch)["missing_fields"]


def test_unstructured_calibration_conditions_cannot_qualify_peaks(mm_row, hcu_arch):
    hcu_arch["hcu_roofline"]["provenance"]["conditions"] = "unstructured"
    assert qualify_hcu_arch(hcu_arch)["mem_bw_gbps"] is None
    hcu_arch["hcu_roofline"]["provenance"]["conditions"] = {
        "compute": {"matrix_fp32": 57.9}
    }
    assert (
        evaluate_hcu_roofline_row(mm_row, hcu_arch)["calibration_shape_match"] is None
    )


def test_ambiguous_mapping_is_not_silently_selected(mm_row, hcu_arch):
    mappings = hcu_arch["hcu_roofline"]["workload_mappings"]
    mappings.append(copy.deepcopy(mappings[0]))
    assert (
        "ambiguous_workload_mapping"
        in evaluate_hcu_roofline_row(mm_row, hcu_arch)["missing_fields"]
    )


@pytest.mark.parametrize("count", [None, 0, -1, 0.5, float("inf")])
def test_missing_or_invalid_count_stays_unknown(count, mm_row, hcu_arch):
    mm_row["operation_count"] = count
    result = evaluate_hcu_roofline_row(mm_row, hcu_arch)
    assert result["status"] == "unknown"
    assert result["conditional_headroom_ratio"] is None


@pytest.mark.parametrize("provenance", [None, "unverified", {"kind": "vendor"}])
def test_unqualified_architecture_cannot_feed_native_peaks(
    provenance, mm_row, hcu_arch
):
    hcu_arch["hcu_roofline"]["provenance"] = provenance
    qualified = qualify_hcu_arch(hcu_arch)
    assert qualified["mem_bw_gbps"] is None
    assert qualified["max_achievable_tflops"] == {}
    assert qualified["hcu_roofline"]["supplied_sustained"]["mem_bw_gbps"] == 100
    assert evaluate_hcu_roofline_row(mm_row, hcu_arch)["status"] == "unknown"
    assert resolve_gpu_arch(gpu_arch=hcu_arch)["max_achievable_tflops"] == {}


def test_device_identity_and_isa_mismatch_fail(hcu_arch):
    with pytest.raises(ValueError, match="platform"):
        qualify_hcu_arch(hcu_arch, expected_name="MI300X")
    with pytest.raises(ValueError, match="ISA"):
        qualify_hcu_arch(hcu_arch, expected_device_arch="gfx938")
    hcu_arch["hcu_roofline"]["provenance"]["device_name"] = "MI300X"
    assert qualify_hcu_arch(hcu_arch)["max_achievable_tflops"] == {}


def test_vendor_theory_is_separate_from_measured_envelope(mm_row, hcu_arch):
    hcu_arch["hcu_roofline"]["theoretical"] = {
        "mem_bw_gbps": 200,
        "max_achievable_tflops": {"matrix_fp32": 20},
        "provenance": {
            "kind": "vendor",
            "source": "vendor device specification",
            "device_name": "BW1000_H",
            "device_arch": "gfx936",
        },
    }
    result = evaluate_hcu_roofline_row(mm_row, hcu_arch)
    assert result["measured_reference_time_us"] == pytest.approx(2.29376)
    assert result["theoretical_lower_bound_us"] == pytest.approx(1.14688)
    assert result["theoretical_kernel_time_savings_ceiling_us"] == pytest.approx(
        20 - 2 * 1.14688
    )
    assert result["achievable_kernel_speedup"] is None
    assert result["achievable_e2e_savings_pct"] is None


def test_measured_envelope_can_be_exceeded_without_becoming_a_ceiling(mm_row, hcu_arch):
    mm_row["Kernel Time (µs)_sum"] = 2
    result = evaluate_hcu_roofline_row(mm_row, hcu_arch)
    assert result["efficiency_vs_measured_pct"] > 100
    assert result["conditional_kernel_time_headroom_us"] == 0
    assert result["guaranteed_upper_bound"] is False
    assert any("exceeds" in warning for warning in result["warnings"])


def test_aggregate_overlap_keeps_wallclock_separate(mm_row, hcu_arch):
    result = export_hcu_roofline(
        pd.DataFrame([mm_row, mm_row]), hcu_arch, wallclock_time_us=15
    )
    assert result["summary"]["observed_kernel_time_sum_us"] == 40
    assert result["summary"]["wallclock_time_us"] == 15
    assert result["summary"]["kernel_time_sum_is_wallclock"] is False
    assert result["summary"]["aggregate_e2e_speedup"] is None
    assert result["operations"][0]["share_of_summed_kernel_time_pct"] == 50
    assert result["operations"][0]["conditional_screening_score"] < 50


def test_missing_observed_time_is_unknown_in_summary_and_priority(mm_row, hcu_arch):
    missing = dict(mm_row, name="unsupported::custom", **{"Kernel Time (µs)_sum": None})
    result = export_hcu_roofline(pd.DataFrame([mm_row, missing]), hcu_arch)
    assert result["summary"]["observed_kernel_time_sum_us"] is None
    assert result["summary"]["observed_known_kernel_time_sum_us"] == 20
    assert result["summary"]["missing_observed_time_rows"] == 1
    assert result["operations"][1]["share_of_summed_kernel_time_pct"] is None
    assert result["operations"][0]["conditional_screening_score"] is None


def test_export_cli_writes_json_safe_unknowns(tmp_path, mm_row, hcu_arch):
    unknown = dict(mm_row, name="unsupported::custom", GFLOPS=float("nan"))
    csv = tmp_path / "ops.csv"
    pd.DataFrame([mm_row, unknown]).to_csv(csv, index=False)
    arch_path = tmp_path / "arch.json"
    arch_path.write_text(json.dumps(hcu_arch), encoding="utf-8")
    output = tmp_path / "headroom.json"
    main(
        ["--perf-csv", str(csv), "--arch-json", str(arch_path), "--output", str(output)]
    )
    result = json.loads(output.read_text())
    assert result["summary"]["supported_rows"] == 1
    assert result["summary"]["unknown_rows"] == 1
    assert result["operations"][1]["observed_kernel_time_mean_us"] == 10
    assert result["operations"][1]["measured_reference_time_us"] is None
    assert "NaN" not in output.read_text()


def test_non_hcu_architecture_contract_is_preserved():
    arch = {
        "name": "MI300X",
        "mem_bw_gbps": 5300,
        "max_achievable_tflops": {"matrix_bf16": 708},
    }
    assert qualify_hcu_arch(arch) == arch
    assert resolve_gpu_arch(gpu_arch=arch) == arch
