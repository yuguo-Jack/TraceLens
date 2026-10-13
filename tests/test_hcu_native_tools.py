###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Cross-platform contracts for native tools fed HIP traces and DTK installs."""

import json
from types import SimpleNamespace

import pytest

from TraceLens.TraceFusion.trace_fuse import TraceFuse
from TraceLens.TraceUtils.kernel_source import library_artifact
from TraceLens.PerfModel.benchmarking import microbench_utils
from TraceLens.TraceDiff.trace_diff import TraceDiff
from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TraceUtils.kernel_source import build_index
from tests.test_hip_runtime_categories import trace


@pytest.mark.parametrize("category", ["hip_runtime", "hip_driver"])
def test_tracediff_runtime_equivalence(category):
    trees = []
    for runtime in ("cuda_runtime", category):
        tree = TraceToTree(trace(runtime), prune_nongpu_paths=False)
        tree.build_tree()
        trees.append(tree)
    diff = TraceDiff(*trees)
    merged, _ = diff.merged_tree
    matched = [
        e for e in merged if e.get("uid1") is not None and e.get("uid2") is not None
    ]
    assert any(trees[0].get_UID2event(e["uid1"])["name"] == "aten::mm" for e in matched)


def test_self_comparison_retains_both_sources_for_report_enrichment():
    trees = []
    for _ in range(2):
        tree = TraceToTree(trace("hip_runtime"), prune_nongpu_paths=False)
        tree.build_tree()
        trees.append(tree)
    diff = TraceDiff(*trees)
    diff.generate_tracediff_report(collapse_identical=False)
    assert set(diff.diff_stats_df["source"]) == {"trace1", "trace2"}


@pytest.mark.parametrize("suffix", [".hip", ".cpp", ".cc", ".cxx"])
def test_hip_device_definitions_in_cpp_sources(tmp_path, suffix):
    source = tmp_path / ("kernels" + suffix)
    source.write_text("__global__ void hcu_probe(float* x) { x[threadIdx.x] = 1; }\n")
    index = build_index([str(tmp_path)])
    assert "hcu_probe" in index.symbol_index


@pytest.mark.parametrize(
    "category", ["hip_runtime", "hip_driver", "cuda_runtime", "cuda_driver"]
)
def test_fusion_preserves_rank_local_launch_links(tmp_path, category):
    files = []
    for rank in range(2):
        p = tmp_path / f"rank{rank}.json"
        p.write_text(json.dumps({"traceEvents": trace(category)}))
        files.append(str(p))
    fuser = TraceFuse(files)
    events = [e for e in fuser.merge() if e["ph"] == "X"]
    assert fuser.linking_key == "correlation"
    ids = []
    for rank in range(2):
        own = [e for e in events if e["args"]["rank"] == rank]
        runtime = next(e for e in own if e["cat"] == category)
        kernel = next(e for e in own if e["cat"] == "kernel")
        assert runtime["args"]["correlation"] == kernel["args"]["correlation"]
        ids.append(kernel["args"]["correlation"])
    assert ids[0] != ids[1]


def test_dtk_component_library_discovery(tmp_path, monkeypatch):
    dirs = [tmp_path / name / "lib/library" for name in ("rocblas", "hipblaslt")]
    for d in dirs:
        d.mkdir(parents=True)
    monkeypatch.delenv("TRACELENS_ROCM_LIBRARY_DIRS", raising=False)
    monkeypatch.setattr(library_artifact, "_rocm_bases", lambda: [tmp_path])
    assert set(library_artifact.discover_library_dirs()) == set(dirs)


def test_fusion_offsets_include_later_rank_ranges(tmp_path):
    files = []
    for rank, correlation in enumerate((1, 100, 0)):
        events = trace("hip_runtime")
        for event in events:
            if "correlation" in event.get("args", {}):
                event["args"]["correlation"] = correlation
        path = tmp_path / f"rank{rank}.json"
        path.write_text(json.dumps({"traceEvents": events}))
        files.append(str(path))
    merged = TraceFuse(files).merge()
    kernels = [e for e in merged if e.get("cat") == "kernel"]
    assert len({e["args"]["correlation"] for e in kernels}) == 3


@pytest.mark.parametrize(
    "util,used,expected",
    [
        ("0.0", "2", True),
        ("99", "2", False),
        ("0", "40000", False),
        ("N/A", "2", False),
    ],
)
def test_hcu_idle_metrics(tmp_path, monkeypatch, util, used, expected):
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        microbench_utils.shutil,
        "which",
        lambda name: "/opt/dtk/bin/hy-smi" if name == "hy-smi" else None,
    )
    payload = {
        "card0": {
            "HCU use (%)": util,
            "vram Total Used Memory (MiB)": used,
            "vram Total Memory (MiB)": "65520",
        }
    }
    monkeypatch.setattr(
        microbench_utils.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(payload)),
    )
    idle, explanation = microbench_utils.check_gpu_idle(0)
    assert idle is expected
    assert "hy-smi" in explanation


def test_hcu_visibility_ambiguity_is_not_idle(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "GPU-unknown")
    monkeypatch.setattr(
        microbench_utils.shutil, "which", lambda name: "/opt/dtk/bin/hy-smi"
    )
    idle, message = microbench_utils.check_gpu_idle(0)
    assert not idle
    assert "unresolved device mapping" in message


@pytest.mark.parametrize(
    "module_name",
    ["generate_perf_report_pytorch", "generate_perf_report_pytorch_inference"],
)
def test_short_kernel_helper_initializes_gpu_span_when_called_directly(module_name):
    import importlib
    import pandas as pd

    analyzer = SimpleNamespace(
        gpu_only=True,
        get_df_kernels=lambda: pd.DataFrame(
            {"Kernel name": ["hip_probe"], "Kernel duration (µs)": [50.0]}
        ),
        get_df_gpu_timeline=lambda: pd.DataFrame(
            {"type": ["total_time"], "time ms": [0.1]}
        ),
    )
    module = importlib.import_module("TraceLens.Reporting." + module_name)
    _, table = module.get_dfs_short_kernels(analyzer, short_kernel_threshold_us=1000)
    assert table["Short Kernel duration (µs) percent of total time"].iloc[
        0
    ] == pytest.approx(50.0)
