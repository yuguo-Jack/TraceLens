###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Synthetic Kineto contracts; no hardware or vendor-dependent imports required."""

from copy import deepcopy

import pytest

from TraceLens.Trace2Tree.trace_to_tree import TraceToTree
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer


def trace(runtime_category, launch_name="hipLaunchKernel", gpu_count=1):
    events = [
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": "aten::mm",
            "ts": 0,
            "dur": 40,
            "pid": 10,
            "tid": 11,
            "args": {"Input Dims": [[4, 8], [8, 4]]},
        },
        {
            "ph": "X",
            "cat": runtime_category,
            "name": launch_name,
            "ts": 2,
            "dur": 2,
            "pid": 10,
            "tid": 11,
            "args": {"correlation": 71},
        },
    ]
    for i in range(gpu_count):
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": f"gemm_{i}",
                "ts": 5 + i * 10,
                "dur": 8,
                "pid": 0,
                "tid": 3,
                "args": {"correlation": 71, "stream": 3},
            }
        )
    return events


@pytest.mark.parametrize(
    "category", ["hip_runtime", "hip_driver", "cuda_runtime", "cuda_driver"]
)
def test_runtime_category_links_kernel_and_cpu_shape(category):
    events = trace(category)
    tree = TraceToTree(deepcopy(events), prune_nongpu_paths=False)
    tree.build_tree()
    assert tree.linking_key == "correlation"
    assert tree.events[2]["parent"] == 1
    assert tree.events[1]["parent"] == 0
    assert tree.events[0]["gpu_events"] == [2]
    assert tree.events[1]["cat"] == category  # Never disguise HIP as CUDA.
    assert "gpu_events" not in events[0]  # Input evidence is not rewritten.


def test_hip_graph_launch_links_all_correlated_gpu_events():
    tree = TraceToTree(
        trace("hip_runtime", "hipGraphLaunch", gpu_count=2), prune_nongpu_paths=False
    )
    tree.build_tree()
    assert tree.events[0]["gpu_events"] == [2, 3]


def test_non_graph_ambiguous_correlation_stays_unlinked():
    tree = TraceToTree(trace("hip_driver", gpu_count=2), prune_nongpu_paths=False)
    tree.build_tree()
    assert not tree.events[0].get("gpu_events")
    assert all("parent" not in tree.events[i] for i in [2, 3])


@pytest.mark.parametrize(
    "category", ["hip_runtime", "hip_driver", "cuda_runtime", "cuda_driver"]
)
def test_runtime_leaf_operator_remains_eligible_for_shape_model(category):
    tree = TraceToTree(trace(category), prune_nongpu_paths=False)
    analyzer = TreePerfAnalyzer(tree, add_python_func=False)
    assert analyzer._is_leaf_cpu_op(tree.events[0])
    launchers = analyzer.get_kernel_launchers()
    assert len(launchers) == 1 and launchers[0]["name"] == "aten::mm"
    assert launchers[0]["total_direct_kernel_time"] == 8
