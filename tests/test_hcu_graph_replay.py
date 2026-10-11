###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import copy
import json
from unittest.mock import patch

import pytest

from TraceLens.Trace2Tree.training_graph import (
    TrainingGraphContractError,
    generate_training_graph_report,
    merge_training_capture_trace,
    sha256_file,
    training_annotations,
    main,
)
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    merge_capture_trace_into_graph,
)


def event(name, cat, ts, dur, *, tid=1, pid=1, args=None):
    return {
        "name": name,
        "cat": cat,
        "ph": "X",
        "ts": ts,
        "dur": dur,
        "pid": pid,
        "tid": tid,
        "args": args or {},
    }


def mm_metadata(shape, stride):
    return {
        "Input Dims": shape,
        "Input Strides": stride,
        "Input type": ["float", "float"],
        "Concrete Inputs": ["", ""],
    }


@pytest.fixture
def graph_bundle(tmp_path):
    # Backward runs on a worker thread and is outside the annotation's host
    # ancestry, while remaining inside the explicit capture API boundaries.
    capture = [
        event("tracelens.train_graph.capture:step:full", "user_annotation", 0, 80),
        event("hipStreamBeginCapture", "hip_runtime", 5, 1),
        event(
            "aten::mm",
            "cpu_op",
            10,
            10,
            args=mm_metadata([[2, 3], [3, 4]], [[3, 1], [4, 1]]),
        ),
        event(
            "hipLaunchKernel",
            "hip_runtime",
            12,
            1,
            args={"kernel": "forward_mm", "correlation": 1},
        ),
        event(
            "autograd::engine::evaluate_function:MmBackward0", "cpu_op", 29, 18, tid=2
        ),
        event(
            "aten::mm",
            "cpu_op",
            30,
            10,
            tid=2,
            args=mm_metadata([[3, 2], [2, 4]], [[1, 3], [4, 1]]),
        ),
        event(
            "hipExtModuleLaunchKernel",
            "hip_driver",
            32,
            1,
            tid=2,
            args={"kernel": "backward_mm", "correlation": 2},
        ),
        event("hipStreamEndCapture", "hip_runtime", 50, 1),
    ]
    replay = []
    for ts, corr in ((100, 10), (200, 20)):
        replay.extend(
            [
                event(
                    "tracelens.train_graph.replay:step:full", "user_annotation", ts, 10
                ),
                event(
                    "hipGraphLaunch",
                    "hip_runtime",
                    ts + 1,
                    2,
                    args={"correlation": corr},
                ),
                event(
                    "forward_mm",
                    "kernel",
                    ts + 5,
                    4,
                    pid=0,
                    tid=0,
                    args={"correlation": corr, "stream": 0, "device": 0},
                ),
                event(
                    "backward_mm",
                    "kernel",
                    ts + 10,
                    3,
                    pid=0,
                    tid=0,
                    args={"correlation": corr, "stream": 0, "device": 0},
                ),
                event(
                    "tracelens.train_graph.replay:step:full",
                    "gpu_user_annotation",
                    ts + 5,
                    8,
                    pid=0,
                    tid=0,
                ),
            ]
        )
    for filename, events in (("capture.json", capture), ("replay.json", replay)):
        (tmp_path / filename).write_text(json.dumps({"traceEvents": events}))
    signature = [
        {"name": "x", "shape": [2, 3], "dtype": "torch.float32", "stride": [3, 1]}
    ]
    manifest = {
        "schema": "tracelens.training_graph.v1",
        "mapping_contract": "serialized_single_stream",
        "replay_file": "replay.json",
        "replay_sha256": sha256_file(tmp_path / "replay.json"),
        "provenance": {
            "torch_version": "2.11.0",
            "hip_version": "7.2.26365",
            "device_type": "hcu",
            "signature_source": "runtime_tensor_metadata",
        },
        "captures": [
            {
                "capture_id": "step",
                "segment_id": "full",
                "phase": "train_step",
                "file": "capture.json",
                "sha256": sha256_file(tmp_path / "capture.json"),
                "input_signature": signature,
                "replay_input_signature": copy.deepcopy(signature),
            }
        ],
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(manifest))
    return tmp_path, path, manifest


def rewrite(graph_bundle, file, transform):
    folder, path, manifest = graph_bundle
    data = json.loads((folder / file).read_text())
    transform(data["traceEvents"])
    (folder / file).write_text(json.dumps(data))
    if file == "replay.json":
        manifest["replay_sha256"] = sha256_file(folder / file)
    else:
        manifest["captures"][0]["sha256"] = sha256_file(folder / file)
    path.write_text(json.dumps(manifest))


def test_full_mapping_retains_backward_shapes_and_source_indexes(graph_bundle):
    folder, path, _ = graph_bundle
    tree = merge_training_capture_trace(folder, path)
    contract = tree.trace_metadata["tracelens_training_graph_merge"]
    assert contract["status"] == "complete" and len(contract["instances"]) == 2
    assert contract["training_state_equivalence"] == "not_evaluated"
    kernels = [e for e in tree.events if e.get("cat") == "kernel"]
    assert len(kernels) == 4
    backward = [e for e in kernels if e["name"] == "backward_mm"]
    assert all(e["graph_attribution"]["capture_operator_tid"] == 2 for e in backward)
    assert all(
        e["graph_attribution"]["capture_operator_event_index"] == 5 for e in backward
    )
    assert all(
        e["graph_attribution"]["Input Strides"] == [[1, 3], [4, 1]] for e in backward
    )
    assert [e["graph_attribution"]["replay_event_index"] for e in kernels] == [
        2,
        3,
        7,
        8,
    ]
    for e in tree.events:
        assert len(e.get("gpu_events", [])) == len(set(e.get("gpu_events", [])))
        if "capture_origin" in e:
            assert e["dur"] == 0 and e["timing_origin"] == "capture_structure_only"


def test_existing_merge_entry_dispatches_training_contract(graph_bundle):
    folder, path, _ = graph_bundle
    tree = merge_capture_trace_into_graph(
        str(folder), str(path), str(folder / "replay.json")
    )
    assert tree.trace_metadata["tracelens_training_graph_merge"]["status"] == "complete"


def test_report_contains_native_models_and_exact_attribution(graph_bundle, tmp_path):
    folder, path, _ = graph_bundle
    tables = generate_training_graph_report(
        folder,
        path,
        output_csvs_dir=str(tmp_path / "report"),
        collective_analysis=False,
    )
    assert len(tables["graph_attribution"]) == 4
    assert len(tables["GEMM"]) == 2
    assert set(tables["ops_unique_args"]["name"]) == {"aten::mm"}
    sanity = tables["graph_sanity"].iloc[0]
    assert sanity["kl_kernel_count"] == sanity["up_kernel_count"] == 4
    assert (tmp_path / "report" / "graph_attribution.csv").is_file()


@pytest.mark.parametrize(
    "field,value",
    [
        ("mapping_contract", "greedy"),
        ("schema", "wrong"),
        ("captures", []),
        ("provenance", None),
    ],
)
def test_missing_or_invalid_contract_rejected(graph_bundle, field, value):
    folder, path, manifest = graph_bundle
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(TrainingGraphContractError):
        merge_training_capture_trace(folder, path)


def test_shape_change_rejected_instead_of_batch_rounding(graph_bundle):
    folder, path, manifest = graph_bundle
    manifest["captures"][0]["replay_input_signature"][0]["shape"][0] = 3
    path.write_text(json.dumps(manifest))
    with pytest.raises(TrainingGraphContractError, match="signatures differ"):
        merge_training_capture_trace(folder, path)


def test_digest_change_rejected(graph_bundle):
    folder, path, _ = graph_bundle
    (folder / "capture.json").write_text("{}")
    with pytest.raises(TrainingGraphContractError, match="digest"):
        merge_training_capture_trace(folder, path)


def test_extra_dispatch_rejected_instead_of_greedy_skip(graph_bundle):
    rewrite(
        graph_bundle,
        "capture.json",
        lambda es: es.append(
            event("hipLaunchKernel", "hip_runtime", 42, 1, args={"kernel": "extra"})
        ),
    )
    folder, path, _ = graph_bundle
    with pytest.raises(TrainingGraphContractError, match="counts differ"):
        merge_training_capture_trace(folder, path)


def test_missing_kernel_identity_rejected(graph_bundle):
    rewrite(graph_bundle, "capture.json", lambda es: es[3]["args"].pop("kernel"))
    folder, path, _ = graph_bundle
    with pytest.raises(
        TrainingGraphContractError, match="capture_kernel_identity_missing"
    ):
        merge_training_capture_trace(folder, path)


def test_reordered_kernel_sequence_rejected(graph_bundle):
    rewrite(graph_bundle, "replay.json", lambda es: es[3].update(name="forward_mm"))
    folder, path, _ = graph_bundle
    with pytest.raises(TrainingGraphContractError, match="ordered kernel identity"):
        merge_training_capture_trace(folder, path)


def test_multistream_replay_rejected(graph_bundle):
    rewrite(graph_bundle, "replay.json", lambda es: es[3]["args"].update(stream=1))
    folder, path, _ = graph_bundle
    with pytest.raises(TrainingGraphContractError, match="multistream"):
        merge_training_capture_trace(folder, path)


def test_wrong_host_thread_rejected(graph_bundle):
    rewrite(graph_bundle, "replay.json", lambda es: es[1].update(tid=8))
    folder, path, _ = graph_bundle
    with pytest.raises(TrainingGraphContractError, match="exactly one"):
        merge_training_capture_trace(folder, path)


def test_unannotated_graph_launch_rejected(graph_bundle):
    rewrite(
        graph_bundle,
        "replay.json",
        lambda es: es.append(
            event("hipGraphLaunch", "hip_runtime", 300, 1, args={"correlation": 30})
        ),
    )
    folder, path, _ = graph_bundle
    with pytest.raises(TrainingGraphContractError, match="partial"):
        merge_training_capture_trace(folder, path)


def test_duplicate_capture_contract_rejected(graph_bundle):
    folder, path, manifest = graph_bundle
    manifest["captures"].append(copy.deepcopy(manifest["captures"][0]))
    path.write_text(json.dumps(manifest))
    with pytest.raises(TrainingGraphContractError, match="unique"):
        merge_training_capture_trace(folder, path)


def test_nested_replay_annotations_rejected_as_ambiguous(graph_bundle):
    rewrite(
        graph_bundle,
        "replay.json",
        lambda es: es.append(
            event("tracelens.train_graph.replay:step:full", "user_annotation", 99, 20)
        ),
    )
    folder, path, _ = graph_bundle
    with pytest.raises(TrainingGraphContractError, match="distinct"):
        merge_training_capture_trace(folder, path)


def test_cli_rejection_is_nonzero_and_persists_reason(graph_bundle):
    folder, path, manifest = graph_bundle
    manifest["provenance"].pop("hip_version")
    path.write_text(json.dumps(manifest))
    report = folder / "result.json"
    assert (
        main(
            [
                "--manifest",
                str(path),
                "--capture-folder",
                str(folder),
                "--report",
                str(report),
            ]
        )
        == 1
    )
    result = json.loads(report.read_text())
    assert result["status"] == "rejected" and "hip_version" in result["reason"]


def test_gpu_annotation_duplicate_is_ignored():
    es = [event("tracelens.train_graph.replay:a:b", "gpu_user_annotation", 0, 5, pid=0)]
    assert training_annotations(es) == []


def test_overlapping_gpu_events_rejected(graph_bundle):
    rewrite(graph_bundle, "replay.json", lambda es: es[3].update(ts=107))
    folder, path, _ = graph_bundle
    with pytest.raises(TrainingGraphContractError, match="Overlapping"):
        merge_training_capture_trace(folder, path)


def test_missing_compute_shape_source_rejected(graph_bundle):
    rewrite(graph_bundle, "capture.json", lambda es: es[2]["args"].pop("Input Strides"))
    folder, path, _ = graph_bundle
    with pytest.raises(TrainingGraphContractError, match="shape/dtype/stride"):
        merge_training_capture_trace(folder, path)


def test_graph_report_sanity_failure_raises_instead_of_printing_pass(graph_bundle):
    folder, path, _ = graph_bundle
    bad = {
        "kl_time_pass": True,
        "up_time_pass": True,
        "kl_count_pass": False,
        "up_count_pass": True,
    }
    with patch(
        "TraceLens.Reporting.generate_perf_report_pytorch_inference.perf_report_sanity_check",
        return_value=bad,
    ):
        with pytest.raises(ValueError, match="sanity"):
            generate_training_graph_report(folder, path, collective_analysis=False)


def test_forward_backward_uid_links_are_remapped_per_replay(graph_bundle):
    def link(es):
        es[2]["args"]["Sequence number"] = 1
        es[4]["args"]["Sequence number"] = 1

    rewrite(graph_bundle, "capture.json", link)
    folder, path, _ = graph_bundle
    tree = merge_training_capture_trace(folder, path)
    forward = [
        e
        for e in tree.events
        if e.get("name") == "aten::mm" and e.get("capture_origin", {}).get("uid") == 2
    ]
    assert len(forward) == 2
    for fwd in forward:
        assert len(fwd["bwd_events"]) == 1
        bwd = tree.events_by_uid[fwd["bwd_events"][0]]
        assert bwd["name"] == "autograd::engine::evaluate_function:MmBackward0"
        assert bwd["fwd_event"] == fwd["UID"]
        assert bwd["capture_origin"]["uid"] == 4
