###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Format/accounting regressions authored from DTK format, not GPU benchmarks."""

from copy import deepcopy
import json

import pytest

from TraceLens.Reporting.hipprof import normalize_hipprof, generate_perf_report_hipprof


def event(
    cat, index, pid=81, begin=1700000000000000001, duration=321, name=None, **extra
):
    args = dict(
        BeginNs=str(begin),
        EndNs=str(begin + duration),
        DurationNs=duration,
        pid=pid,
        tid=88,
        index=index,
        **extra
    )
    if cat != "HIP":
        args.update(devId=0, queueId="0")
    return dict(
        ph="X",
        cat=cat,
        name=name or ("hipLaunchKernel" if cat == "HIP" else "gemm"),
        ts=10,
        dur=duration / 1000,
        pid=1,
        tid="Thread88",
        args=args,
    )


def test_worker_tid_is_not_process_and_epoch_ns_remain_exact():
    data, receipt = normalize_hipprof(
        {"traceEvents": [event("HIP", 9), event("HIPOPS", 9, procId=95)]}
    )
    assert receipt["unlinked_device_events"] == 0
    assert data["traceEvents"][1]["args"]["begin_ns"] == 1700000000000000001
    assert data["traceEvents"][1]["dur"] == 0.321


def test_correlation_ids_never_join_different_processes():
    data, receipt = normalize_hipprof(
        {"traceEvents": [event("HIP", 9, pid=81), event("HIPOPS", 9, pid=82)]}
    )
    assert receipt["status"] == "incomplete"
    assert (
        data["traceEvents"][0]["args"]["corr_id"]
        != data["traceEvents"][1]["args"]["corr_id"]
    )


def test_perfetto_end_can_precede_begin_after_microsecond_rounding():
    b = event("HIPOPS", 9, args="(4,2,1),(256,1,1)")
    b.update(ph="b", id2={"local": "0x3"}, ts=10)
    e = {k: v for k, v in b.items() if k != "args"}
    e.update(ph="e")
    out, receipt = normalize_hipprof({"traceEvents": [e, b]})
    assert receipt["output_events"] == 1
    assert out["traceEvents"][0]["args"]["grid"] == [4, 2, 1]
    assert out["traceEvents"][0]["args"]["block"] == [256, 1, 1]
    with pytest.raises(ValueError, match="Unpaired"):
        normalize_hipprof({"traceEvents": [b]})


@pytest.mark.parametrize(
    "change", [{"DurationNs": 322}, {"BeginNs": "nan"}, {"pid": True}]
)
def test_bad_timing_and_identity_are_rejected(change):
    e = event("HIPOPS", 1)
    e["args"].update(change)
    with pytest.raises(ValueError):
        normalize_hipprof({"traceEvents": [e]})


def test_memory_set_not_reported_as_copy_and_peer_end_not_invented(tmp_path):
    events = [
        event("HIP", 1),
        event("HIPOPS", 1),
        event("HIPCOPY", 2, name="Memset", Bytes=1024),
        event("HIPCOPY", 3, name="DeviceToDevice", Bytes=2048),
    ]
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": events}))
    tables = generate_perf_report_hipprof(trace, tmp_path / "report")
    assert len(tables["memory_set_summary"]) == 1
    assert tables["memory_copy_summary"].iloc[0]["direction"] == "d2d (GPU ? -> GPU ?)"
    assert tables["memory_copy_summary"].iloc[0]["count"] == 1


def test_ambiguous_api_identity_is_not_last_record_wins():
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_hipprof(
            {"traceEvents": [event("HIP", 1), event("HIP", 1), event("HIPOPS", 1)]}
        )


def test_json_and_paired_pftrace_records_have_identical_kernel_time(tmp_path):
    raw = [event("HIP", 1), event("HIPOPS", 1, duration=432)]
    paired = []
    for n, e in enumerate(raw):
        b = deepcopy(e)
        b.update(ph="b", id2={"local": str(n)})
        end = {k: v for k, v in b.items() if k != "args"}
        end.update(ph="e")
        paired.extend([end, b])
    a, _ = normalize_hipprof({"traceEvents": raw})
    b, _ = normalize_hipprof({"traceEvents": paired})
    assert [e["args"]["delta_ns"] for e in a["traceEvents"]] == [
        e["args"]["delta_ns"] for e in b["traceEvents"]
    ]
