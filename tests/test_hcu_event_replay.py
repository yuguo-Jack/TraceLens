###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import copy
import json
from unittest.mock import patch

import pytest

from TraceLens.EventReplay.event_replay import EventReplayer
from TraceLens.EventReplay.hcu_replay import (
    benchmark_replayer,
    main,
    run_bundle,
    write_replay_bundle,
)
from TraceLens.EventReplay.safety import ReplayContractError, compare_outputs

torch = pytest.importorskip("torch")


def event(name="aten::mm", shapes=None, strides=None, types=None, concrete=None):
    return {
        "cat": "cpu_op",
        "name": name,
        "args": {
            "Input Dims": shapes if shapes is not None else [[2, 3], [3, 4]],
            "Input Strides": strides if strides is not None else [[3, 1], [4, 1]],
            "Input type": types if types is not None else ["float", "float"],
            "Concrete Inputs": concrete if concrete is not None else ["", ""],
        },
    }


def test_replay_returns_output_and_repro_export_does_not_destroy_ir():
    replay = EventReplayer(event(), device="cpu")
    expected = torch.mm(*replay.args)
    info = replay.get_repro_info()
    assert info["input_semantics"] == "synthetic" and "operator_schema" in info
    torch.testing.assert_close(replay.replay(), expected)
    torch.testing.assert_close(replay.replay(), expected)


def test_captured_transposed_values_verify_without_mutating_inputs():
    x = torch.randn(3, 2).t()
    w = torch.randn(3, 4)
    source = event(strides=[[1, 2], [4, 1]])
    replay = EventReplayer(
        source,
        device="cpu",
        input_semantics="captured",
        input_values={"self": x, "mat2": w},
    )
    before = x.clone()
    result = replay.verify(x @ w)
    assert replay.args[0].stride() == x.stride()
    torch.testing.assert_close(result, x @ w)
    torch.testing.assert_close(x, before)
    assert not result.requires_grad


@pytest.mark.parametrize(
    "dtype,type_name",
    [
        (torch.float32, "float"),
        (torch.float16, "c10::Half"),
        (torch.bfloat16, "c10::BFloat16"),
        (torch.float64, "double"),
    ],
)
def test_synthetic_dtype_and_stride_preserved(dtype, type_name):
    replay = EventReplayer(
        event(
            "aten::mul",
            shapes=[[2, 3], []],
            strides=[[1, 2], []],
            types=[type_name, "Scalar"],
            concrete=["", "2"],
        ),
        device="cpu",
    )
    assert replay.args[0].dtype == dtype and replay.args[0].stride() == (1, 2)
    torch.testing.assert_close(replay.replay(), replay.args[0] * 2)


def test_zero_stride_readonly_input_is_supported():
    replay = EventReplayer(
        event(
            "aten::mul",
            shapes=[[3, 2], []],
            strides=[[0, 1], []],
            types=["float", "Scalar"],
            concrete=["", "2"],
        ),
        device="cpu",
    )
    assert replay.args[0].stride() == (0, 1)
    torch.testing.assert_close(replay.replay(), replay.args[0] * 2)


def test_same_shape_backward_gelu_output_matches_autograd():
    x = torch.randn(2, 3, requires_grad=True)
    grad = torch.randn_like(x)
    expected = torch.autograd.grad(
        torch.nn.functional.gelu(x, approximate="tanh"), x, grad
    )[0]
    source = event(
        "aten::gelu_backward",
        shapes=[[2, 3], [2, 3], []],
        strides=[[3, 1], [3, 1], []],
        types=["float", "float", "Scalar"],
        concrete=["", "", "tanh"],
    )
    replay = EventReplayer(
        source,
        device="cpu",
        input_semantics="captured",
        input_values={"grad_output": grad, "self": x},
    )
    torch.testing.assert_close(replay.verify(expected), expected)


def test_same_shape_backward_softmax_output_matches_autograd():
    x = torch.randn(2, 3, requires_grad=True)
    output = torch.softmax(x, dim=1)
    grad = torch.randn_like(output)
    expected = torch.autograd.grad(output, x, grad)[0]
    source = event(
        "aten::_softmax_backward_data",
        shapes=[[2, 3], [2, 3], [], []],
        strides=[[3, 1], [3, 1], [], []],
        types=["float", "float", "Scalar", "Scalar"],
        concrete=["", "", "1", "6"],
    )
    replay = EventReplayer(
        source,
        device="cpu",
        input_semantics="captured",
        input_values={"grad_output": grad, "output": output},
    )
    torch.testing.assert_close(replay.verify(expected), expected)


def test_same_shape_backward_layernorm_tuple_output_matches_autograd():
    x = torch.randn(2, 3, requires_grad=True)
    weight = torch.randn(3, requires_grad=True)
    bias = torch.randn(3, requires_grad=True)
    grad = torch.randn_like(x)
    output, mean, rstd = torch.native_layer_norm(x, [3], weight, bias, 1e-5)
    expected = torch.autograd.grad(output, (x, weight, bias), grad)
    shapes = [[2, 3], [2, 3], [], [2, 1], [2, 1], [3], [3], []]
    strides = [[3, 1], [3, 1], [], [1, 1], [1, 1], [1], [1], []]
    source = event(
        "aten::native_layer_norm_backward",
        shapes=shapes,
        strides=strides,
        types=[
            "float",
            "float",
            "ScalarList",
            "float",
            "float",
            "float",
            "float",
            "ScalarList",
        ],
        concrete=["", "", "[3]", "", "", "", "", "[true, true, true]"],
    )
    replay = EventReplayer(
        source,
        device="cpu",
        input_semantics="captured",
        input_values={
            "grad_out": grad,
            "input": x,
            "mean": mean,
            "rstd": rstd,
            "weight": weight,
            "bias": bias,
        },
    )
    result = replay.verify(expected)
    assert isinstance(result, tuple) and len(result) == 3


@pytest.mark.parametrize(
    "name,shapes,strides,types,concrete",
    [
        (
            "aten::copy_",
            [[2, 3], [2, 3], []],
            [[3, 1], [3, 1], []],
            ["float", "float", "Scalar"],
            ["", "", "false"],
        ),
        (
            "aten::dropout",
            [[2, 3], [], []],
            [[3, 1], [], []],
            ["float", "Scalar", "Scalar"],
            ["", "0.5", "true"],
        ),
        (
            "aten::add_",
            [[2, 3], [2, 3], []],
            [[3, 1], [3, 1], []],
            ["float", "float", "Scalar"],
            ["", "", "1"],
        ),
    ],
)
def test_mutation_and_rng_rejected_before_tensor_allocation(
    name, shapes, strides, types, concrete
):
    with patch(
        "TraceLens.EventReplay.event_replay.build_tensor",
        side_effect=AssertionError("must not allocate"),
    ):
        with pytest.raises(ReplayContractError, match="unsupported|Unsupported"):
            EventReplayer(event(name, shapes, strides, types, concrete), device="cpu")


def test_custom_collective_namespace_rejected():
    schema = torch._C.parse_schema("c10d::allreduce_(Tensor input) -> Tensor")
    from TraceLens.EventReplay.safety import operation_for_schema

    with pytest.raises(ReplayContractError, match="collectives"):
        operation_for_schema(schema)


def test_mutating_out_overload_of_supported_operator_is_rejected():
    source = event(
        shapes=[[2, 3], [3, 4], [2, 4]],
        strides=[[3, 1], [4, 1], [4, 1]],
        types=["float"] * 3,
        concrete=[""] * 3,
    )
    with pytest.raises(ReplayContractError, match="Mutation"):
        EventReplayer(source, device="cpu")


def test_legacy_repro_ir_routes_through_safe_schema_and_output_contract():
    from TraceLens.EventReplay.batched_replay import _event_from_legacy_ir

    source = EventReplayer(event(), device="cpu", lazy=True).get_repro_info()
    source.pop("event")
    source.pop("operator_schema")
    source.pop("input_semantics")
    replay = EventReplayer(_event_from_legacy_ir(source), device="cpu")
    assert replay.replay().shape == (2, 4)


@pytest.mark.parametrize(
    "change",
    [
        lambda e: e.update(args=None),
        lambda e: e["args"].pop("Input Strides"),
        lambda e: e["args"]["Input Strides"].pop(),
        lambda e: e["args"]["Input Strides"][0].__setitem__(0, -1),
        lambda e: e["args"]["Input Dims"][0].__setitem__(0, True),
        lambda e: e["args"]["Input Strides"][0].pop(),
    ],
)
def test_missing_and_malformed_metadata_rejected(change):
    source = event()
    change(source)
    with pytest.raises(ReplayContractError):
        EventReplayer(source, device="cpu")


def test_storage_bound_counts_stride_holes_before_allocating():
    source = event(strides=[[1000000, 1], [4, 1]])
    with pytest.raises(ReplayContractError, match="storage span"):
        EventReplayer(source, device="cpu", max_storage_elements=100)


def test_output_bound_prevents_small_inputs_from_allocating_enormous_outputs():
    source = event(shapes=[[100, 1], [1, 100]], strides=[[1, 1], [100, 1]])
    with patch(
        "TraceLens.EventReplay.event_replay.build_tensor",
        side_effect=AssertionError("must not allocate"),
    ):
        with pytest.raises(ReplayContractError, match="Output storage span"):
            EventReplayer(source, device="cpu", max_storage_elements=1000)


def test_captured_semantics_never_accept_synthetic_values_or_missing_names():
    with pytest.raises(ReplayContractError, match="actual tensor values"):
        EventReplayer(event(), device="cpu", input_semantics="captured")
    with pytest.raises(ReplayContractError, match="exactly"):
        EventReplayer(
            event(),
            device="cpu",
            input_semantics="captured",
            input_values={"self": torch.randn(2, 3)},
        )
    with pytest.raises(ReplayContractError, match="Synthetic"):
        EventReplayer(event(), device="cpu", input_values={})


def test_wrong_captured_dtype_and_aliasing_are_rejected():
    with pytest.raises(ReplayContractError, match="dtype"):
        EventReplayer(
            event(),
            device="cpu",
            input_semantics="captured",
            input_values={
                "self": torch.randn(2, 3, dtype=torch.float64),
                "mat2": torch.randn(3, 4),
            },
        )
    source = event(
        "aten::add",
        shapes=[[2, 3], [2, 3], []],
        strides=[[3, 1], [3, 1], []],
        types=["float", "float", "Scalar"],
        concrete=["", "", "1"],
    )
    x = torch.randn(2, 3)
    with pytest.raises(ReplayContractError, match="Aliased"):
        EventReplayer(
            source,
            device="cpu",
            input_semantics="captured",
            input_values={"self": x, "other": x.view_as(x)},
        )


def test_verification_failure_and_nonfinite_are_not_successes():
    replay = EventReplayer(
        event(),
        device="cpu",
        input_semantics="captured",
        input_values={"self": torch.ones(2, 3), "mat2": torch.ones(3, 4)},
    )
    with pytest.raises(ReplayContractError, match="numerical"):
        replay.verify(torch.zeros(2, 4))
    with pytest.raises(ReplayContractError, match="finite"):
        compare_outputs(torch.tensor([float("inf")]), torch.tensor([float("inf")]))
    with pytest.raises(ReplayContractError, match="captured inputs"):
        EventReplayer(event(), device="cpu").verify(torch.ones(2, 4))


@pytest.fixture
def replay_bundle(tmp_path):
    source = event(strides=[[1, 2], [4, 1]])
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": [source]}))
    x = torch.randn(3, 2).t()
    w = torch.randn(3, 4)
    path = tmp_path / "replay.json"
    provenance = {
        "torch_version": "2.11.0",
        "hip_version": "7.2.26365",
        "device_type": "hcu",
        "signature_source": "runtime_tensor_metadata",
        "values_source": "captured_runtime",
    }
    write_replay_bundle(
        trace,
        0,
        path,
        provenance=provenance,
        input_values={"self": x, "mat2": w},
        expected_output=x @ w,
    )
    return path


def test_roundtrip_bundle_executes_and_verifies(replay_bundle):
    report = run_bundle(replay_bundle, device="cpu", warmup=0, steps=2, max_seconds=5)
    assert (
        report["status"] == "complete" and report["numerical_verification"] == "passed"
    )
    assert (
        report["input_semantics"] == "captured"
        and report["training_state_equivalence"] == "not_evaluated"
    )
    assert report["timing"]["steps"] == 2 and len(report["timing"]["samples_us"]) == 2
    assert report["runtime"]["device"] == "cpu"


def test_bundle_cannot_overwrite_source_trace(replay_bundle):
    data = json.loads(replay_bundle.read_text())
    trace = replay_bundle.parent / data["trace_file"]
    original = trace.read_bytes()
    with pytest.raises(ReplayContractError, match="cannot be overwritten"):
        write_replay_bundle(trace, 0, trace, provenance=data["provenance"])
    assert trace.read_bytes() == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("trace_sha256", "0" * 64),
        ("values_sha256", "0" * 64),
        ("operator_schema", "aten::mm.invalid(Tensor self) -> Tensor"),
    ],
)
def test_bundle_rejects_changed_trace_values_and_schema(replay_bundle, field, value):
    data = json.loads(replay_bundle.read_text())
    data[field] = value
    replay_bundle.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        run_bundle(replay_bundle, device="cpu", warmup=0, steps=1)


def test_missing_value_provenance_rejected(replay_bundle):
    data = json.loads(replay_bundle.read_text())
    data["provenance"].pop("values_source")
    replay_bundle.write_text(json.dumps(data))
    with pytest.raises(ReplayContractError, match="captured_runtime"):
        run_bundle(replay_bundle, device="cpu", warmup=0, steps=1)


def test_hcu_requires_hip_and_never_falls_back(replay_bundle):
    with patch.object(torch.version, "hip", None):
        with pytest.raises(ReplayContractError, match="HIP"):
            run_bundle(replay_bundle, device="hcu", warmup=0, steps=1)


@pytest.mark.parametrize(
    "options",
    [
        {"steps": 0},
        {"steps": 1001},
        {"warmup": 101},
        {"max_seconds": 0},
        {"max_seconds": 301},
    ],
)
def test_benchmark_bounds_are_mandatory(options):
    replay = EventReplayer(event(), device="cpu")
    with pytest.raises(ReplayContractError):
        benchmark_replayer(replay, **options)


def test_benchmark_timeout_and_cli_numerical_rejection(replay_bundle, tmp_path):
    replay = EventReplayer(event(), device="cpu")
    with patch(
        "TraceLens.EventReplay.hcu_replay.time.perf_counter", side_effect=[0, 10]
    ):
        with pytest.raises(ReplayContractError, match="wall-time"):
            benchmark_replayer(replay, max_seconds=1)
    report_path = tmp_path / "report.json"
    with patch.object(torch.version, "hip", None):
        assert (
            main(
                [
                    "--bundle",
                    str(replay_bundle),
                    "--device",
                    "hcu",
                    "--report",
                    str(report_path),
                ]
            )
            == 1
        )
    assert json.loads(report_path.read_text())["status"] == "rejected"
