###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Contracts for isolated, deterministic operator replay (including HIP/HCU).

A profiler records tensor metadata, not tensor values, aliasing or training
state. Replay is consequently an isolated operator experiment. This module
intentionally permits only a bounded set of functional ATen operators.
"""

import math
from .utils import TensorCfg, _get_torch_or_raise, list_profile_tensor_types


class ReplayContractError(ValueError):
    """The supplied event cannot be replayed under the safe contract."""


SUPPORTED_OPERATORS = frozenset(
    """mm bmm matmul addmm baddbmm add sub mul div true_divide neg abs
    pow exp log sqrt rsqrt reciprocal square sigmoid tanh relu silu gelu
    threshold softmax _softmax log_softmax _log_softmax
    sigmoid_backward tanh_backward threshold_backward silu_backward
    gelu_backward _softmax_backward_data _log_softmax_backward_data
    sum mean amax amin max min native_layer_norm native_layer_norm_backward
    layer_norm transpose t permute view reshape flatten squeeze unsqueeze
    expand contiguous clone detach alias select slice narrow
    """.split()
)


def operation_for_schema(schema):
    """Select one exact installed overload; never redispatch by operator name."""
    torch = _get_torch_or_raise()
    namespace, name = schema.name.split("::", 1)
    if namespace != "aten" or name not in SUPPORTED_OPERATORS:
        raise ReplayContractError(
            f"Unsupported safe replay operator: {schema.name}; mutation, collectives, "
            "custom operators and stateful/RNG operators have no supported contract"
        )
    if schema.is_mutable or any(
        a.alias_info is not None and a.alias_info.is_write for a in schema.arguments
    ):
        raise ReplayContractError(f"Mutation is unsupported for replay: {schema}")
    operation = getattr(
        getattr(torch.ops.aten, name), schema.overload_name or "default"
    )
    unsafe_tags = {"nondeterministic_seeded", "maybe_aliasing_or_mutating"}
    if any(str(tag).split(".")[-1] in unsafe_tags for tag in operation.tags):
        raise ReplayContractError(f"Stateful or RNG replay is unsupported: {schema}")
    return operation


def validate_event_metadata(event, max_storage_elements=16_777_216):
    """Reject incomplete metadata before allocating any tensors."""
    if (
        not isinstance(max_storage_elements, int)
        or isinstance(max_storage_elements, bool)
        or max_storage_elements < 1
    ):
        raise ReplayContractError("max_storage_elements must be a positive integer")
    if not isinstance(event, dict) or not isinstance(event.get("name"), str):
        raise ReplayContractError("event must contain an operator name")
    metadata = event.get("args", {})
    if not isinstance(metadata, dict):
        raise ReplayContractError("Event args must be a metadata object")
    keys = ("Input type", "Input Dims", "Input Strides", "Concrete Inputs")
    if any(not isinstance(metadata.get(k), (list, tuple)) for k in keys):
        raise ReplayContractError(
            "Replay needs explicit Input type, Input Dims, Input Strides and Concrete Inputs; layouts are never inferred"
        )
    count = len(metadata["Input type"])
    if any(len(metadata[k]) != count for k in keys):
        raise ReplayContractError("Input metadata lengths do not agree")
    if any(
        not isinstance(v, str)
        for v in list(metadata["Input type"]) + list(metadata["Concrete Inputs"])
    ):
        raise ReplayContractError(
            "Input type and Concrete Inputs entries must be explicit strings"
        )
    total_span = 0
    for dtype, shape, stride in zip(
        metadata["Input type"], metadata["Input Dims"], metadata["Input Strides"]
    ):
        if dtype not in list_profile_tensor_types:
            continue
        if (
            not isinstance(shape, (list, tuple))
            or not isinstance(stride, (list, tuple))
            or len(shape) != len(stride)
        ):
            raise ReplayContractError(
                "Each tensor needs a shape and a stride of the same rank"
            )
        if any(
            not isinstance(x, int) or isinstance(x, bool) or x < 0
            for x in list(shape) + list(stride)
        ):
            raise ReplayContractError(
                "Tensor shape and strides must be nonnegative integers"
            )
        span = 0 if 0 in shape else 1 + sum((d - 1) * s for d, s in zip(shape, stride))
        total_span += span
    if total_span > max_storage_elements:
        raise ReplayContractError(
            f"Tensor storage span {total_span} exceeds replay bound {max_storage_elements}"
        )


def validate_captured_inputs(ir, input_values, device):
    """Use caller-owned captured tensors without changing their dtype/layout.

    Same-device tensors are detached views. Cross-device transfers copy the
    minimal backing-storage span and reconstruct the exact strided view.
    """
    torch = _get_torch_or_raise()
    required = {
        entry["arg_name"]: entry["value"]
        for entry in ir["list_pos_args"] + ir["list_kwargs"]
        if isinstance(entry["value"], TensorCfg)
    }
    if not isinstance(input_values, dict) or set(input_values) != set(required):
        raise ReplayContractError(
            f"Captured inputs must contain exactly these tensor arguments: {sorted(required)}"
        )
    from .utils import profile_dtype

    result = {}
    target = torch.device(device)
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    storages = [
        (str(v.device), v.untyped_storage().data_ptr())
        for v in input_values.values()
        if isinstance(v, torch.Tensor) and v.numel()
    ]
    if len(storages) != len(set(storages)):
        raise ReplayContractError(
            "Aliased captured inputs have no supported replay contract"
        )
    for name, cfg in required.items():
        value = input_values[name]
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise ReplayContractError(
                f"Captured argument {name} must be a strided tensor"
            )
        if (
            list(value.shape) != list(cfg.shape)
            or list(value.stride()) != list(cfg.strides)
            or value.dtype != profile_dtype(cfg.dtype)
        ):
            raise ReplayContractError(
                f"Captured argument {name} does not match the profiled shape, dtype and strides"
            )
        value = value.detach()
        # Tensor.to can densify expanded or holey layouts. Transfer a 1-D
        # storage view so shapes/strides and all logical values survive.
        if value.device.type != target.type or (
            target.index is not None and value.device.index != target.index
        ):
            span = (
                0
                if 0 in cfg.shape
                else 1 + sum((d - 1) * s for d, s in zip(cfg.shape, cfg.strides))
            )
            backing = value.as_strided((span,), (1,), value.storage_offset()).to(target)
            value = backing.as_strided(cfg.shape, cfg.strides)
        result[name] = value
    # Independent serialized tensors cannot prove or preserve shared storage.
    return result


def validate_output_bound(operation, ir, max_storage_elements):
    """Use installed meta kernels to bound outputs before real allocation.

    Small inputs can otherwise broadcast to an enormous output. Operators
    without an installed shape-only implementation are explicitly unsupported.
    """
    torch = _get_torch_or_raise()
    from .utils import profile_dtype

    def value(entry):
        cfg = entry["value"]
        if isinstance(cfg, TensorCfg):
            return torch.empty_strided(
                cfg.shape, cfg.strides, dtype=profile_dtype(cfg.dtype), device="meta"
            )
        return cfg

    args = [value(e) for e in ir["list_pos_args"]]
    kwargs = {e["arg_name"]: value(e) for e in ir["list_kwargs"]}
    try:
        with torch.no_grad():
            output = operation(*args, **kwargs)
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        raise ReplayContractError(
            f"Installed meta kernel could not establish the output shape/dtype contract: {exc}"
        ) from exc

    def storage_span(result):
        if isinstance(result, torch.Tensor):
            return (
                0
                if 0 in result.shape
                else 1 + sum((d - 1) * s for d, s in zip(result.shape, result.stride()))
            )
        if isinstance(result, (tuple, list)):
            return sum(storage_span(v) for v in result)
        return 0

    span = storage_span(output)
    if span > max_storage_elements:
        raise ReplayContractError(
            f"Output storage span {span} exceeds replay bound {max_storage_elements}"
        )


def compare_outputs(actual, expected, rtol=1e-5, atol=1e-6):
    """Check output structure, shape, dtype and values; raise on mismatch."""
    torch = _get_torch_or_raise()
    if any(
        not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
        for x in (rtol, atol)
    ):
        raise ReplayContractError("rtol and atol must be finite nonnegative numbers")
    if isinstance(actual, torch.Tensor):
        if (
            not isinstance(expected, torch.Tensor)
            or actual.shape != expected.shape
            or actual.dtype != expected.dtype
        ):
            raise ReplayContractError(
                "Output tensor structure, shape or dtype differs from the reference"
            )
        if actual.is_floating_point() or actual.is_complex():
            if (
                not torch.isfinite(actual).all().item()
                or not torch.isfinite(expected).all().item()
            ):
                raise ReplayContractError(
                    "Output numerical verification requires finite actual and reference values"
                )
        try:
            torch.testing.assert_close(
                actual.detach().cpu(),
                expected.detach().cpu(),
                rtol=rtol,
                atol=atol,
                equal_nan=False,
                check_stride=False,
            )
        except AssertionError as exc:
            raise ReplayContractError(
                f"Output numerical verification failed: {exc}"
            ) from exc
        return
    if isinstance(actual, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ReplayContractError(
                "Output sequence structure differs from the reference"
            )
        for a, e in zip(actual, expected):
            compare_outputs(a, e, rtol, atol)
        return
    if type(actual) is not type(expected) or actual != expected:
        raise ReplayContractError(
            "Output scalar or optional value differs from the reference"
        )
