###############################################################################
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import List, Any
import time

_torch_module = None


def _get_torch_or_raise() -> Any:  # Changed return type to Any for flexibility
    """Lazily imports and returns the torch module."""
    global _torch_module
    if _torch_module is None:
        try:
            import torch

            _torch_module = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for EventReplayer functionality that is being used. "
                "Please install PyTorch."
            )
    return _torch_module


list_profile_tensor_types = [
    "double",
    "float",
    "c10::Half",
    "c10::BFloat16",
    "long",
    "int",
    "bool",
]

from dataclasses import dataclass


@dataclass
class TensorCfg:
    """
    A class to represent a dummy tensor.
    """

    shape: List[int]
    dtype: str
    strides: List[int]
    init: str = "normal"


def build_tensor(cfg: TensorCfg, device: str = "cuda") -> "torch.Tensor":

    torch = _get_torch_or_raise()
    dtype = profile_dtype(cfg.dtype)
    size = cfg.shape
    stride = cfg.strides
    # Fill backing storage, then construct the view. This also supports
    # expanded (zero-stride) read-only inputs without overlapping writes.
    span = 0 if 0 in size else 1 + sum((d - 1) * s for d, s in zip(size, stride))
    storage = torch.empty(span, dtype=dtype, device=device)
    t = storage.as_strided(size, stride)
    is_floating = t.is_floating_point() or t.is_complex()
    init = cfg.init
    if init == "normal":
        if not is_floating:
            raise ValueError(
                f"Cannot initialize tensor of type {cfg.dtype} with 'normal' init."
            )
        storage.normal_()
    elif init == "zeros":
        storage.zero_()
    elif init is not None:
        raise ValueError(f"Unsupported tensor initialization: {init}")
    return t


def profile_dtype(name):
    """Resolve only profiler dtypes explicitly supported by replay."""
    torch = _get_torch_or_raise()
    dict_profile2torchdtype = {
        "bool": torch.bool,
        "int": torch.int,
        "long": torch.long,
        "double": torch.float64,
        "float": torch.float32,
        "c10::Half": torch.float16,
        "c10::BFloat16": torch.bfloat16,
    }
    return dict_profile2torchdtype[name]


def summarize_tensor(tensor: "torch.Tensor") -> str:
    """
    Summarize the tensor information.

    Args:
        tensor (torch.Tensor): The tensor to summarize.

    Returns:
        str: The summary string.
    """
    return f"Tensor(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}, strides={tensor.stride()})"


def benchmark_func(func, device, warmup=50, avg_steps=100):
    """
    Benchmark a function with warmup and average steps.
    Disclaimer: This method would be inaccurate for very short ops.
    Args:
        func (callable): The function to benchmark.
        warmup (int): Number of warmup iterations.
        avg_steps (int): Number of iterations to average over.
    Returns:
        float: Average time taken per iteration in microseconds.
    """
    torch = _get_torch_or_raise()
    # Warmup phase
    for _ in range(warmup):
        func()

    # Benchmarking phase
    torch.cuda.synchronize(device)
    start_time = time.time()
    for _ in range(avg_steps):
        func()
    torch.cuda.synchronize(device)
    end_time = time.time()

    elapsed_time = end_time - start_time
    avg_time_sec = elapsed_time / avg_steps
    avg_time_us = avg_time_sec * 1e6

    return avg_time_us
