###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Bounded, task-private HCU qualification workload. Not a model benchmark."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["torch", "hipprof"], default="torch")
    args = parser.parse_args()
    if not torch.version.hip:
        raise RuntimeError("This qualification probe requires a HIP PyTorch runtime")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(5317)
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    props = torch.cuda.get_device_properties(0)
    metadata = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": str(props),
        "device_name": props.name,
        "gfx": props.gcnArchName,
        "workload": "qualification-only-not-model-performance",
        "dtype": "float32",
        "shape": [512, 768, 256],
        "timestamp": time.time(),
    }
    x = torch.randn(512, 768, device="cuda", requires_grad=True)
    w = torch.randn(768, 256, device="cuda", requires_grad=True)

    def step():
        x.grad = None
        w.grad = None
        with torch.profiler.record_function("train_forward"):
            y = x @ w
            z = y.square().mean()
        with torch.profiler.record_function("train_backward"):
            z.backward()
        return z

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    if args.mode == "hipprof":
        host = torch.randn(128 * 1024, pin_memory=True)
        dev = torch.empty_like(host, device="cuda")
        for _ in range(4):
            dev.copy_(host, non_blocking=True)
            step()
            dev.copy_(dev + 1)
        torch.cuda.synchronize()
        metadata["finite_gradients"] = bool(torch.isfinite(w.grad).all())
    else:
        opts = dict(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )
        with torch.profiler.profile(**opts) as prof:
            for i in range(4):
                with torch.profiler.record_function("ProfilerStep#" + str(i)):
                    step()
                    torch.cuda.synchronize()
                prof.step()
        prof.export_chrome_trace(str(out / "eager.json"))
        ref_loss = step().detach().clone()
        ref_grad = w.grad.detach().clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        x.grad = None
        w.grad = None
        try:
            with torch.profiler.profile(**opts) as capture_prof:
                with torch.profiler.record_function(
                    "tracelens.train_graph.capture:qualification:trainstep"
                ):
                    with torch.cuda.graph(graph):
                        graph_loss = step()
                torch.cuda.synchronize()
            capture_prof.export_chrome_trace(str(out / "capture.json"))
            graph.replay()
            torch.cuda.synchronize()
            metadata["graph_loss_close"] = bool(
                torch.allclose(graph_loss, ref_loss, rtol=1e-5, atol=1e-6)
            )
            metadata["graph_grad_close"] = bool(
                torch.allclose(w.grad, ref_grad, rtol=1e-5, atol=1e-6)
            )
            with torch.profiler.profile(**opts) as replay_prof:
                for i in range(4):
                    with torch.profiler.record_function("ProfilerStep#" + str(i)):
                        with torch.profiler.record_function(
                            "tracelens.train_graph.replay:qualification:trainstep"
                        ):
                            graph.replay()
                        torch.cuda.synchronize()
                    replay_prof.step()
            replay_prof.export_chrome_trace(str(out / "replay.json"))
        except Exception as error:
            metadata["graph_error"] = type(error).__name__ + ": " + str(error)

        def timing(fn, repeats=20):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            samples = []
            for _ in range(repeats):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000)
            return {"median_us": statistics.median(samples), "samples_us": samples}

        # Calibrated observations for this device/runtime/shape, not theoretical peaks.
        calibration = {}
        for dtype in (torch.float32, torch.bfloat16):
            a = torch.randn(4096, 4096, device="cuda", dtype=dtype)
            b = torch.randn_like(a)
            c = torch.empty_like(a)
            row = timing(lambda: torch.mm(a, b, out=c))
            row.update(
                shape=[4096] * 3,
                dtype=str(dtype),
                operation="torch.mm",
                flops=2 * 4096**3,
                measured_tflops=(2 * 4096**3) / (row["median_us"] * 1e6),
            )
            calibration[str(dtype)] = row
            del a, b, c
        # 256 MiB per tensor > 8 MiB L2; report algorithmic read+write byte rate.
        a = torch.randn(64 * 1024 * 1024, device="cuda", dtype=torch.float32)
        b = torch.empty_like(a)
        row = timing(lambda: b.copy_(a))
        row.update(
            operation="device-copy",
            bytes=2 * a.numel() * a.element_size(),
            measured_gbps=2 * a.numel() * a.element_size() / (row["median_us"] * 1e3),
        )
        calibration["copy"] = row
        metadata["calibration"] = calibration
        torch.save(
            {
                "args": (x.detach().cpu(), w.detach().cpu()),
                "expected": (x.detach() @ w.detach()).cpu(),
            },
            out / "mm-values.pt",
        )
        metadata["input_signature"] = [
            dict(
                name=n, shape=list(t.shape), dtype=str(t.dtype), stride=list(t.stride())
            )
            for n, t in [("x", x), ("w", w)]
        ]
    (out / ("metadata-" + args.mode + ".json")).write_text(
        json.dumps(metadata, indent=2)
    )
    print(json.dumps(metadata, indent=2))
    if (
        metadata.get("graph_error")
        or metadata.get("graph_loss_close") is False
        or metadata.get("graph_grad_close") is False
        or metadata.get("finite_gradients") is False
    ):
        raise RuntimeError("Qualification failed; inspect retained metadata")


if __name__ == "__main__":
    main()
