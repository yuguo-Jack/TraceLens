###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Small two-rank RCCL trace: torchrun --standalone --nproc-per-node=2."""

import argparse
from datetime import timedelta
import os
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not torch.version.hip:
        raise RuntimeError("Requires a HIP PyTorch build and two admitted HCU devices")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        value = torch.ones(256 * 1024, device="cuda")
        for _ in range(3):
            value.fill_(1)
            dist.all_reduce(value)
        torch.cuda.synchronize()
        dist.barrier()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as profiler:
            for step in range(4):
                with torch.profiler.record_function(f"ProfilerStep#{step}"):
                    value.fill_(1)
                    dist.all_reduce(value)
                    torch.cuda.synchronize()
                profiler.step()
        if not bool(torch.all(value == dist.get_world_size())):
            raise RuntimeError("Collective numerical check failed")
        profiler.export_chrome_trace(str(args.output / f"rank{dist.get_rank()}.json"))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
