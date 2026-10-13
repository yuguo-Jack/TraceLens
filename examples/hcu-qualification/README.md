<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Bounded HCU tool qualification

These small workloads exercise tool paths, not model training performance.
Reserve an idle device, use the task's isolated directory and an external process
deadline. Never run against another user's active job. The probe synchronizes
each profiled step deliberately; its idle ratio is not a production bottleneck.

```bash
HIP_VISIBLE_DEVICES=0 timeout 120s python torch_probe.py --output /private/evidence
hipcc -O2 --offload-arch=<actual-gfx> counter_probe.hip -o /private/counter_probe
```

The Torch probe writes eager/capture/replay traces, actual matrix input/output
values, loss/gradient checks and same-device reference calibrations. It requires
HIP PyTorch, not a CPU-only Torch wheel. The measured large-matrix/copy rates are
conditional empirical references, not vendor peaks or small-shape attainable
performance. Read metadata and exit status; a file existing does not mean pass.

Use the matching DTK hipprof, XProf and XCompute version's help to collect the
counter probe and export its native CSV. A qualified combination was XProf
3.16.0/aqlprofile3.6.2, XComputeLight4.8.0 and gfx936/DTK26.10; other combinations
need admission and validation. Counter and SQTT collection can replay kernels.
See [counter input](../../docs/how-to/hcu-xprof-xcompute.md) and
[Graph/replay contracts](../../docs/how-to/hcu-graph-replay.md).

Keep outputs private. Bind manifests to actual files, signatures and hashes;
do not substitute checked-in sample data for hardware execution. Reusable reader
tests use authored fixtures and run without these captures.

`analyze_probe.py --probe-dir <probe> --output <new-directory>` consumes the
Torch probe and exercises the native CPU analysis tools. Add `--execute-replay`
only with an admitted HCU device; it loads captured values with `weights_only`.
Its native Agent stage produces metrics, not fabricated findings. Successful
commands are recorded in `execution.json`; inspect the actual tables as well.

`collective_probe.py` runs under `torchrun --standalone --nproc-per-node=2` on
two admitted devices. It checks the all-reduce result and records per-rank traces
with actual process-group metadata. The native collective report must contain
both ranks and four matched collectives. Do not use this intentionally bounded,
synchronized probe to establish a cluster's peak bandwidth.
