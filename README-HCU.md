<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceLens for HCU

This fork adds HCU compatibility to upstream TraceLens. It keeps the native
reporting tools, Python APIs and Agent workflow. HCU support includes HIP launch
correlation, training Graph attribution, operator replay, qualified Roofline
estimates, DTK hipprof traces and XProf/XCompute output.

Use the `hcu` branch. The merged upstream baseline is
[`493f750`](https://github.com/AMD-AGI/TraceLens/commit/493f7508505f36e6ac850aaa29a252bef51e23fc).
See the [maintenance and patch ledger](docs/hcu-integration.md) for the fork
contract and [upstream README](README.md) for all modules and native commands.

## Install

```bash
git clone --branch hcu https://github.com/yuguo-Jack/TraceLens.git
cd TraceLens
python -m pip install -e '.[kernel_source]'
```

Offline JSON/CSV analysis needs no HCU device. Collecting traces, replaying
operators and benchmarking require the site's compatible DTK, driver and HIP
PyTorch. Use that environment's existing Torch/Triton packages; do not replace
them with CUDA wheels. Source the site's DTK activation script before execution.
The `dev`, `jax` and `comparative` extras are optional. JAX's PyPI `xprof` is
**not** the HCU XProf profiler; install the latter from the approved HCU toolkit.

## Choose an analysis

| Capability | HCU path | Scope and requirements |
| --- | --- | --- |
| Training reports, overlap, host-to-device tree | `TraceLens_generate_perf_report_pytorch` | HIP Kineto may emit CUDA-named or HIP-named runtime categories; both are recognized. Retain unknown/unlinked work. |
| Inference and Graph reports | Native inference reporting; training uses `TraceLens.Trace2Tree.training_graph` | Inference capture metadata and explicit training capture/replay manifests are different contracts. Training currently verifies serialized single-stream attribution. |
| Multi-rank collectives | `TraceLens_generate_multi_rank_collective_report_pytorch` | RCCL operations use the native NCCL-named analyzer. Supply full process-group/rank metadata; verify topology and clocks separately. |
| Before/after comparison | `TraceLens_compare_perf_reports_pytorch`, `TraceDiff` | Report-table comparison and tree alignment are both available. Keep shapes, phases, precision and timing windows comparable. |
| Multi-rank visualization | `TraceFusion.TraceFuse` | HIP correlation and rank ID namespaces are preserved. Merging does not synchronize node clocks or establish a causal critical path. |
| Roofline and efficiency | Native PerfModel plus `TraceLens.PerfModel.hcu_roofline` | Supply actual HCU identity, precision-specific references and operation work. Practical FLOPs/bytes bounds are sufficient; unmodeled work stays unknown. |
| Hardware calibration | `TraceLens.PerfModel.benchmarking.microbench` | HCU names are not inferred as MI300/MI350 from VRAM size. `hy-smi` checks idle state; uncertain mapping is not idle. Unsupported precision is not a usable ceiling. |
| Isolated operator replay | `TraceLens.EventReplay.hcu_replay` | Captured values can verify outputs. Synthetic inputs only benchmark shape/layout. Stateful, RNG, aliased and collective semantics need dedicated harnesses. |
| Agent analysis and fusion opportunities | Native preparation/category scripts, `hcu_analysis`, Agent instructions | Scripts produce evidence; a configured/active Agent writes and reviews conclusions. No LLM is silently started by offline analysis. |
| Source and library lookup | `TraceLens_resolve_kernel_source` | HIP/CUDA and C++ device definitions, durable Triton source, DTK rocBLAS/hipBLASLt artifacts. A library artifact is not editable source; generated kernels require their generating code. |
| Trace slicing and search | `TraceLens_split_trace`, `TraceLens_trace_index` | Explicit training-step ranges; SQLite corpus search. Inference steady-state heuristics are not training-step selection rules. |
| HCU API/kernel/copy timelines | `TraceLens.Reporting.hipprof` | DTK JSON and pftrace via an explicit Perfetto `traceconv`. Preserve process-scoped correlations and original timestamps. |
| HCU counters and instruction samples | `TraceLens.Reporting.xcompute` | XProf `.perf` → matching XCompute CSV → manifest-bound tables. Counter replay and sampled SQTT time are not profiler-off latency. |
| rocprofv3, native pftrace, Genesis, JAX/XPlane | Preserved upstream parsers and tests | Matching input formats still work. Preserving a parser is not evidence that the corresponding collector/runtime runs on every HCU stack. |
| Origami simulation / external comparative LLMs | Preserved optional integrations | Require separately supported packages, hardware models or provider configuration. No validated generic HCU Origami model is bundled. |

```bash
TraceLens_generate_perf_report_pytorch \
  --profile_json_path rank0.json --output_csvs_dir results/rank0 \
  --include_unlinked_kernels --enable_kernel_summary --include_overlap_info

TraceLens_compare_perf_reports_pytorch results/before results/after \
  --names before after --output results/comparison.xlsx

TraceLens_split_trace rank0.json --output-dir results/steps --iterations 10:20
TraceLens_trace_index --db results/traces.sqlite append \
  --trace-path rank0.json --report-dir results/rank0
TraceLens_trace_index --db results/traces.sqlite search aten::mm
```

For HCU efficiency, supply `--gpu_arch_json_path` using the
[qualified architecture contract](docs/how-to/hcu-roofline-agent.md). An identity-only
file is valid but leaves headroom unknown. Observed kernel sums, the GPU activity
span and the complete training-step wall time are different denominators.
Measured bandwidth/GEMM envelopes, theoretical peaks and same-shape baselines
must remain distinguishable. A model estimate prioritizes experiments; it does
not prove end-to-end speedup or eliminate a high-impact unknown operation.

## Reproduce qualification

After confirming the selected devices are idle and reserved for this task:

```bash
HIP_VISIBLE_DEVICES=0 python examples/hcu-qualification/torch_probe.py --output work/probe
HIP_VISIBLE_DEVICES=0 python examples/hcu-qualification/analyze_probe.py \
  --probe-dir work/probe --output work/analysis --execute-replay
HIP_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  examples/hcu-qualification/collective_probe.py --output work/collective
TraceLens_generate_multi_rank_collective_report_pytorch \
  --trace_pattern 'work/collective/rank*.json' --world_size 2 \
  --gpus_per_node 2 --output_csvs_dir work/collective-report
```

Use new output directories. Keep external deadlines and resource admission for
GPU commands. The bounded example exercises reporting, modeling, Graph,
comparison, TraceDiff, slicing, indexing, source lookup, Agent metrics and optional
replay. It checks execution and selected invariants; inspect the tables and raw
evidence as well. Self-comparison validates comparison behavior, not an optimization.
Agent findings and the final diagnosis still need actual reasoning and review.

Qualification was exercised on BW1000_H/gfx936, DTK 26.10 and HIP PyTorch 2.11.
XProf 3.16.0 / aqlprofile 3.6.2 / XCompute Light 4.8.0 form the counter/SQTT
qualification stack. This scope does not claim all HCU generations, JAX training,
complex multistream training Graphs, all operator types, GUI interaction or
production loss convergence. See [qualification details](docs/hcu-integration.md).

## More detailed guides

- [Graph and captured-value replay](docs/how-to/hcu-graph-replay.md)
- [Hardware references, Roofline and Agent analysis](docs/how-to/hcu-roofline-agent.md)
- [XProf/XCompute counter and SQTT evidence](docs/how-to/hcu-xprof-xcompute.md)
- [Fork maintenance, provenance and validation boundaries](docs/hcu-integration.md)

Analysis traces, tensors, private paths and credentials belong in the caller's
workspace. This public repository contains reusable code and authored tests.
