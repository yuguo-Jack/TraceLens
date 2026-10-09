<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
See LICENSE for license information.
-->

# HCU integration and fork maintenance

This public fork preserves the upstream TraceLens modules, APIs, command-line
tools, tests and license. Its `hcu` branch carries focused integration changes;
`main` tracks the upstream baseline. The upstream project remains
[AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens).

[HCU-TrainFlow](https://github.com/yuguo-Jack/HCU-TrainFlow) pins a reviewed commit
from this fork. Its two report wrappers are convenient training entrypoints;
they do not replace or restrict the native TraceLens functionality.

## Installation and preserved capabilities

Use a supported local Python environment and follow the upstream installation
guide. `python -m pip install -e .` installs the base analysis dependencies. The
fork currently preserves the upstream dependency declarations; optional packages
remain governed by `setup.py`. Selecting all extras is unnecessary for ordinary
PyTorch trace reporting.

| Capability | Reusable upstream entrypoint or module | Additional requirements and HCU checks |
| --- | --- | --- |
| Training op/kernel reports and overlap | `TraceLens_generate_perf_report_pytorch`, `Trace2Tree`, `TreePerf` | Compatible exported trace and CPU/device correlation; no local training runtime required for ordinary offline reports. |
| Graph/inference reports | `TraceLens_generate_perf_report_pytorch_inference` | Capture metadata must support graph replay attribution; report unresolved attribution explicitly. |
| Collective latency, bandwidth and skew | `TraceLens_generate_multi_rank_collective_report_pytorch`, `NcclAnalyser` | Complete rank set where required, process groups and message metadata; RCCL naming support alone does not prove HCU topology correctness. |
| Before/after comparisons | `TraceLens_compare_perf_reports_pytorch`, `TraceDiff` | Comparable shapes, precision, steady-state windows and timing denominators. |
| Roofline and performance models | `PerfModel`, operator mappings and extensions | Verified target architecture and dtype-specific peaks/traffic; optional Origami simulation needs its package and a validated hardware model. |
| Source lookup | `TraceLens_resolve_kernel_source` | Actual source roots/build artifacts; optional demangling package via the `kernel_source` extra. |
| Splitting and corpus search | `TraceLens_split_trace`, `TraceLens_trace_index` | Private input/output directories; indexing does not establish semantic correctness. |
| Replay and targeted experiments | `EventReplay` | Replay execution needs a compatible training/runtime environment and valid captured inputs. |
| rocprofv3, pftrace HIP/HSA, memory-copy and Genesis reports | Native `TraceLens_generate_perf_report_*` tools | Matching input schema; these parsers are not automatically hipprof or XProf/XCompute adapters. |
| JAX/XPlane | `TraceLens_generate_perf_report_jax` | Use the upstream `jax` extra when needed; validate the target export format. |
| Agent analysis and comparative tooling | Upstream Agent directories and optional `comparative` extra | Configure the selected agent/provider separately. Offline report generation requires no cloud model credentials. |

All console entrypoints are retained in `setup.py`. Run the command's `--help`
and consult [the module documentation](../README.md#documentation) for its actual
arguments. A successful help/import check establishes installation reachability,
not correctness on a real HCU workload.

## Prefer extension over replacement

1. Reuse `DataLoader`, the existing event tree, timeline analysis, collective
   classifiers and report writers. Avoid another parser or another set of
   conflicting accounting rules.
2. Supply verified hardware JSON through the existing report API. Upstream
   architecture discovery also supports `TL_EXTENSION`; inspect its current
   contract before adding an extension. Never substitute an AMD/NVIDIA peak for
   an unknown HCU peak.
3. Extend existing op mappings/performance-model registrations for observed HCU
   TE, Flash-Train, GEMM, Triton and communication implementations. Record shape,
   layout, forward/backward, precision and traffic assumptions alongside evidence.
4. Change core parsing only when a minimal reproducer shows that the supported
   extension points cannot represent the actual trace. Keep the original input
   available and mark unknown events, missing ranks and unmodeled time.
5. Keep deployment, credentials, capture commands and training decisions in the
   consuming workflow or private task configuration.

## Patch ledger and validation boundary

Baseline: `c5da90ee7d59e49bf75e8602f0211da1f6dcd8b3` from upstream `main`.

| Patch | Evidence and scope | Validation |
| --- | --- | --- |
| Portable CSV field bound | Native report CLI imports failed on a Windows Python build because `sys.maxsize` exceeded the C long accepted by `csv.field_size_limit`. Keep the existing large bound where supported and handle only `OverflowError`. | Native CLI help smoke tests plus large-field parsing under simulated platform limits; Linux/Windows CI. |
| Fork CI isolation | Preserve the upstream private-repository sync workflow, but run its notification job only in the upstream repository. | Repository-identity condition; the HCU branch has its own CPU portability checks. |
| HIP runtime categories | Recognize `hip_runtime` and `hip_driver` wherever host trees, launch correlation and GPU attachment recognize CUDA runtime events. Preserve original event labels, and retain CUDA-named events emitted by HIP Kineto builds. | Authored fixtures cover all four categories, HIP Graph one-to-many launches and rejection of ambiguous non-graph links; existing kernel-launcher, overlap and graph tests continue to pass. Real site traces remain a separate validation step. |

The native `gpu_timeline` denominator spans the first GPU event start through the
last GPU event end. It excludes CPU-only time before and after that span; it is
not automatically the end-to-end training-step denominator. Use a retained,
explicit step window for end-to-end hotspot coverage, and do not infer CPU or
communication critical paths solely from overlapping intervals. Collective
analysis may use inference stream-order matching when process-group metadata is
absent; that fallback must not be treated as validated training rank-group
evidence. Keep missing shapes, allocator events, backward annotations and rank
clock alignment explicit in the consuming workflow.

No HCU peak table, undocumented kernel rule or new hardware prediction is supplied
by this initial fork. TrainFlow has run synthetic single-rank, compressed-trace
and two-rank collective reports locally. Real HCU DTK/PyTorch versions, graph
traces, custom collectives, predictions and training loss remain to be validated.

For each real environment, retain the software/source versions, capture settings,
known op-to-kernel workload, parallel groups, clock alignment, steady-state
window and independent measurements. Check represented and missing time before
trusting conclusions. Public regressions must use authored synthetic inputs or
explicitly publishable sanitized data; real training traces stay private.

## Updating the fork

Maintain a separate development checkout, with `origin` pointing to this fork and
`upstream` to AMD-AGI/TraceLens. Keep TrainFlow's dependency checkout clean.

1. Fetch upstream and inspect changes to parsing, dependencies, models, extension
   contracts, reporting and tests against the recorded baseline.
2. Merge the reviewed upstream commit into a topic branch based on `hcu`; resolve
   changes at their owning layer. Do not discard modules or commands to make an
   integration pass.
3. Run the relevant upstream regressions, the portability suite and available HCU
   fixtures. Missing extras or unavailable hardware must remain visible. Never
   regenerate reference results merely to suppress a regression.
4. Record the new upstream baseline and patch status here. Retire a local patch
   once an equivalent upstream fix is included and validated. Propose general
   fixes upstream through its contribution process when authorized.
5. Publish the reviewed fork commit, then update TrainFlow's dependency lock,
   integration guidance and corresponding regression evidence. Running tasks
   retain their original source/environment snapshots.

Rollback uses a previously validated fork commit and its dependency environment;
there is no force-reset of a user's modified checkout.
