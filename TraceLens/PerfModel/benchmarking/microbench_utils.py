###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared utilities for the microbenchmark scripts.

- resolve_physical_device: logical torch index -> physical id used by smi tools.
- check_gpu_idle: pre-flight idle check via hy-smi, amdsmi or nvidia-smi.
"""

from __future__ import annotations

import os
import json
import math
import shutil
import subprocess
from typing import Any, Tuple


def resolve_physical_device(logical: int) -> Tuple[int, str]:
    """Map a logical torch device index to the physical id used by amdsmi /
    nvidia-smi via HIP/ROCR/CUDA_VISIBLE_DEVICES. Returns (phys, source)."""
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        val = os.environ.get(var, "").strip()
        if not val:
            continue
        parts = [p.strip() for p in val.split(",") if p.strip()]
        try:
            mapped = [int(p) for p in parts]
        except ValueError:
            return logical, f"{var}={val} (non-numeric, identity fallback)"
        if 0 <= logical < len(mapped):
            return mapped[logical], var
        return (
            logical,
            f"{var}={val} (logical {logical} out of range, identity fallback)",
        )
    return logical, "identity"


def _int_metric(value: Any, *, default: int = 0) -> int:
    """Coerce an amdsmi metric to int; treat ``N/A`` and non-numeric as *default*."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)


def _check_amd_gpu_idle(
    phys: int,
    dev_tag: str,
    *,
    util_threshold: int,
    mem_threshold_pct: float,
) -> Tuple[bool, str]:
    import amdsmi

    amdsmi.amdsmi_init()
    try:
        devices = amdsmi.amdsmi_get_processor_handles()
        if phys < 0 or phys >= len(devices):
            return (
                True,
                f"amdsmi {dev_tag}: physical id {phys} out of range "
                f"(found {len(devices)} GPU(s)); check skipped",
            )

        device = devices[phys]
        activity = amdsmi.amdsmi_get_gpu_activity(device)
        util = _int_metric(activity.get("gfx_activity"))

        vram = amdsmi.amdsmi_get_gpu_vram_usage(device)
        mem_used_mib = _int_metric(vram.get("vram_used"))
        mem_total_mib = _int_metric(vram.get("vram_total"))
        mem_pct = 100.0 * mem_used_mib / mem_total_mib if mem_total_mib > 0 else None

        mem_busy = mem_pct is not None and mem_pct > mem_threshold_pct
        mem_str = (
            f"{mem_pct:.1f}% of {mem_total_mib}MiB"
            if mem_pct is not None
            else "total unknown"
        )
        if util > util_threshold or mem_busy:
            return False, (
                f"amdsmi {dev_tag}: util={util}% mem={mem_used_mib}MiB " f"({mem_str})"
            )
        return True, (
            f"amdsmi {dev_tag}: idle (util={util}% mem={mem_used_mib}MiB {mem_str})"
        )
    finally:
        amdsmi.amdsmi_shut_down()


def check_gpu_idle(
    device: int, *, util_threshold: int = 5, mem_threshold_pct: float = 10.0
) -> Tuple[bool, str]:
    """Pre-flight utilization check. Returns (is_idle, msg), not a reservation.

    HCU hy-smi rejects ambiguous mappings and missing metrics. The legacy AMD/NV
    fallback may skip unavailable tools; callers still need resource admission.
    """
    phys, src = resolve_physical_device(device)
    dev_tag = f"logical {device} -> physical {phys} (via {src})"

    hcu_smi = shutil.which("hy-smi")
    if hcu_smi:
        # An ambiguous visibility mapping or unavailable metric cannot grant
        # admission on a shared HCU device. The caller may explicitly override.
        if "fallback" in src or os.environ.get("ROCR_VISIBLE_DEVICES"):
            return (
                False,
                f"hy-smi: unresolved device mapping ({src}); verify physical device admission",
            )
        try:
            result = subprocess.run(
                [
                    hcu_smi,
                    "--showuse",
                    "--showmeminfo",
                    "vram",
                    "--json",
                    "-d",
                    str(phys),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode:
                return (
                    False,
                    f"hy-smi {dev_tag}: query failed ({result.returncode}); idle state unknown",
                )
            row = json.loads(result.stdout)[f"card{phys}"]
            util = float(row["HCU use (%)"])
            used = float(row["vram Total Used Memory (MiB)"])
            total = float(row["vram Total Memory (MiB)"])
            if (
                not all(math.isfinite(v) for v in (util, used, total))
                or not 0 <= util <= 100
                or not 0 <= used <= total
                or total <= 0
            ):
                raise ValueError("Invalid utilization/memory metrics")
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError):
            return (
                False,
                f"hy-smi {dev_tag}: missing or invalid metrics; idle state unknown",
            )
        pct = 100.0 * used / total
        idle = util <= util_threshold and pct <= mem_threshold_pct
        return (
            idle,
            f"hy-smi {dev_tag}: {'idle' if idle else 'busy'} (util={util}% mem={pct:.1f}%)",
        )

    try:
        import amdsmi  # noqa: F401
    except ImportError:
        amdsmi = None

    if amdsmi is not None:
        try:
            return _check_amd_gpu_idle(
                phys,
                dev_tag,
                util_threshold=util_threshold,
                mem_threshold_pct=mem_threshold_pct,
            )
        except Exception as e:
            return True, f"amdsmi check skipped ({e})"

    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(phys),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            parts = [p.strip() for p in (r.stdout or "").split(",")]
            util = int(parts[0]) if parts and parts[0].isdigit() else 0
            mem_used_mib = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            mem_total_mib = (
                int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            )
            mem_pct = (
                100.0 * mem_used_mib / mem_total_mib if mem_total_mib > 0 else None
            )
            mem_busy = mem_pct is not None and mem_pct > mem_threshold_pct
            mem_str = (
                f"{mem_pct:.1f}% of {mem_total_mib}MiB"
                if mem_pct is not None
                else "total unknown"
            )
            if util > util_threshold or mem_busy:
                return False, (
                    f"nvidia-smi {dev_tag}: util={util}% mem={mem_used_mib}MiB "
                    f"({mem_str})"
                )
            return True, (
                f"nvidia-smi {dev_tag}: idle (util={util}% mem={mem_used_mib}MiB {mem_str})"
            )
        except Exception as e:
            return True, f"nvidia-smi check skipped ({e})"

    return True, "no amdsmi/nvidia-smi available; skipping idle check"
