###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Loader for per-platform GPU architecture JSON specs.

Each JSON file in utils/arch/ follows the TraceLens arch dict schema
(see examples/gpu_arch_example.md) with an additional ``memory_gb`` field.

Resolution order (matches skills/analysis-orchestrator/reference.md Step 1):
  1. Check ``Agent/Analysis/utils/arch/`` inside each ``$TL_EXTENSION`` package
  2. Fall back to the bundled ``utils/arch/`` directory next to this file
"""

import json
import os

_ARCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arch")


def _collect_arch_jsons():
    """Build a dict mapping platform name to its JSON file path.

    Bundled ``utils/arch/`` entries are added first, then augmented (and, for
    collisions, overridden) by matches in ``Agent/Analysis/utils/arch/``
    inside each ``$TL_EXTENSION`` package (colon-separated import names).
    """
    result = {
        f.removesuffix(".json"): os.path.join(_ARCH_DIR, f)
        for f in os.listdir(_ARCH_DIR)
        if f.endswith(".json")
    }
    tl_extension = os.environ.get("TL_EXTENSION", "").strip()
    if tl_extension:
        for pkg_name in tl_extension.split(":"):
            try:
                pkg = __import__(pkg_name)
                pkg_root = os.path.dirname(pkg.__file__)
            except ImportError:
                continue
            ext_arch = os.path.join(pkg_root, "Agent", "Analysis", "utils", "arch")
            if not os.path.isdir(ext_arch):
                continue
            for f in os.listdir(ext_arch):
                if f.endswith(".json"):
                    result[f.removesuffix(".json")] = os.path.join(ext_arch, f)
    return result


def list_platforms():
    """Return available platform names from TL_EXTENSION (if set) and bundled arch dir."""
    return sorted(_collect_arch_jsons())


def load_arch(platform: str, gpu_arch_json_path=None) -> dict:
    """Load a bundled/extension platform or an explicit site architecture file.

    An explicit HCU file is qualified before native categories receive its
    measured envelope. Missing calibration stays unknown; another platform's
    values cannot substitute for the selected device.
    """
    if gpu_arch_json_path is None:
        path = _collect_arch_jsons()[platform]
    else:
        path = os.fspath(gpu_arch_json_path)
    with open(path, encoding="utf-8") as f:
        arch = json.load(f)
    from TraceLens.PerfModel.hcu_roofline import is_hcu_arch, qualify_hcu_arch

    if is_hcu_arch(arch):
        return qualify_hcu_arch(arch, expected_name=platform)
    if gpu_arch_json_path is not None and arch.get("name") != platform:
        raise ValueError("Architecture name does not match the selected platform")
    return arch
