#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Patch installed cudaq_qec package with workspace decoder sources."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[patch] {src} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch installed cudaq_qec with local plugin sources.")
    parser.add_argument("--workspace",
                        type=str,
                        default=".",
                        help="Workspace root containing cudaqx/ directory.")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    src_root = workspace / "cudaqx/libs/qec/python/cudaq_qec/plugins/decoders"
    if not src_root.exists():
        raise FileNotFoundError(f"Cannot find source decoder path: {src_root}")

    # Find cudaq_qec install path WITHOUT fully importing it (avoids
    # triggering plugin auto-discovery which may fail on partial patches).
    spec = importlib.util.find_spec("cudaq_qec")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "cudaq_qec is not installed. Run: pip install cudaq-qec")
    installed_root = Path(spec.origin).resolve().parent
    dst_root = installed_root / "plugins/decoders"
    print(f"[patch] installed cudaq_qec root: {installed_root}")

    # Copy MPS decoder entry point.
    _copy_file(src_root / "tensor_network_mps_decoder.py",
               dst_root / "tensor_network_mps_decoder.py")

    # Copy ALL .py files in tensor_network_utils/ (mps_decoder_core, mps_node_np,
    # npsvd, tensor_network_factory, contractors, noise_models, __init__, etc.)
    tnu_src = src_root / "tensor_network_utils"
    tnu_dst = dst_root / "tensor_network_utils"
    for py_file in sorted(tnu_src.glob("*.py")):
        _copy_file(py_file, tnu_dst / py_file.name)

    print("[patch] done")


if __name__ == "__main__":
    main()
