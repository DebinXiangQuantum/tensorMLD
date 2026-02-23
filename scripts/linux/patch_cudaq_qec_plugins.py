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
import shutil
from pathlib import Path

import cudaq_qec


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

    installed_root = Path(cudaq_qec.__file__).resolve().parent
    dst_root = installed_root / "plugins/decoders"
    print(f"[patch] installed cudaq_qec root: {installed_root}")

    targets = [
        ("tensor_network_mps_decoder.py", "tensor_network_mps_decoder.py"),
        ("tensor_network_utils/mps_decoder_core.py",
         "tensor_network_utils/mps_decoder_core.py"),
    ]

    for rel_src, rel_dst in targets:
        src = src_root / rel_src
        dst = dst_root / rel_dst
        if not src.exists():
            raise FileNotFoundError(src)
        _copy_file(src, dst)

    print("[patch] done")


if __name__ == "__main__":
    main()
