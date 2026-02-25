from __future__ import annotations

import os
from typing import Any, Optional

from experiments.codes.isca_revision_codes import (
    ISCA_REVISION_CODE_ORDER,
    load_isca_revision_codes,
)


def load_isca_code_cases(
    *,
    code_names: Optional[list[str]] = None,
    skip_missing: bool = False,
) -> tuple[list[tuple[str, Any]], list[str]]:
    loaded, skipped = load_isca_revision_codes(skip_missing=skip_missing)
    loaded_map = {name: code for name, code in loaded}

    if code_names is None:
        ordered_names = list(ISCA_REVISION_CODE_ORDER)
    else:
        ordered_names = [str(name).upper() for name in code_names]
        unknown = sorted(set(ordered_names) - set(ISCA_REVISION_CODE_ORDER))
        if unknown:
            raise ValueError(
                "Unknown code names: "
                f"{unknown}. Supported: {list(ISCA_REVISION_CODE_ORDER)}")

    selected: list[tuple[str, Any]] = []
    missing_selected: list[str] = []
    for name in ordered_names:
        code = loaded_map.get(name)
        if code is None:
            missing_selected.append(name)
            continue
        selected.append((name, code))

    if missing_selected and not skip_missing:
        raise RuntimeError(
            "Failed to load required ISCA codes: "
            f"{missing_selected}. Install dependencies/files or pass skip_missing=True."
        )
    return selected, skipped


def resolve_parallel_workers(
    *,
    use_gpu: bool,
    parallel_workers: Optional[int],
) -> int:
    if parallel_workers is not None:
        return max(1, int(parallel_workers))
    if use_gpu:
        return 1
    return max(1, min(4, int(os.cpu_count() or 1)))
