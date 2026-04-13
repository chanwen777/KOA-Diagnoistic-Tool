# -*- coding: utf-8 -*-
"""Resolve integer label IDs from ``label_mapping`` keys (scalar or list values)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def label_ids_from_mapping_keys(
    label_mapping: Dict[str, Any],
    keys: Sequence[str],
) -> List[int]:
    """
    For each key, append its value: a single int or every int in a list/tuple.
    Preserves order, drops duplicates and 0.
    """
    out: List[int] = []
    for k in keys:
        if k not in label_mapping:
            raise KeyError(f"label_mapping has no key {k!r}")
        v = label_mapping[k]
        if isinstance(v, (list, tuple)):
            out.extend(int(x) for x in v)
        else:
            out.append(int(v))
    seen: set[int] = set()
    res: List[int] = []
    for x in out:
        if x != 0 and x not in seen:
            seen.add(x)
            res.append(x)
    return res
