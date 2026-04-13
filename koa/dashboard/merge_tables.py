# -*- coding: utf-8 -*-
"""Merge JSN / osteophyte / sclerosis batch CSVs on ``case_id``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def merge_koa_result_csvs(
    jsn_csv: Path,
    ost_csv: Path,
    scl_csv: Path,
    out_csv: Path,
    *,
    how: str = "outer",
) -> pd.DataFrame:
    """
    Left-merge style outer join on ``case_id``. Column names are prefixed with
    ``jsn_``, ``ost_``, ``scl_`` (except ``case_id``) to avoid collisions.
    """
    j = pd.read_csv(jsn_csv)
    o = pd.read_csv(ost_csv)
    s = pd.read_csv(scl_csv)

    def _prefix(df: pd.DataFrame, pfx: str) -> pd.DataFrame:
        out = df.copy()
        out.columns = [
            pfx + c if c != "case_id" else "case_id" for c in out.columns
        ]
        return out

    j = _prefix(j, "jsn_")
    o = _prefix(o, "ost_")
    s = _prefix(s, "scl_")

    merged = j.merge(o, on="case_id", how=how).merge(s, on="case_id", how=how)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding="utf-8")
    return merged


def load_merged_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    return pd.read_csv(path)
