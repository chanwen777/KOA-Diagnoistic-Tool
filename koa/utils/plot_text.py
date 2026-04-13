# -*- coding: utf-8 -*-
"""Plot-facing text: keep annotations Latin-only when case IDs may contain CJK."""

from __future__ import annotations

import re
from typing import Optional

# Broad CJK / fullwidth blocks; strip from strings shown on matplotlib figures.
_CJK_RE = re.compile(
    r"[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff\u3400-\u4dbf"
    r"\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]+"
)


def sanitize_plot_text(s: Optional[str], *, fallback: str = "") -> str:
    """
    Remove CJK / fullwidth segments; collapse whitespace.
    If nothing remains, return ``fallback``.
    """
    if s is None:
        return fallback
    t = _CJK_RE.sub("", str(s)).strip()
    if not t:
        return fallback
    return re.sub(r"\s+", " ", t).strip()
