# -*- coding: utf-8 -*-
"""
``compute_osteophyte_ratio`` — 髌骨区骨刺（OST）像素比与百分比。

定义（用文字说明）
----------------
对每个 **独立成像视野**（一张髌骨图，对应 **单一患者侧**：左或右各一张 volume）：

- **推荐**：``label_mapping`` 中 ``Patella`` 为 int 或 int 列表（髌骨区域全部类别，如 ``[1, 2]``），
  ``Patella_Osteophyte`` 为骨赘类别（如 ``2``）。
- **分母（髌骨区域总像素）**：``Patella`` 与 ``Patella_Osteophyte`` 所列 ID 的 **并集** 在 mask 上的像素和
  （与硬化模块「腔室 plain∪硬化」一致；如 1+2 覆盖整块髌骨）。
- **分子（骨赘像素）**：默认 ``Patella_Osteophyte`` 下各 ID 像素和；若传入 ``osteophyte_label_id`` 则只统计该 ID。
- **百分比**：``100 * 骨赘像素 / 分母``；分母为 0 → NaN。

**回退**：若 mapping 中无 ``Patella`` / ``Patella_Osteophyte``，则使用 ``patella_label_ids``（须为两个不同整数），
分子由显式 ``osteophyte_label_id`` 或「少像素 = 骨刺」自动规则决定。

**工作流**：左、右 **各一张** mask → ``osteophyte_ratios_lr_files_auto``，返回 ``(右膝, 左膝)``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class OsteophyteSideAutoResult:
    """单侧 OST：``percentage`` = 骨赘像素 / 髌骨区域（分母）像素 ×100（%）。"""

    side: str
    osteophyte_label: int
    patella_label_other: int
    osteophyte_pixels: int
    patella_pixels: int
    percentage: float


def _ost_mapping_value_to_ids(val: Any) -> List[int]:
    """与硬化模块一致：值为 int 或 int 可迭代；0 忽略。"""
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        out: set[int] = set()
        for x in val:
            try:
                xi = int(x)
            except (TypeError, ValueError):
                continue
            if xi != 0:
                out.add(xi)
        return sorted(out)
    try:
        vi = int(val)
    except (TypeError, ValueError):
        return []
    return [vi] if vi != 0 else []


def osteophyte_label_sets_from_config(cfg: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    """
    返回 ``(髌骨区域分母 ID 列表, 骨赘分子 ID 列表)``。
    分子列表 **为空** 表示走 ``patella_label_ids`` 双类自动/显式规则。
    """
    lm = cfg.get("label_mapping") or {}
    p_ids = _ost_mapping_value_to_ids(lm.get("Patella"))
    o_ids = _ost_mapping_value_to_ids(lm.get("Patella_Osteophyte"))

    if p_ids or o_ids:
        denom = sorted(set(p_ids) | set(o_ids))
        num = sorted(set(o_ids))
        return denom, num

    pl = cfg.get("patella_label_ids")
    if not pl or len(pl) != 2 or int(pl[0]) == int(pl[1]):
        raise ValueError(
            "需配置 label_mapping 的 Patella / Patella_Osteophyte，或提供两个不同的 patella_label_ids"
        )
    a, b = int(pl[0]), int(pl[1])
    return sorted([a, b]), []


def resolve_osteophyte_label_id(
    cfg: Dict[str, Any],
    *,
    cli_override: Optional[int] = None,
) -> Optional[int]:
    """
    显式骨刺标签 ID；返回 ``None`` 时在 **双类回退** 模式下使用「少像素 = 骨刺」。
    （mapping 模式下若已配置 ``Patella_Osteophyte`` 可省略。）

    优先级：``cli_override`` > ``cfg['osteophyte_label_id']`` >
    ``cfg['label_mapping']['Patella_Osteophyte']``（单个 int 或单元素 list/tuple）。
    """
    if cli_override is not None:
        return int(cli_override)
    raw = cfg.get("osteophyte_label_id")
    if raw is not None:
        return int(raw)
    lm = cfg.get("label_mapping") or {}
    po = lm.get("Patella_Osteophyte")
    if po is None:
        return None
    if isinstance(po, int) and not isinstance(po, bool):
        return int(po)
    if isinstance(po, (list, tuple)) and len(po) == 1:
        return int(po[0])
    return None


def _pick_osteophyte_label(
    count_a: int,
    count_b: int,
    label_a: int,
    label_b: int,
    *,
    tie_osteophyte_is_higher_id: bool = True,
) -> Tuple[int, int]:
    """返回 (osteophyte_label, other_label)。"""
    if count_a < count_b:
        return label_a, label_b
    if count_b < count_a:
        return label_b, label_a
    if tie_osteophyte_is_higher_id:
        hi, lo = (label_b, label_a) if label_b > label_a else (label_a, label_b)
        return hi, lo
    lo, hi = (label_a, label_b) if label_a < label_b else (label_b, label_a)
    return lo, hi


def osteophyte_ratio_full_field(
    mask_2d: np.ndarray,
    cfg: Dict[str, Any],
    *,
    side: str,
    tie_osteophyte_is_higher_id: bool = True,
    osteophyte_label_id: Optional[int] = None,
) -> OsteophyteSideAutoResult:
    """
    单张图全视野：骨赘像素 / 髌骨区域（``Patella`` ∪ ``Patella_Osteophyte``）像素。
    """
    denom_ids, num_ids = osteophyte_label_sets_from_config(cfg)
    m = mask_2d.astype(np.int32)

    def _count_ids(ids: Sequence[int]) -> int:
        return sum(int(np.sum(m == int(v))) for v in ids)

    pat_px = _count_ids(denom_ids)
    if not denom_ids:
        raise ValueError("髌骨区域标签列表为空，请检查 label_mapping / patella_label_ids")

    if num_ids:
        if osteophyte_label_id is not None:
            ost_lab = int(osteophyte_label_id)
            if ost_lab not in denom_ids:
                raise ValueError(
                    f"osteophyte_label_id={ost_lab} 不在髌骨区域标签 {denom_ids} 中"
                )
            ost_px = int(np.sum(m == ost_lab))
        else:
            ost_px = _count_ids(num_ids)
            ost_lab = num_ids[0]
        others = [x for x in denom_ids if x not in set(num_ids)]
        if others:
            other_lab = others[0]
        else:
            other_lab = denom_ids[0] if denom_ids[0] != ost_lab else denom_ids[-1]
    else:
        if len(denom_ids) != 2:
            raise ValueError(
                "缺少 Patella_Osteophyte 时须通过 patella_label_ids 提供恰好两个髌骨标签"
            )
        label_a, label_b = denom_ids[0], denom_ids[1]
        ca = int(np.sum(m == label_a))
        cb = int(np.sum(m == label_b))
        if osteophyte_label_id is not None:
            ost_lab = int(osteophyte_label_id)
            if ost_lab not in (label_a, label_b):
                raise ValueError(
                    f"osteophyte_label_id={ost_lab} 不在 patella_label_ids={denom_ids} 中"
                )
            other_lab = label_b if ost_lab == label_a else label_a
        else:
            ost_lab, other_lab = _pick_osteophyte_label(
                ca,
                cb,
                label_a,
                label_b,
                tie_osteophyte_is_higher_id=tie_osteophyte_is_higher_id,
            )
        ost_px = int(np.sum(m == ost_lab))

    pct = float(np.nan) if pat_px == 0 else 100.0 * ost_px / pat_px
    return OsteophyteSideAutoResult(
        side=side,
        osteophyte_label=ost_lab,
        patella_label_other=other_lab,
        osteophyte_pixels=ost_px,
        patella_pixels=pat_px,
        percentage=pct,
    )


def osteophyte_ratios_lr_files_auto(
    mask_left: np.ndarray,
    mask_right: np.ndarray,
    cfg: Dict[str, Any],
    *,
    tie_osteophyte_is_higher_id: bool = True,
    osteophyte_label_id: Optional[int] = None,
) -> Tuple[OsteophyteSideAutoResult, OsteophyteSideAutoResult]:
    """
    左、右 **各自一张** mask → 各算一条骨刺占比百分数；
    返回 ``(右膝结果, 左膝结果)``（与 ``plot_lr_knee_images_overlay`` 等约定一致）。
    """
    r = osteophyte_ratio_full_field(
        mask_right,
        cfg,
        side="right",
        tie_osteophyte_is_higher_id=tie_osteophyte_is_higher_id,
        osteophyte_label_id=osteophyte_label_id,
    )
    lres = osteophyte_ratio_full_field(
        mask_left,
        cfg,
        side="left",
        tie_osteophyte_is_higher_id=tie_osteophyte_is_higher_id,
        osteophyte_label_id=osteophyte_label_id,
    )
    return r, lres


def osteophyte_bbox_label_ids_from_config(cfg: Dict[str, Any]) -> List[int]:
    """
    静态配置下的 **骨刺类** bbox 用标签 ID（不含髌骨主体）；脚本出图优先用每侧
    ``OsteophyteSideAutoResult.osteophyte_label``。

    解析顺序：

    1. ``osteophyte_bbox_label_ids`` 若存在（可为空列表以关闭框）。
    2. ``osteophyte_bbox_label_mapping_keys`` → ``label_ids_from_mapping_keys``。
    3. ``label_mapping`` 中 ``Patella_Osteophyte``。
    4. ``resolve_osteophyte_label_id(cfg)`` 若得单一 ID。
    5. 否则 ``[]``。
    """
    from koa.utils.label_mapping import label_ids_from_mapping_keys

    raw = cfg.get("osteophyte_bbox_label_ids")
    if raw is not None:
        return [int(x) for x in raw]
    keys = cfg.get("osteophyte_bbox_label_mapping_keys")
    if keys is not None:
        return label_ids_from_mapping_keys(cfg["label_mapping"], keys)
    lm = cfg["label_mapping"]
    if "Patella_Osteophyte" in lm:
        return label_ids_from_mapping_keys(lm, ["Patella_Osteophyte"])
    ost = resolve_osteophyte_label_id(cfg)
    return [int(ost)] if ost is not None else []


__all__ = [
    "OsteophyteSideAutoResult",
    "osteophyte_bbox_label_ids_from_config",
    "osteophyte_label_sets_from_config",
    "osteophyte_ratio_full_field",
    "osteophyte_ratios_lr_files_auto",
    "resolve_osteophyte_label_id",
]
