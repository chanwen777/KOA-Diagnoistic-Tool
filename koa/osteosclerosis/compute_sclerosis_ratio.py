# -*- coding: utf-8 -*-
"""
``compute_sclerosis_ratio`` — 软骨下骨硬化（SCL）像素比与百分比。

全图多标签 mask 的类别 ID 来自配置里的 ``label_mapping``（见 ``koa/configs/sclerosis_config.py``）。
具体股骨 / 胫骨 / 硬化分组由 ``sclerosis_label_sets_from_mapping`` 按类名规则解析。

共 **四个比例**（彼此独立）：

- **右股骨**：该侧「股骨硬化」类像素 ÷ **整段股骨腔室** 像素 ×100%（腔室 = plain 键如 ``Femur_R`` 所列 ID
  与 ``Femur_R_Osteosclerosis`` 的 ID **并集**，即非硬化与硬化标签共同构成股骨区域，如 1 与 5 之和为分母，5 为分子）。
- **右胫骨**、**左股骨**、**左胫骨** 同理。

分子仍是 **对应腔室** 的硬化标签像素；CSV 列名仍沿用 ``*_bone_pixels`` 表示 **分母侧腔室总像素**。

``label_mapping`` 的值可为 **int** 或 **int 列表**（如 ``Femur_R: [1, 5]``）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SclerosisLabelMapping = Dict[str, Any]

if TYPE_CHECKING:
    import pandas as pd


def _sclerosis_mapping_value_to_ids(val: Any) -> List[int]:
    """``label_mapping`` 值为 int 或 int 可迭代；0 忽略。"""
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


def sclerosis_label_sets_from_mapping(
    label_mapping: SclerosisLabelMapping,
) -> Dict[str, List[int]]:
    """
    从 ``label_mapping`` 拆出 ``sclerosis_ratios_bilateral_vs_bone`` 所需的 **八个** 列表。

    - **硬化（分子）**：``*_Osteosclerosis`` 键下全部 ID（可为 list）。
    - **腔室分母**：同名 plain 键（``Femur_R`` / ``Tibia_R`` / …，键名不含 ``Osteosclerosis``）下全部 ID
      与对应 ``Femur_R_Osteosclerosis`` 等键下 ID 的 **并集**（整段解剖腔像素 = 非硬化 + 硬化）。
    """
    keys = [k for k in label_mapping if k != "background"]

    def collect_ids(pred) -> List[int]:
        s: set[int] = set()
        for k in keys:
            if pred(k):
                s.update(_sclerosis_mapping_value_to_ids(label_mapping[k]))
        return sorted(s)

    def plain_femur_r(k: str) -> bool:
        return "Osteosclerosis" not in k and k.startswith("Femur") and "_R" in k

    def plain_tibia_r(k: str) -> bool:
        return "Osteosclerosis" not in k and k.startswith("Tibia") and "_R" in k

    def plain_femur_l(k: str) -> bool:
        return "Osteosclerosis" not in k and k.startswith("Femur") and "_L" in k and "_R" not in k

    def plain_tibia_l(k: str) -> bool:
        return "Osteosclerosis" not in k and k.startswith("Tibia") and "_L" in k and "_R" not in k

    def scl_femur_r(k: str) -> bool:
        return k.startswith("Femur") and "_R_Osteosclerosis" in k

    def scl_tibia_r(k: str) -> bool:
        return k.startswith("Tibia") and "_R_Osteosclerosis" in k

    def scl_femur_l(k: str) -> bool:
        return k.startswith("Femur") and "_L_Osteosclerosis" in k

    def scl_tibia_l(k: str) -> bool:
        return k.startswith("Tibia") and "_L_Osteosclerosis" in k

    s_rf = set(collect_ids(scl_femur_r))
    s_rt = set(collect_ids(scl_tibia_r))
    s_lf = set(collect_ids(scl_femur_l))
    s_lt = set(collect_ids(scl_tibia_l))

    p_rf = set(collect_ids(plain_femur_r))
    p_rt = set(collect_ids(plain_tibia_r))
    p_lf = set(collect_ids(plain_femur_l))
    p_lt = set(collect_ids(plain_tibia_l))

    return {
        "labels_femur_right": sorted(p_rf | s_rf),
        "labels_tibia_right": sorted(p_rt | s_rt),
        "labels_femur_left": sorted(p_lf | s_lf),
        "labels_tibia_left": sorted(p_lt | s_lt),
        "labels_sclerosis_right_femur": sorted(s_rf),
        "labels_sclerosis_right_tibia": sorted(s_rt),
        "labels_sclerosis_left_femur": sorted(s_lf),
        "labels_sclerosis_left_tibia": sorted(s_lt),
    }


def default_knee_scl_bone_label_sets(
    label_mapping: Optional[SclerosisLabelMapping] = None,
) -> Dict[str, List[int]]:
    """用于 ``sclerosis_ratios_bilateral_vs_bone`` 的列表；默认读 ``sclerosis_config`` 当前项的 ``label_mapping``。"""
    if label_mapping is None:
        from koa.configs.sclerosis_config import (
            CURRENT_SCLEROSIS_CONFIG_KEY,
            SCLEROSIS_CONFIGS,
        )

        label_mapping = SCLEROSIS_CONFIGS[CURRENT_SCLEROSIS_CONFIG_KEY]["label_mapping"]
    return sclerosis_label_sets_from_mapping(label_mapping)


@dataclass
class SclerosisCompartmentRatios:
    """双膝四分腔硬化比（%）及像素计数。``*_bone_pixels`` 为分母：该腔室 plain∪硬化 ID 的像素总和。"""

    pct_right_femur: float
    pct_right_tibia: float
    pct_left_femur: float
    pct_left_tibia: float
    right_femur_sclerosis_pixels: int
    right_femur_bone_pixels: int
    right_tibia_sclerosis_pixels: int
    right_tibia_bone_pixels: int
    left_femur_sclerosis_pixels: int
    left_femur_bone_pixels: int
    left_tibia_sclerosis_pixels: int
    left_tibia_bone_pixels: int


def sclerosis_ratios_bilateral_vs_bone(
    mask_2d: np.ndarray,
    *,
    labels_femur_right: Sequence[int],
    labels_tibia_right: Sequence[int],
    labels_femur_left: Sequence[int],
    labels_tibia_left: Sequence[int],
    labels_sclerosis_right_femur: Sequence[int],
    labels_sclerosis_right_tibia: Sequence[int],
    labels_sclerosis_left_femur: Sequence[int],
    labels_sclerosis_left_tibia: Sequence[int],
) -> SclerosisCompartmentRatios:
    """
    四个分腔比例：各为「该腔室硬化像素 / 该腔室全部体素（plain∪硬化标签）」×100%；分母为 0 时为 NaN。
    """
    m = mask_2d.astype(np.int32)

    def _sum_ids(labels: Sequence[int]) -> int:
        return sum(int(np.sum(m == int(v))) for v in labels)

    n_srf = _sum_ids(labels_sclerosis_right_femur)
    n_brf = _sum_ids(labels_femur_right)
    n_srt = _sum_ids(labels_sclerosis_right_tibia)
    n_brt = _sum_ids(labels_tibia_right)
    n_slf = _sum_ids(labels_sclerosis_left_femur)
    n_blf = _sum_ids(labels_femur_left)
    n_slt = _sum_ids(labels_sclerosis_left_tibia)
    n_blt = _sum_ids(labels_tibia_left)

    p_rf = float(np.nan) if n_brf == 0 else 100.0 * n_srf / n_brf
    p_rt = float(np.nan) if n_brt == 0 else 100.0 * n_srt / n_brt
    p_lf = float(np.nan) if n_blf == 0 else 100.0 * n_slf / n_blf
    p_lt = float(np.nan) if n_blt == 0 else 100.0 * n_slt / n_blt

    return SclerosisCompartmentRatios(
        pct_right_femur=p_rf,
        pct_right_tibia=p_rt,
        pct_left_femur=p_lf,
        pct_left_tibia=p_lt,
        right_femur_sclerosis_pixels=n_srf,
        right_femur_bone_pixels=n_brf,
        right_tibia_sclerosis_pixels=n_srt,
        right_tibia_bone_pixels=n_brt,
        left_femur_sclerosis_pixels=n_slf,
        left_femur_bone_pixels=n_blf,
        left_tibia_sclerosis_pixels=n_slt,
        left_tibia_bone_pixels=n_blt,
    )


def sclerosis_case_metrics_row(
    case_id: str,
    mask_2d: np.ndarray,
    label_mapping: SclerosisLabelMapping,
) -> Dict[str, Any]:
    """
    单例一行：四分腔硬化像素、腔室分母像素（``*_bone_pixels``）与四个百分比（与脚本 / 笔记本 CSV 列一致）。
    """
    lbl = sclerosis_label_sets_from_mapping(label_mapping)
    r = sclerosis_ratios_bilateral_vs_bone(mask_2d, **lbl)
    return {
        "case_id": case_id,
        "right_femur_sclerosis_pixels": r.right_femur_sclerosis_pixels,
        "right_femur_bone_pixels": r.right_femur_bone_pixels,
        "right_tibia_sclerosis_pixels": r.right_tibia_sclerosis_pixels,
        "right_tibia_bone_pixels": r.right_tibia_bone_pixels,
        "right_femur_sclerosis_pct": r.pct_right_femur,
        "right_tibia_sclerosis_pct": r.pct_right_tibia,
        "left_femur_sclerosis_pixels": r.left_femur_sclerosis_pixels,
        "left_femur_bone_pixels": r.left_femur_bone_pixels,
        "left_tibia_sclerosis_pixels": r.left_tibia_sclerosis_pixels,
        "left_tibia_bone_pixels": r.left_tibia_bone_pixels,
        "left_femur_sclerosis_pct": r.pct_left_femur,
        "left_tibia_sclerosis_pct": r.pct_left_tibia,
    }


def _sitk_mask_to_2d_int(mask_img: Any) -> np.ndarray:
    import SimpleITK as sitk

    arr = sitk.GetArrayFromImage(mask_img)
    if arr.ndim == 3:
        arr = arr[0]
    return np.asarray(arr, dtype=np.int32)


def sclerosis_results_dataframe_from_config(cfg: Dict[str, Any]) -> "pd.DataFrame":
    """
    按 ``sclerosis_config`` 式配置批量读 ``mask_dir``、写与 ``sclerosis_case_metrics_row`` 相同结构的行。
    需 ``mask_dir``、``label_mapping``；病例列表来自 ``list_case_ids_from_config``。
    """
    import pandas as pd
    from pathlib import Path

    from koa.utils.case_list import (
        find_mask_path,
        list_case_ids_from_config,
        resolve_volume_extensions,
    )
    from koa.utils.sitk_utils import load_sitk_image

    mask_dir = Path(cfg["mask_dir"])
    lm = cfg["label_mapping"]
    exts = tuple(resolve_volume_extensions(cfg))
    rows: List[Dict[str, Any]] = []
    for cid in list_case_ids_from_config(cfg):
        mp = find_mask_path(mask_dir, cid, exts)
        if mp is None:
            continue
        msk = load_sitk_image(str(mp))
        mask_2d = _sitk_mask_to_2d_int(msk)
        rows.append(sclerosis_case_metrics_row(cid, mask_2d, lm))
    return pd.DataFrame(rows)


def sclerosis_bilateral_overlay_figure(
    image_2d: np.ndarray,
    mask_2d: np.ndarray,
    label_mapping: SclerosisLabelMapping,
):
    """
    双膝 SCL 叠图（半透明硬化 + 命名 bbox 图例），与 ``scripts/osteoscierosis.py`` 单例/批量一致。
    视野为 **膝关节裁剪**：以全部硬化像素的包围盒为基准加边距（与 JSN 展示图同量级 padding），保证图中包含每一处硬化；无硬化像素时改为股骨/胫骨腔室并集。
    返回 ``(matplotlib.figure.Figure, SclerosisCompartmentRatios)``。
    """
    import matplotlib.pyplot as plt

    from koa.utils.bilateral_viz import (
        plot_bilateral_overlay,
        zoom_limits_from_mask_label_union,
    )

    # 计算四个分腔：右/左 femur、右/左 tibia。
    lbl = sclerosis_label_sets_from_mapping(label_mapping)
    r = sclerosis_ratios_bilateral_vs_bone(mask_2d, **lbl)
    pr = float(np.nanmean([r.pct_right_femur, r.pct_right_tibia]))
    pl = float(np.nanmean([r.pct_left_femur, r.pct_left_tibia]))

    # 半透明叠色：仅 **硬化** 标签（plain 股骨/胫骨主体不参与上色）。
    overlay_labels = list(
        dict.fromkeys(
            lbl["labels_sclerosis_right_femur"]
            + lbl["labels_sclerosis_right_tibia"]
            + lbl["labels_sclerosis_left_femur"]
            + lbl["labels_sclerosis_left_tibia"]
        )
    )

    # bbox 仍然仅用于画硬化类（osteosclerosis 键）的连通域。
    named_bbox = sclerosis_named_bbox_layers_from_label_mapping(label_mapping)
    # Knee crop: union bbox of **all** sclerosis pixels (guaranteed in view) + padding like JSN.
    # If no sclerosis voxels, fall back to femur/tibia compartment union so the frame still shows the knee.
    bone_union_ids = sorted(
        set(lbl["labels_femur_right"])
        | set(lbl["labels_tibia_right"])
        | set(lbl["labels_femur_left"])
        | set(lbl["labels_tibia_left"])
    )
    lims = zoom_limits_from_mask_label_union(
        mask_2d, overlay_labels, image_2d.shape
    )
    if lims is None and bone_union_ids:
        lims = zoom_limits_from_mask_label_union(
            mask_2d, bone_union_ids, image_2d.shape
        )
    fig = plot_bilateral_overlay(
        image_2d,
        mask_2d,
        overlay_labels,
        pr,
        pl,
        overlay_rgba=(0.0, 0.75, 0.85, 0.4),
        title="Subchondral sclerosis (SCL) — bilateral overlay",
        label_right="Right knee (mean femur/tibia sclerosis %)",
        label_left="Left knee (mean femur/tibia sclerosis %)",
        named_bbox_layers=named_bbox if named_bbox else None,
        pct_right_femur=r.pct_right_femur,
        pct_right_tibia=r.pct_right_tibia,
        pct_left_femur=r.pct_left_femur,
        pct_left_tibia=r.pct_left_tibia,
        axis_zoom_limits=lims,
    )
    return fig, r


def sclerosis_named_bbox_layers_from_label_mapping(
    label_mapping: SclerosisLabelMapping,
) -> List[Tuple[str, List[int]]]:
    """
    可视化用：仅 **硬化** 键（键名含 ``osteosclerosis`` 子串，**不区分大小写**），按 ``label_mapping`` **插入顺序** 每层一条图例。
    plain 骨键（如 ``Femur_R``）不参与 bbox，与叠图一致（叠图仅高亮硬化类）。
    """
    out: List[Tuple[str, List[int]]] = []
    for name, val in label_mapping.items():
        nm = str(name)
        if nm.lower() == "background" or "osteosclerosis" not in nm.lower():
            continue
        ids = _sclerosis_mapping_value_to_ids(val)
        if ids:
            out.append((name, ids))
    return out
