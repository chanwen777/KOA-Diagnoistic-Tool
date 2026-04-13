# -*- coding: utf-8 -*-
"""
在膝关节 X 线图上叠加半透明分割，并标注左 / 右（患者）侧比例文字。

传入的 ``pct_right`` / ``pct_left`` 常为 **患者右 / 左** 各一个代表比例
（硬化模块可传股骨/胫骨两比例的均值，骨刺模块传髌骨相关占比；见各自 ``compute_*`` 模块）。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy import ndimage

# ``named_bbox_layers`` 每层一种颜色（循环使用）
_NAMED_BBOX_EDGE_COLORS = (
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
)


def zoom_limits_from_mask_label_union(
    mask_2d: np.ndarray,
    label_ids: Sequence[int],
    image_shape: Tuple[int, ...],
    *,
    margin_frac: float = 0.58,
    min_pad_px: int = 120,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Bounding box around **all** pixels whose label is in ``label_ids``, then pad (same spirit as JSN knee crop).

    Returns ``(c0, c1, r0, r1)`` inclusive pixel indices, or ``None`` if no matching pixels.
    """
    if not label_ids:
        return None
    m = mask_2d.astype(np.int32)
    sel = np.zeros(m.shape, dtype=bool)
    for v in label_ids:
        sel |= m == int(v)
    if not np.any(sel):
        return None
    ys, xs = np.where(sel)
    r0, r1 = float(ys.min()), float(ys.max())
    c0, c1 = float(xs.min()), float(xs.max())
    H, W = int(image_shape[0]), int(image_shape[1])
    rh, cw = max(1e-6, r1 - r0), max(1e-6, c1 - c0)
    pad_r = max(float(min_pad_px), rh * margin_frac, rh * 0.62, H * 0.17)
    pad_c = max(float(min_pad_px), cw * margin_frac, cw * 0.62, W * 0.15)
    r0i = int(max(0, np.floor(r0 - pad_r)))
    r1i = int(min(H - 1, np.ceil(r1 + pad_r)))
    c0i = int(max(0, np.floor(c0 - pad_c)))
    c1i = int(min(W - 1, np.ceil(c1 + pad_c)))
    if r1i <= r0i or c1i <= c0i:
        return None
    return (c0i, c1i, r0i, r1i)


def apply_axis_zoom(ax: plt.Axes, lims: Optional[Tuple[int, int, int, int]]) -> None:
    """``lims`` is ``(c0, c1, r0, r1)`` in pixel indices; ``imshow`` uses y increasing downward."""
    if lims is None:
        return
    c0, c1, r0, r1 = lims
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)


def _to_gray01(image_2d: np.ndarray) -> np.ndarray:
    x = image_2d.astype(np.float64)
    x = x - np.nanmin(x)
    ma = np.nanmax(x)
    if ma > 0:
        x = x / ma
    return np.clip(x, 0, 1)


def plot_bilateral_overlay(
    image_2d: np.ndarray,
    mask_2d: np.ndarray,
    overlay_label_values: Sequence[int],
    pct_right: float,
    pct_left: float,
    *,
    overlay_rgba: Tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.35),
    title: str = "",
    label_right: str = "Right knee",
    label_left: str = "Left knee",
    show_midline: bool = True,
    figsize: Tuple[float, float] = (10, 8),
    draw_region_bbox: bool = True,
    bbox_label_ids: Optional[Sequence[int]] = None,
    bbox_edgecolor: str = "#00ff88",
    named_bbox_layers: Optional[Sequence[Tuple[str, Sequence[int]]]] = None,
    legend_overlay_label: str = "Sclerosis overlay",
    legend_bbox_label: str = "ROI bounding box",
    # SCL（硬化）需要显示四个分腔百分比（右/左 femur + 右/左 tibia）。
    pct_right_femur: Optional[float] = None,
    pct_right_tibia: Optional[float] = None,
    pct_left_femur: Optional[float] = None,
    pct_left_tibia: Optional[float] = None,
    axis_zoom_limits: Optional[Tuple[int, int, int, int]] = None,
) -> plt.Figure:
    """
    显示灰度图 + 指定标签的彩色半透明叠加。

    若传入四个分腔 ``pct_*_femur/tibia``（硬化模块），百分比以 **仅文字** 条目出现在 **右侧** ``fig.legend`` 中，不再单独在图底画文本框。

    pct_right / pct_left 顺序与 ``osteophyte_ratios_lr_files_auto``（右先、左后）一致；
    硬化模块常传该侧股骨/胫骨两腔比例的均值。

    ``axis_zoom_limits``：若给定 ``(c0, c1, r0, r1)`` 列/行像素范围，在绘图结束后 ``set_xlim`` / ``set_ylim`` 做膝关节裁剪（硬化模块用：先包住全部硬化像素再扩边）。

    ``named_bbox_layers``：若提供（非空），按 **全图** 对每层 ``(图例名, 标签 ID 列表)`` 做连通域：
    **同一层共用一种描边色**（如图例名 ``Femur_R_Osteosclerosis``），该层下每个非连通斑块各画一个矩形。
    图例中每层一条。**不再**按图像左右半幅分栏。提供时忽略 ``bbox_label_ids`` 的左右分栏逻辑。

    未提供 ``named_bbox_layers`` 时：``bbox_label_ids`` 对这些标签画框；每标签每个连通域一个矩形；
    并在 **左 / 右半幅** 分别计算连通域。为 ``None`` 时与 ``overlay_label_values`` 相同。
    """
    gray = _to_gray01(image_2d)
    rgb = np.stack([gray, gray, gray], axis=-1)
    m = mask_2d.astype(np.int32)
    ov = np.zeros_like(rgb)
    sel = np.zeros(m.shape, dtype=bool)
    for v in overlay_label_values:
        sel |= m == int(v)
    ov[sel] = overlay_rgba[:3]
    blend = rgb.copy()
    a = overlay_rgba[3]
    blend[sel] = (1 - a) * blend[sel] + a * ov[sel]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        blend,
        cmap=None,
        vmin=0,
        vmax=1,
        aspect="equal",
        interpolation="nearest",
    )
    ax.axis("off")
    if title:
        ax.set_title(title)

    h, w = m.shape
    if show_midline:
        mid = w // 2
        ax.axvline(mid - 0.5, color="cyan", linestyle="--", linewidth=1, alpha=0.8)

    legend_handles: List[Any] = []
    drew_bbox = False
    _, w = m.shape

    if named_bbox_layers:
        for idx, (layer_name, lids) in enumerate(named_bbox_layers):
            if not lids:
                continue
            ec = _NAMED_BBOX_EDGE_COLORS[idx % len(_NAMED_BBOX_EDGE_COLORS)]
            layer_drew = False
            for bb in _bboxes_cc_in_region(m, lids, col_lo=0, col_hi_exclusive=w):
                drew_bbox = True
                layer_drew = True
                c0, c1, r0, r1 = bb
                _add_bbox_rect(ax, c0, c1, r0, r1, edgecolor=ec)
            if layer_drew:
                legend_handles.append(
                    Line2D([0], [0], color=ec, linewidth=2, label=layer_name)
                )
    else:
        bbox_src = list(bbox_label_ids) if bbox_label_ids is not None else list(
            overlay_label_values
        )
        if draw_region_bbox and bbox_src:
            mid = max(1, w // 2)
            for bb in _bboxes_cc_in_region(m, bbox_src, col_lo=0, col_hi_exclusive=mid):
                drew_bbox = True
                c0, c1, r0, r1 = bb
                _add_bbox_rect(ax, c0, c1, r0, r1, edgecolor=bbox_edgecolor)
            for bb in _bboxes_cc_in_region(m, bbox_src, col_lo=mid, col_hi_exclusive=w):
                drew_bbox = True
                c0, c1, r0, r1 = bb
                _add_bbox_rect(ax, c0, c1, r0, r1, edgecolor=bbox_edgecolor)
    if overlay_label_values:
        legend_handles.append(
            Patch(
                facecolor=(*overlay_rgba[:3], min(1.0, overlay_rgba[3])),
                edgecolor="none",
                label=legend_overlay_label,
            )
        )
    if drew_bbox and not named_bbox_layers:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=bbox_edgecolor,
                linewidth=2,
                label=legend_bbox_label,
            )
        )
    if (
        pct_right_femur is not None
        or pct_right_tibia is not None
        or pct_left_femur is not None
        or pct_left_tibia is not None
    ):
        def _fmt_scl(p: Optional[float]) -> str:
            if p is None or p != p:
                return "—%"
            return f"{p:.2f}%"

        for name, val in (
            ("Left femur", pct_left_femur),
            ("Right femur", pct_right_femur),
            ("Left tibia", pct_left_tibia),
            ("Right tibia", pct_right_tibia),
        ):
            legend_handles.append(
                Line2D([], [], linestyle="none", label=f"{name}: {_fmt_scl(val)}")
            )
    # Zoom first so axis limits are final; then reserve right margin + fig.legend in figure coords.
    # (Calling legend before apply_axis_zoom made the legend relocate to the bottom after set_xlim/ylim.)
    apply_axis_zoom(ax, axis_zoom_limits)
    fig.subplots_adjust(bottom=0.1, right=0.72, top=0.92)
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            bbox_to_anchor=(0.74, 0.52),
            loc="center left",
            bbox_transform=fig.transFigure,
            fontsize=9,
            framealpha=0.9,
            borderaxespad=0,
        )
    return fig


def _blend_overlay(
    image_2d: np.ndarray,
    mask_2d: np.ndarray,
    overlay_label_values: Sequence[int],
    overlay_rgba: Tuple[float, float, float, float],
) -> np.ndarray:
    gray = _to_gray01(image_2d)
    rgb = np.stack([gray, gray, gray], axis=-1)
    m = mask_2d.astype(np.int32)
    ov = np.zeros_like(rgb)
    sel = np.zeros(m.shape, dtype=bool)
    for v in overlay_label_values:
        sel |= m == int(v)
    ov[sel] = overlay_rgba[:3]
    blend = rgb.copy()
    a = overlay_rgba[3]
    blend[sel] = (1 - a) * blend[sel] + a * ov[sel]
    return blend


def _fmt_pct(p: float) -> str:
    if p != p:
        return "—%"
    return f"{p:.2f}%"


def _pad_image_mask_vertical_center(
    image_2d: np.ndarray, mask_2d: np.ndarray, target_h: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad top/bottom so height ``target_h``; masks stay aligned for bbox drawing."""
    h = int(image_2d.shape[0])
    if h == target_h:
        return np.asarray(image_2d), np.asarray(mask_2d)
    pt = (target_h - h) // 2
    pb = target_h - h - pt
    pad2 = ((pt, pb), (0, 0))
    im = np.pad(np.asarray(image_2d), pad2, mode="constant", constant_values=0)
    ms = np.pad(np.asarray(mask_2d), pad2, mode="constant", constant_values=0)
    return im, ms.astype(np.asarray(mask_2d).dtype, copy=False)


def _bboxes_cc_in_region(
    mask_2d: np.ndarray,
    labels: Sequence[int],
    *,
    col_lo: Optional[int] = None,
    col_hi_exclusive: Optional[int] = None,
) -> List[Tuple[int, int, int, int]]:
    """
    One axis-aligned box per connected component per label, after optional column crop.
    Returns (cmin, cmax, rmin, rmax) inclusive.
    """
    m = mask_2d.astype(np.int32)
    _, w = m.shape
    clo = int(col_lo if col_lo is not None else 0)
    chi = int(col_hi_exclusive if col_hi_exclusive is not None else w)
    out: List[Tuple[int, int, int, int]] = []
    for v in labels:
        sel = (m == int(v)).astype(np.uint8)
        sel[:, :clo] = 0
        sel[:, chi:] = 0
        if not np.any(sel):
            continue
        labeled, nlab = ndimage.label(sel)
        for i in range(1, nlab + 1):
            ys, xs = np.where(labeled == i)
            if ys.size == 0:
                continue
            out.append(
                (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
            )
    return out


def _add_bbox_rect(
    ax: plt.Axes,
    c0: int,
    c1: int,
    r0: int,
    r1: int,
    *,
    edgecolor: str,
    linewidth: float = 2.0,
) -> None:
    rect = Rectangle(
        (c0 - 0.5, r0 - 0.5),
        c1 - c0 + 1,
        r1 - r0 + 1,
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor="none",
    )
    ax.add_patch(rect)


def plot_lr_knee_images_overlay(
    image_left: np.ndarray,
    mask_left: np.ndarray,
    pct_left: float,
    image_right: np.ndarray,
    mask_right: np.ndarray,
    pct_right: float,
    overlay_label_values: Sequence[int],
    *,
    overlay_label_values_left: Optional[Sequence[int]] = None,
    overlay_label_values_right: Optional[Sequence[int]] = None,
    overlay_rgba: Tuple[float, float, float, float] = (1.0, 0.85, 0.0, 0.45),
    suptitle: str = "",
    subtitle_left: str = "Left knee",
    subtitle_right: str = "Right knee",
    figsize: Tuple[float, float] = (14, 6),
    bbox_label_ids: Optional[Sequence[int]] = None,
    bbox_label_ids_left: Optional[Sequence[int]] = None,
    bbox_label_ids_right: Optional[Sequence[int]] = None,
    draw_osteophyte_bbox: bool = True,
    bbox_edgecolor: str = "#00ff88",
    legend_overlay_label: str = "Osteophyte overlay",
    legend_bbox_label: str = "ROI bounding box",
    wspace: float = 0.0,
) -> plt.Figure:
    """
    左 / 右膝 **各一张图** 时：左右在 **同一水平条** 上 **水平拼接** 为一张图（无中间缝隙；
    避免双子图 ``aspect=equal`` 各自留白造成的「中间大间隙」）。

    参数仍按 **患者左** / **患者右** 传入（``*_L`` / ``*_R`` 数据）；
    **显示顺序**为观者从左到右：**右膝 | 左膝**（与常见读片排版一致）。

    ``wspace`` 保留为兼容旧调用，当前实现中 **忽略**（单轴拼接图）。

    ``overlay_label_values_right`` / ``overlay_label_values_left``：若指定，则该侧叠图 **仅** 高亮这些
    标签（例如仅骨刺类）；未指定的一侧回退用 ``overlay_label_values``。

    ``bbox_label_ids*``：仅对这些标签画框（每标签每连通域一框）。未指定侧别时用
    ``bbox_label_ids``；可分别用 ``bbox_label_ids_left`` / ``bbox_label_ids_right`` 覆盖。
    全部为 ``None`` 时不画框。
    """
    _ = wspace  # API compatibility; stitching uses one axes (no subplot spacing)
    base_ov = list(overlay_label_values)
    ov_r = (
        list(overlay_label_values_right)
        if overlay_label_values_right is not None
        else base_ov
    )
    ov_l = (
        list(overlay_label_values_left)
        if overlay_label_values_left is not None
        else base_ov
    )
    # Same height so horizontal concat has one continuous row of pixels (no seam gap).
    H = max(int(image_left.shape[0]), int(image_right.shape[0]))
    img_l, msk_l = _pad_image_mask_vertical_center(image_left, mask_left, H)
    img_r, msk_r = _pad_image_mask_vertical_center(image_right, mask_right, H)
    blend_l = _blend_overlay(img_l, msk_l, ov_l, overlay_rgba)
    blend_r = _blend_overlay(img_r, msk_r, ov_r, overlay_rgba)
    combo = np.concatenate([blend_r, blend_l], axis=1)
    w_r = int(blend_r.shape[1])
    w_l = int(blend_l.shape[1])
    w_tot = w_r + w_l

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(combo, vmin=0, vmax=1, aspect="equal", interpolation="nearest")
    ax.axis("off")
    ax.margins(0)

    txt_l = f"Osteophyte (region %): {_fmt_pct(pct_left)}"
    txt_r = f"Osteophyte (region %): {_fmt_pct(pct_right)}"

    if draw_osteophyte_bbox:
        w_r_m = int(msk_r.shape[1])
        w_l_m = int(msk_l.shape[1])

        def _ids_for_right() -> List[int]:
            if bbox_label_ids_right is not None:
                return [int(x) for x in bbox_label_ids_right]
            if bbox_label_ids is not None:
                return [int(x) for x in bbox_label_ids]
            return []

        def _ids_for_left() -> List[int]:
            if bbox_label_ids_left is not None:
                return [int(x) for x in bbox_label_ids_left]
            if bbox_label_ids is not None:
                return [int(x) for x in bbox_label_ids]
            return []

        any_bbox = False
        for bb in _bboxes_cc_in_region(
            msk_r, _ids_for_right(), col_lo=0, col_hi_exclusive=w_r_m
        ):
            any_bbox = True
            c0, c1, r0, r1 = bb
            _add_bbox_rect(ax, c0, c1, r0, r1, edgecolor=bbox_edgecolor)
        for bb in _bboxes_cc_in_region(
            msk_l, _ids_for_left(), col_lo=0, col_hi_exclusive=w_l_m
        ):
            any_bbox = True
            c0, c1, r0, r1 = bb
            _add_bbox_rect(
                ax, c0 + w_r, c1 + w_r, r0, r1, edgecolor=bbox_edgecolor
            )
        show_legend_overlay = bool(ov_r) or bool(ov_l)
        if any_bbox or show_legend_overlay:
            handles = [
                Patch(
                    facecolor=(*overlay_rgba[:3], min(1.0, overlay_rgba[3])),
                    edgecolor="none",
                    label=legend_overlay_label,
                ),
            ]
            if any_bbox:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=bbox_edgecolor,
                        linewidth=2,
                        label=legend_bbox_label,
                    )
                )
            fig.legend(
                handles=handles,
                loc="upper center",
                ncol=2,
                fontsize=9,
                framealpha=0.92,
                bbox_to_anchor=(0.5, 0.02),
                bbox_transform=fig.transFigure,
            )
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.2)
    pos = ax.get_position()
    # Titles and stats: figure coords from single axes bbox (left = right knee, right = left knee)
    fig.text(
        pos.x0 + pos.width * (w_r / (2.0 * w_tot)),
        pos.y1 + 0.012,
        subtitle_right,
        ha="center",
        va="bottom",
        fontsize=12,
        transform=fig.transFigure,
    )
    fig.text(
        pos.x0 + pos.width * ((w_r + w_l / 2.0) / w_tot),
        pos.y1 + 0.012,
        subtitle_left,
        ha="center",
        va="bottom",
        fontsize=12,
        transform=fig.transFigure,
    )
    fig.text(
        pos.x0 + pos.width * (w_r / (2.0 * w_tot)),
        pos.y0 - 0.008,
        txt_r,
        ha="center",
        va="top",
        fontsize=12,
        transform=fig.transFigure,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        family="sans-serif",
    )
    fig.text(
        pos.x0 + pos.width * ((w_r + w_l / 2.0) / w_tot),
        pos.y0 - 0.008,
        txt_l,
        ha="center",
        va="top",
        fontsize=12,
        transform=fig.transFigure,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        family="sans-serif",
    )
    return fig
