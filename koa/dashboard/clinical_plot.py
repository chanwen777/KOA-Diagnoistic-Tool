# -*- coding: utf-8 -*-
"""
2×2 clinical-style layout: top row bilateral AP (R/L) with JSN + optional SCL;
bottom row patellar views with OST arrows. Matches common KOA reporting boards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from koa.osteophyte import osteophyte_ratios_lr_files_auto
from koa.utils.plot_text import sanitize_plot_text


def _to_gray01(image_2d: np.ndarray) -> np.ndarray:
    x = image_2d.astype(np.float64)
    x = x - np.nanmin(x)
    ma = np.nanmax(x)
    if ma > 0:
        x = x / ma
    return np.clip(x, 0, 1)


def _centroid_label(mask: np.ndarray, label: int) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask.astype(np.int32) == int(label))
    if ys.size == 0:
        return None
    return float(np.mean(ys)), float(np.mean(xs))


def _draw_jsn_segments(
    ax: plt.Axes,
    jsn_result: Dict[str, Any],
    patient_side: str,
    color: str = "magenta",
    lw: float = 2.0,
) -> int:
    """Draw femur–tibia segments for medial/lateral of patient ``left`` or ``right``. Returns count drawn."""
    n = 0
    for part in ("medial", "lateral"):
        f_key = f"jsn_{patient_side}_{part}_femur_pt"
        t_key = f"jsn_{patient_side}_{part}_tibia_pt"
        f_pt = jsn_result.get(f_key)
        t_pt = jsn_result.get(t_key)
        if f_pt is None or t_pt is None:
            continue
        f_pt = np.asarray(f_pt, dtype=float).ravel()
        t_pt = np.asarray(t_pt, dtype=float).ravel()
        if f_pt.size < 2 or t_pt.size < 2:
            continue
        r0, c0 = f_pt[0], f_pt[1]
        r1, c1 = t_pt[0], t_pt[1]
        ax.plot([c0, c1], [r0, r1], color=color, linewidth=lw, solid_capstyle="round")
        n += 1
    return n


def _draw_scl_contours(
    ax: plt.Axes,
    scl_mask: np.ndarray,
    label_ids: Sequence[int],
    col_min: int,
    col_max: int,
    color: str = "cyan",
    lw: float = 1.5,
) -> int:
    """Dashed contours for sclerosis labels, clipped to column range [col_min, col_max)."""
    m = scl_mask.astype(np.int32)
    h, w = m.shape
    col_min = max(0, int(col_min))
    col_max = min(w, int(col_max))
    sub = np.zeros((h, w), dtype=float)
    for v in label_ids:
        sub |= (m == int(v)).astype(float)
    sub[:, :col_min] = 0
    sub[:, col_max:] = 0
    if not np.any(sub > 0):
        return 0
    try:
        ax.contour(
            sub,
            levels=[0.5],
            colors=[color],
            linestyles="--",
            linewidths=lw,
            extent=[0, w, h, 0],
            origin="upper",
        )
    except ValueError:
        return 0
    return 1


def _draw_ost_arrows(
    ax: plt.Axes,
    mask: np.ndarray,
    osteophyte_label: int,
    color: str = "gold",
    lw: float = 2.0,
) -> int:
    """One or two arrows toward osteophyte centroid from image edges."""
    c = _centroid_label(mask, osteophyte_label)
    if c is None:
        return 0
    cy, cx = c
    h, w = mask.shape
    patches = []
    for dx, dy in ((-1, 0), (0, -1)):
        sx = w - 1 if dx > 0 else 0
        sy = h - 1 if dy > 0 else 0
        ex, ey = cx + dx * (w * 0.35), cy + dy * (h * 0.35)
        ex = float(np.clip(ex, 2, w - 3))
        ey = float(np.clip(ey, 2, h - 3))
        arr = FancyArrowPatch(
            (ex, ey),
            (cx, cy),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=lw,
            facecolor=color,
            edgecolor=color,
            zorder=5,
        )
        ax.add_patch(arr)
        patches.append(arr)
    return len(patches)


def _panel_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(
        title,
        color="white",
        fontsize=11,
        fontweight="bold",
        loc="center",
        pad=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="black", edgecolor="none"),
    )


def _legend_bar(
    ax: plt.Axes,
    *,
    show_jsn: bool,
    show_ost: bool,
    show_scl: bool,
) -> None:
    handles: List = []
    if show_jsn:
        handles.append(Line2D([0], [0], color="magenta", lw=2.5, label="JSN"))
    if show_ost:
        handles.append(
            Line2D(
                [0],
                [0],
                color="gold",
                lw=0,
                marker=">",
                markersize=10,
                label="OST",
            )
        )
    if show_scl:
        handles.append(
            Line2D([0], [0], color="cyan", lw=2.5, linestyle="--", label="SCL")
        )
    if not handles:
        return
    ax.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=True,
        facecolor="black",
        edgecolor="gray",
        labelcolor="white",
        fontsize=9,
        framealpha=0.92,
    )


def plot_clinical_koa_dashboard(
    ap_image: np.ndarray,
    jsn_result: Dict[str, Any],
    *,
    scl_mask: Optional[np.ndarray] = None,
    scl_label_ids: Optional[Sequence[int]] = None,
    image_ost_r: np.ndarray,
    image_ost_l: np.ndarray,
    mask_ost_r: np.ndarray,
    mask_ost_l: np.ndarray,
    osteophyte_cfg: Dict[str, Any],
    tie_osteophyte_is_higher_id: bool = True,
    osteophyte_label_id: Optional[int] = None,
    study_title: str = "",
    figsize: Tuple[float, float] = (14, 12),
) -> plt.Figure:
    """
    Top: AP bilateral split — left panel patient right, right panel patient left.
    Bottom: patellar / OST views — left panel patient left mask, right panel patient right
    (``osteophyte_ratios_lr_files_auto`` order: mask_left=患者左, mask_right=患者右).
    ``osteophyte_cfg``：含 ``label_mapping``（``Patella`` / ``Patella_Osteophyte``）或回退 ``patella_label_ids``。
    """
    ap = _to_gray01(ap_image)
    h, w = ap.shape[:2]
    mid = max(1, w // 2)

    rr, rl = osteophyte_ratios_lr_files_auto(
        mask_ost_l.astype(np.int32),
        mask_ost_r.astype(np.int32),
        osteophyte_cfg,
        tie_osteophyte_is_higher_id=tie_osteophyte_is_higher_id,
        osteophyte_label_id=osteophyte_label_id,
    )

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_top_r = axes[0, 0]  # top-left: patient Right (image left half)
    ax_top_l = axes[0, 1]  # top-right: patient Left (image right half)
    ax_bot_r = axes[1, 0]  # bottom-left: patient Right patella
    ax_bot_l = axes[1, 1]  # bottom-right: patient Left patella

    # --- Top row AP ---
    for ax, x0, x1, side, title in (
        (ax_top_r, 0, mid, "right", "Right knee"),
        (ax_top_l, mid, w, "left", "Left knee"),
    ):
        ax.imshow(ap, cmap="gray", vmin=0, vmax=1, aspect="equal")
        ax.set_xlim(x0, x1)
        ax.set_ylim(h, 0)
        _draw_jsn_segments(ax, jsn_result, side)
        if scl_mask is not None and scl_label_ids and scl_mask.shape == ap.shape:
            ids = list(scl_label_ids)
            _draw_scl_contours(ax, scl_mask, ids, x0, x1)
        _panel_title(ax, title)
        _legend_bar(ax, show_jsn=True, show_ost=False, show_scl=bool(scl_label_ids))

    # --- Bottom row OST: left panel = Right patella, right panel = Left patella ---
    img_l = _to_gray01(image_ost_l)
    img_r = _to_gray01(image_ost_r)
    ax_bot_r.imshow(img_r, cmap="gray", vmin=0, vmax=1, aspect="equal")
    ax_bot_l.imshow(img_l, cmap="gray", vmin=0, vmax=1, aspect="equal")
    _draw_ost_arrows(ax_bot_r, mask_ost_r.astype(np.int32), rr.osteophyte_label)
    _draw_ost_arrows(ax_bot_l, mask_ost_l.astype(np.int32), rl.osteophyte_label)
    _panel_title(ax_bot_r, "Right knee (patella)")
    _panel_title(ax_bot_l, "Left knee (patella)")
    _legend_bar(ax_bot_r, show_jsn=False, show_ost=True, show_scl=False)
    _legend_bar(ax_bot_l, show_jsn=False, show_ost=True, show_scl=False)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    if study_title:
        fig.suptitle(
            sanitize_plot_text(study_title, fallback="Study"),
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
