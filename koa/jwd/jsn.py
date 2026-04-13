"""
步骤三：由股骨–胫骨缘点计算 **JWD**（关节间隙宽度，mm）、过窄判定、无效/0 处理；服务于 **JSN** 评估。

临床文献中常用 JSW（joint space width），与本包 **JWD** 同指可测宽度；函数名 ``compute_jsn_mm`` 等为历史命名，
返回值语义为 JWD（mm）。
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

__all__ = ["compute_jsn_mm", "get_min_distance_pair", "aggregate_jsn_results"]


def get_min_distance_pair(
    points_femur: np.ndarray,
    points_tibia: np.ndarray,
    spacing: tuple,
) -> Tuple[float, str, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    找出股骨点集与胫骨点集中距离最近的一对点（用于与画图一致的最小距离展示）。
    点坐标与输入一致，为像素 (row, col)。
    返回 (jwd_mm, status, f_pt, t_pt)；``jwd_mm`` 为关节间隙宽度（JWD，mm）。status 为 "ok" 时 f_pt/t_pt 为 shape (2,) 的数组；
    否则为 (nan, status, None, None)。
    """
    if len(points_femur) == 0 and len(points_tibia) == 0:
        return float("nan"), "empty_side", None, None
    if len(points_femur) == 0:
        return float("nan"), "no_femur_points", None, None
    if len(points_tibia) == 0:
        return float("nan"), "no_tibia_points", None, None

    sp = (float(spacing[0]), float(spacing[1])) if len(spacing) >= 2 else (1.0, 1.0)
    pts_f_mm = points_femur * np.array([sp[0], sp[1]])
    pts_t_mm = points_tibia * np.array([sp[0], sp[1]])
    dist = cdist(pts_f_mm, pts_t_mm, metric="euclidean")
    min_idx = np.unravel_index(np.argmin(dist), dist.shape)
    jwd_mm = float(dist[min_idx])
    f_pt = np.asarray(points_femur[min_idx[0]], dtype=np.float64)
    t_pt = np.asarray(points_tibia[min_idx[1]], dtype=np.float64)
    return jwd_mm, "ok", f_pt, t_pt


def compute_jsn_mm(
    points_femur: np.ndarray,
    points_tibia: np.ndarray,
    spacing: tuple,
    config: Dict[str, Any],
) -> Tuple[float, str]:
    """
    将点集乘 spacing 转 mm 后，按 config 的 distance_method 计算 **JWD**（关节间隙宽度，mm）。
    支持：min（最小距离）、percentile（某分位数）、mean_in_roi（最小 K% 距离的均值）。
    返回 (jwd_mm, status)。status 为 "ok" 表示有效；否则为 "no_femur_points" / "no_tibia_points" / "empty_side"。
    """
    if len(points_femur) == 0 and len(points_tibia) == 0:
        return float("nan"), "empty_side"
    if len(points_femur) == 0:
        return float("nan"), "no_femur_points"
    if len(points_tibia) == 0:
        return float("nan"), "no_tibia_points"

    sp = (float(spacing[0]), float(spacing[1])) if len(spacing) >= 2 else (1.0, 1.0)
    pts_f_mm = points_femur * np.array([sp[0], sp[1]])
    pts_t_mm = points_tibia * np.array([sp[0], sp[1]])

    dist = cdist(pts_f_mm, pts_t_mm, metric="euclidean")
    method = config.get("distance_method", "min")
    if method == "min":
        jwd_mm = float(np.min(dist))
    elif method == "percentile":
        pct = config.get("distance_percentile", 5.0)
        jwd_mm = float(np.percentile(dist.ravel(), pct))
    elif method == "mean_in_roi":
        pct = config.get("mean_in_roi_percentile", 10.0)
        dist_flat = dist.ravel()
        n_roi = max(1, int(len(dist_flat) * (pct / 100.0)))
        roi_distances = np.sort(dist_flat)[:n_roi]
        jwd_mm = float(np.mean(roi_distances))
    else:
        jwd_mm = float(np.min(dist))
    return jwd_mm, "ok"


def _jsn_narrow_threshold_mm(config: Dict[str, Any]) -> float:
    """兼容旧键 jsw_narrow_mm。"""
    if "jsn_narrow_mm" in config:
        return float(config["jsn_narrow_mm"])
    return float(config.get("jsw_narrow_mm", 3.0))


def aggregate_jsn_results(
    edges_result: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    spacing: tuple,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    对 edges 返回的左右侧、内外侧点对分别算 **JWD**（mm），并做过窄（**JSN**）判定、status 汇总。
    返回 dict：键名仍为 ``jsn_*_mm``（历史命名），值为 JWD（mm）；及 ``jsn_*_status``、``jsn_*_narrow``、点对坐标等。
    """
    jsn_narrow_mm = _jsn_narrow_threshold_mm(config)
    distance_method = config.get("distance_method", "min")
    out = {}
    for side in ["left", "right"]:
        if side not in edges_result:
            out[f"jsn_{side}_medial_mm"] = float("nan")
            out[f"jsn_{side}_lateral_mm"] = float("nan")
            out[f"jsn_{side}_medial_status"] = "empty_side"
            out[f"jsn_{side}_lateral_status"] = "empty_side"
            out[f"jsn_{side}_medial_narrow"] = False
            out[f"jsn_{side}_lateral_narrow"] = False
            out[f"jsn_{side}_medial_femur_pt"] = None
            out[f"jsn_{side}_medial_tibia_pt"] = None
            out[f"jsn_{side}_lateral_femur_pt"] = None
            out[f"jsn_{side}_lateral_tibia_pt"] = None
            continue
        for part in ["medial", "lateral"]:
            fem_pts, tib_pts = edges_result[side][part]
            key_mm = f"jsn_{side}_{part}_mm"
            key_status = f"jsn_{side}_{part}_status"
            key_narrow = f"jsn_{side}_{part}_narrow"
            key_femur_pt = f"jsn_{side}_{part}_femur_pt"
            key_tibia_pt = f"jsn_{side}_{part}_tibia_pt"
            if distance_method == "min":
                jwd_mm, status, f_pt, t_pt = get_min_distance_pair(fem_pts, tib_pts, spacing)
                out[key_mm] = jwd_mm
                out[key_status] = status
                out[key_narrow] = status == "ok" and jwd_mm < jsn_narrow_mm
                out[key_femur_pt] = f_pt
                out[key_tibia_pt] = t_pt
            else:
                jwd_mm, status = compute_jsn_mm(fem_pts, tib_pts, spacing, config)
                out[key_mm] = jwd_mm
                out[key_status] = status
                out[key_narrow] = status == "ok" and jwd_mm < jsn_narrow_mm
                out[key_femur_pt] = None
                out[key_tibia_pt] = None
    return out
