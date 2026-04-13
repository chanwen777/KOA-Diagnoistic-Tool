"""
步骤二：确定股骨下缘与胫骨上缘（关节侧边界）。只输出每侧点集，不做内外侧划分。
内外侧划分由 compartments 模块负责。
支持 edge_method: distance_percentile、axis_projection、morphological。
"""
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from scipy.spatial.distance import cdist

__all__ = ["extract_femur_tibia_edges"]


def extract_femur_tibia_edges(
    mask_2d: np.ndarray,
    spacing: tuple,
    label_mapping: Dict[str, int],
    axis_info: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    对左右两侧分别提取股骨关节侧边界与胫骨关节侧边界（不划分内外侧）。
    返回: {"left": (femur_pts, tibia_pts), "right": (...)}
    点集为 numpy (N, 2) 行/列坐标。内外侧划分请调用 compartments.split_medial_lateral。
    """
    method = config.get("edge_method", "distance_percentile")
    if method == "distance_percentile":
        return _extract_distance_percentile(mask_2d, spacing, label_mapping, axis_info, config)
    if method == "axis_projection":
        return _extract_axis_projection(mask_2d, spacing, label_mapping, axis_info, config)
    if method == "morphological":
        return _extract_morphological(mask_2d, spacing, label_mapping, axis_info, config)
    return _extract_distance_percentile(mask_2d, spacing, label_mapping, axis_info, config)


def _get_side_masks(
    mask_2d: np.ndarray,
    label_mapping: Dict[str, int],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """按标签取左右侧股骨、胫骨二值 mask。返回 {"left": (femur_mask, tibia_mask), "right": (...)}。"""
    sides = {}
    for side in ["left", "right"]:
        suffix = "_L" if side == "left" else "_R"
        femur_key = "Femur" + suffix
        tibia_key = "Tibia" + suffix
        fv = label_mapping.get(femur_key, 0)
        tv = label_mapping.get(tibia_key, 0)
        if fv == 0 or tv == 0:
            continue
        femur_mask = (mask_2d == fv).astype(np.uint8)
        tibia_mask = (mask_2d == tv).astype(np.uint8)
        sides[side] = (femur_mask, tibia_mask)
    return sides


def _contour_points(mask: np.ndarray) -> np.ndarray:
    """返回 mask 外轮廓点 (N, 2) 行、列。"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros((0, 2), dtype=np.float64)
    pts = np.vstack([c.reshape(-1, 2) for c in cnts])  # (N, 2) col, row -> 需统一为 row,col
    # OpenCV 返回 (x, y) = (col, row)，转为 (row, col)
    return np.column_stack([pts[:, 1], pts[:, 0]])


def _extract_distance_percentile(
    mask_2d: np.ndarray,
    spacing: tuple,
    label_mapping: Dict[str, int],
    axis_info: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    轮廓点到对侧骨的最短距离，保留距离最小分位数的点作为关节侧边界。
    """
    percentile = config.get("edge_distance_percentile", 0.15)
    sides = _get_side_masks(mask_2d, label_mapping)
    result = {}
    empty_pts = (np.zeros((0, 2)), np.zeros((0, 2)))

    for side, (femur_mask, tibia_mask) in sides.items():
        femur_contour = _contour_points(femur_mask)
        tibia_contour = _contour_points(tibia_mask)
        if len(femur_contour) == 0 or len(tibia_contour) == 0:
            result[side] = empty_pts
            continue

        # 股骨轮廓点到胫骨轮廓的最小距离（像素）
        dist_femur_to_tibia = cdist(femur_contour, tibia_contour, metric="euclidean")
        min_dist_femur = np.min(dist_femur_to_tibia, axis=1)
        dist_tibia_to_femur = cdist(tibia_contour, femur_contour, metric="euclidean")
        min_dist_tibia = np.min(dist_tibia_to_femur, axis=1)

        thresh_femur = np.percentile(min_dist_femur, percentile * 100)
        thresh_tibia = np.percentile(min_dist_tibia, percentile * 100)
        femur_joint = femur_contour[min_dist_femur <= thresh_femur]
        tibia_joint = tibia_contour[min_dist_tibia <= thresh_tibia]
        result[side] = (femur_joint, tibia_joint)

    return result


def _extract_axis_projection(
    mask_2d: np.ndarray,
    spacing: tuple,
    label_mapping: Dict[str, int],
    axis_info: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    沿纵轴投影，取股骨最下几行、胫骨最上几行的轮廓点。
    """
    axis_si = axis_info.get("axis_si", 0)
    head_is_low = axis_info.get("head_is_low", True)
    sides = _get_side_masks(mask_2d, label_mapping)
    result = {}
    empty_pts = (np.zeros((0, 2)), np.zeros((0, 2)))
    band = config.get("edge_axis_projection_band", 10)

    for side, (femur_mask, tibia_mask) in sides.items():
        femur_contour = _contour_points(femur_mask)
        tibia_contour = _contour_points(tibia_mask)
        if len(femur_contour) == 0 or len(tibia_contour) == 0:
            result[side] = empty_pts
            continue

        femur_val = femur_contour[:, axis_si]
        tibia_val = tibia_contour[:, axis_si]
        if head_is_low:
            femur_edge_val = np.max(femur_val)
            tibia_edge_val = np.min(tibia_val)
        else:
            femur_edge_val = np.min(femur_val)
            tibia_edge_val = np.max(tibia_val)

        femur_joint = femur_contour[np.abs(femur_val - femur_edge_val) <= band]
        tibia_joint = tibia_contour[np.abs(tibia_val - tibia_edge_val) <= band]
        result[side] = (femur_joint, tibia_joint)

    return result


def _extract_morphological(
    mask_2d: np.ndarray,
    spacing: tuple,
    label_mapping: Dict[str, int],
    axis_info: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    形态学下缘/上缘：用 mask - erode 得到 mask 内侧边界（边缘点在 mask 上），再按纵轴保留关节侧。
    """
    axis_si = axis_info.get("axis_si", 0)
    head_is_low = axis_info.get("head_is_low", True)
    sides = _get_side_masks(mask_2d, label_mapping)
    result = {}
    empty_pts = (np.zeros((0, 2)), np.zeros((0, 2)))
    k = 3
    kernel = np.ones((k, k), np.uint8)

    for side, (femur_mask, tibia_mask) in sides.items():
        fem_ero = cv2.erode(femur_mask, kernel)
        tib_ero = cv2.erode(tibia_mask, kernel)
        fem_edge = femur_mask.astype(np.int32) - fem_ero.astype(np.int32)
        tib_edge = tibia_mask.astype(np.int32) - tib_ero.astype(np.int32)
        fem_edge = np.clip(fem_edge, 0, 1).astype(np.uint8)
        tib_edge = np.clip(tib_edge, 0, 1).astype(np.uint8)

        femur_val = np.argwhere(fem_edge > 0)
        tibia_val = np.argwhere(tib_edge > 0)
        if len(femur_val) == 0 or len(tibia_val) == 0:
            result[side] = empty_pts
            continue
        ax_vals_fem = femur_val[:, axis_si]
        ax_vals_tib = tibia_val[:, axis_si]
        if head_is_low:
            fem_keep = ax_vals_fem >= np.percentile(ax_vals_fem, 75)
            tib_keep = ax_vals_tib <= np.percentile(ax_vals_tib, 25)
        else:
            fem_keep = ax_vals_fem <= np.percentile(ax_vals_fem, 25)
            tib_keep = ax_vals_tib >= np.percentile(ax_vals_tib, 75)
        femur_joint = femur_val[fem_keep].astype(np.float64)
        tibia_joint = tibia_val[tib_keep].astype(np.float64)
        result[side] = (femur_joint, tibia_joint)

    return result
