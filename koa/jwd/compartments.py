"""
内外侧划分：将每侧股骨/胫骨关节侧点按局部中线分为内侧(medial)与外侧(lateral)。
与边缘提取(edges)分离，edges 只负责得到轮廓/边界点，本模块负责分内外侧。

中线有两种取法（由 midline_method 统一控制，划分与排除一致）：
  - "median": 点集列坐标中位数。
  - "range_center": 区间中点 (min+max)/2，按空间对半分。
"""
from typing import Dict, Literal, Optional, Tuple

import numpy as np

__all__ = ["split_medial_lateral"]

MidlineMethod = Literal["median", "range_center"]


def _mid_col_from_cols(cols: np.ndarray, method: MidlineMethod) -> float:
    """根据 method 从列坐标计算中线：median 或 range_center=(min+max)/2。"""
    if method == "range_center":
        return float(cols.min() + cols.max()) / 2.0
    return float(np.median(cols))


def _split_pts_by_mid_col(
    femur_pts: np.ndarray,
    tibia_pts: np.ndarray,
    side: str,
    image_center_col: float,
    midline_method: MidlineMethod = "median",
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    以该侧关节附近局部中线为界，将点集分为内侧/外侧。中线取法由 midline_method 决定：median 或 range_center。
    观者视角下：图左=患者右腿、图右=患者左腿；内侧=靠体中线，外侧=远离体中线。
    左膝（在图右）：col 小=内侧，col 大=外侧；右膝（在图左）：col 大=内侧，col 小=外侧。
    点坐标为 (row, col)，col 小=图左，col 大=图右。
    返回 (fem_med, tib_med), (fem_lat, tib_lat)。
    """
    if len(femur_pts) == 0 or len(tibia_pts) == 0:
        empty = (np.zeros((0, 2)), np.zeros((0, 2)))
        return empty, empty

    all_pts = np.vstack([femur_pts, tibia_pts])
    mid_col = _mid_col_from_cols(all_pts[:, 1], midline_method)

    def split_pts(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        left_pts = pts[pts[:, 1] < mid_col]
        right_pts = pts[pts[:, 1] >= mid_col]
        # 以观者视角：图左=患者右腿，图右=患者左腿。内侧=靠体中线，外侧=远离体中线。
        if side == "left":
            medial_pts = left_pts   # 左膝在图右侧，内侧靠体中线=图左侧
            lateral_pts = right_pts
        else:
            medial_pts = right_pts  # 右膝在图左侧，内侧靠体中线=图右侧
            lateral_pts = left_pts
        return medial_pts, lateral_pts

    fem_med, fem_lat = split_pts(femur_pts)
    tib_med, tib_lat = split_pts(tibia_pts)
    return (fem_med, tib_med), (fem_lat, tib_lat)


def split_medial_lateral(
    edges_result_per_side: Dict[str, Tuple[np.ndarray, np.ndarray]],
    image_width: float,
    spacing: Optional[Tuple[float, ...]] = None,
    exclude_near_midline_ratio: Optional[float] = None,
    exclude_near_midline_mm: Optional[float] = None,
    midline_method: MidlineMethod = "median",
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    对 edges 模块返回的「每侧 (股骨点, 胫骨点)」做内外侧划分。

    edges_result_per_side: {"left": (femur_pts, tibia_pts), "right": (...)}
    image_width: mask 宽度 (列数)，用于 image_center_col = image_width / 2。
    spacing: 像素间距 (row, col)，用于 exclude_near_midline_mm 时转 mm。
    exclude_near_midline_ratio: 可选；去掉中间该比例（整个 range 为 base），如 0.1 表示中间 10%、各边 5%。
    exclude_near_midline_mm: 可选；以 mm 为单位排除中线附近的点。需同时传 spacing。
    midline_method: "median"（中位数）或 "range_center"（区间中点 (min+max)/2），划分与排除共用同一中线定义。
    返回: {"left": {"medial": (fem_pts, tib_pts), "lateral": (...)}, "right": {...}}
    """
    image_center_col = image_width / 2.0
    result = {}
    for side in ["left", "right"]:
        if side not in edges_result_per_side:
            result[side] = {
                "medial": (np.zeros((0, 2)), np.zeros((0, 2))),
                "lateral": (np.zeros((0, 2)), np.zeros((0, 2))),
            }
            continue
        femur_pts, tibia_pts = edges_result_per_side[side]
        (fem_med, tib_med), (fem_lat, tib_lat) = _split_pts_by_mid_col(
            femur_pts, tibia_pts, side, image_center_col, midline_method=midline_method
        )
        # 可选：排除中线附近的点（与划分使用同一 midline_method）
        if exclude_near_midline_ratio is not None or exclude_near_midline_mm is not None:
            all_pts_side = np.vstack(
                [p for p in [fem_med, tib_med, fem_lat, tib_lat] if len(p) > 0]
            )
            if len(all_pts_side) > 0:
                cols = all_pts_side[:, 1]
                mid_col = _mid_col_from_cols(cols, midline_method)
                range_col = float(np.ptp(cols))
                if range_col < 1e-6:
                    range_col = 1.0
                if (
                    exclude_near_midline_mm is not None
                    and spacing is not None
                    and len(spacing) >= 2
                ):
                    threshold_px = exclude_near_midline_mm / float(spacing[1])
                elif exclude_near_midline_ratio is not None:
                    # 去掉中间 exclude_near_midline_ratio 比例的点，中线两侧各占一半（如 0.1 → 中间 10%，各边 5%）
                    threshold_px = range_col * (exclude_near_midline_ratio / 2.0)
                else:
                    threshold_px = 0.0
                if threshold_px > 0:

                    def _filter_far_from_mid(pts: np.ndarray) -> np.ndarray:
                        if len(pts) == 0:
                            return pts
                        keep = np.abs(pts[:, 1] - mid_col) > threshold_px
                        return pts[keep]

                    fem_med = _filter_far_from_mid(fem_med)
                    tib_med = _filter_far_from_mid(tib_med)
                    fem_lat = _filter_far_from_mid(fem_lat)
                    tib_lat = _filter_far_from_mid(tib_lat)
        result[side] = {"medial": (fem_med, tib_med), "lateral": (fem_lat, tib_lat)}
    return result
