"""
``koa.jwd``：**JSN**（关节间隙变窄）评估——方向、骨缘、内外侧划分、最短距离与过窄判定。

**命名**：可计算的间隙宽度等**数值量**称 **JWD**（joint width distance，常用单位 mm）；整条链路回答的是
**JSN** 语境（是否变窄等）。历史函数名与结果列仍多含 ``jsn``，其中 ``*_mm`` 字段语义为 **JWD**。
文献亦常见 JSW（joint space width），与本包 JWD 同指可测宽度。
"""

from __future__ import annotations

from . import compartments, direction, edges, jsn
from .compartments import split_medial_lateral
from .edges import extract_femur_tibia_edges
from .jsn import aggregate_jsn_results, compute_jsn_mm, get_min_distance_pair
from .jwd import measure_knee_joint_space

__all__ = [
    "direction",
    "edges",
    "compartments",
    "jsn",
    "split_medial_lateral",
    "extract_femur_tibia_edges",
    "aggregate_jsn_results",
    "compute_jsn_mm",
    "get_min_distance_pair",
    "measure_knee_joint_space",
]
