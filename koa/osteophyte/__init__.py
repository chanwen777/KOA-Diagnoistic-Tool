# -*- coding: utf-8 -*-
"""
骨刺（OST）：从分割 mask 提取比例；核心实现在 ``compute_osteophyte_ratio``。
默认由 ``label_mapping`` 的 ``Patella`` / ``Patella_Osteophyte`` 定义分母与分子；无则回退 ``patella_label_ids``。
"""

from koa.osteophyte.compute_osteophyte_ratio import (
    OsteophyteSideAutoResult,
    osteophyte_bbox_label_ids_from_config,
    osteophyte_label_sets_from_config,
    osteophyte_ratio_full_field,
    osteophyte_ratios_lr_files_auto,
    resolve_osteophyte_label_id,
)

__all__ = [
    "OsteophyteSideAutoResult",
    "osteophyte_bbox_label_ids_from_config",
    "osteophyte_label_sets_from_config",
    "osteophyte_ratio_full_field",
    "osteophyte_ratios_lr_files_auto",
    "resolve_osteophyte_label_id",
]
