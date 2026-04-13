# -*- coding: utf-8 -*-
"""
软骨下骨硬化（SCL）：从分割 mask 提取比例；核心实现在 ``compute_sclerosis_ratio``。
"""

from koa.osteosclerosis.compute_sclerosis_ratio import (
    SclerosisCompartmentRatios,
    SclerosisLabelMapping,
    default_knee_scl_bone_label_sets,
    sclerosis_bilateral_overlay_figure,
    sclerosis_case_metrics_row,
    sclerosis_label_sets_from_mapping,
    sclerosis_named_bbox_layers_from_label_mapping,
    sclerosis_ratios_bilateral_vs_bone,
    sclerosis_results_dataframe_from_config,
)

__all__ = [
    "SclerosisCompartmentRatios",
    "SclerosisLabelMapping",
    "default_knee_scl_bone_label_sets",
    "sclerosis_bilateral_overlay_figure",
    "sclerosis_case_metrics_row",
    "sclerosis_label_sets_from_mapping",
    "sclerosis_named_bbox_layers_from_label_mapping",
    "sclerosis_ratios_bilateral_vs_bone",
    "sclerosis_results_dataframe_from_config",
]
