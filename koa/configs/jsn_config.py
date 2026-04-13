"""
关节间隙专用配置：**JSN** 评估流程；输出的 mm 量为 **JWD**。过窄阈值键为 ``jsn_narrow_mm``（兼容旧键 ``jsw_narrow_mm``）。

``output_figure_dir``：展示图输出目录，默认与 ``output_csv`` 同级下的 ``figures`` 子文件夹（与 notebook / 批处理出图对齐）。
"""
from pathlib import Path
from typing import Any, Dict

JOINT_SPACE_MEASUREMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "JSN": {
        "label_mapping": {
            "background": 0,
            "Femur_R": 1,
            "Tibia_R": 2,
            "Femur_L": 3,
            "Tibia_L": 4,
        },
        "direction_source": "mask",  # "mask" | "nrrd" | "dicom"
        "edge_method": "distance_percentile",  # "distance_percentile" | "axis_projection" | "morphological"
        "edge_distance_percentile": 0.15,
        "distance_method": "min",  # "min" | "percentile" | "mean_in_roi"
        "distance_percentile": 1.0,
        "mean_in_roi_percentile": 1.0,  # distance_method="mean_in_roi" 时取最小 K% 距离的均值
        "jsn_narrow_mm": 5,
        # 中线附近排除（二选一，用于减少中线附近点对 JSN 的影响）
        "exclude_near_midline_ratio": 0.3,  # 如 0.1 表示去掉中间 10%（整个 range 为 base，各边 5%）
        "exclude_near_midline_mm": None,  # 如 2.0 表示排除中线两侧 2mm 内的点，需与 spacing 配合
        "midline_method": "range_center",  # "median" 或 "range_center"（区间中点），划分与排除共用
        "image_dir": Path("/path/to/your/jsn_image"),
        "mask_dir": Path("/path/to/your/jsn_mask"),
        "output_csv": Path("/path/to/your/koa_outputs/jsn/jsn_results.csv"),
        # 与 ``output_csv`` 同目录下的 ``figures``；notebook 批量出图与脚本若需要可统一读此键
        "output_figure_dir": Path("/path/to/your/koa_outputs/jsn/figures"),
        "meta_data_csv": "/path/to/your/jsn_image/image_metadata_w_machine_info.csv",  # direction_source="dicom" 时按 case_id 查原始
        "file_type": ".nrrd",
    },
}

# 当前使用的配置 key；流水线从 map 取：JOINT_SPACE_MEASUREMENT_CONFIGS[CURRENT_CONFIG_KEY]
CURRENT_CONFIG_KEY = "JSN"

__all__ = [
    "JOINT_SPACE_MEASUREMENT_CONFIGS",
    "CURRENT_CONFIG_KEY",
]
