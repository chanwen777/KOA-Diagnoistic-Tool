# -*- coding: utf-8 -*-
"""
软骨下骨硬化（SCL）：数据路径与 **label_mapping**（类名 → **int** 或 **int 列表**）。

默认命名与 Dataset041 式多类分割一致：plain 腔室键（如 ``Femur_R: [1, 5]``）与 ``Femur_R_Osteosclerosis: 5`` 等；
**比例分母** = 该腔室 **plain 键 ID 与对应硬化键 ID 的并集**（整段股骨/胫骨像素，含 1 与 5），**分子** = 硬化键像素（如 5）。

脚本用 ``sclerosis_label_sets_from_mapping`` 拆四分腔比例；叠图 **仅高亮硬化类**。
``sclerosis_named_bbox_layers_from_label_mapping`` 仅含键名带 ``Osteosclerosis`` 的项，全图连通域画 bbox。

对每个 **解剖腔室**计算「该腔室硬化 ÷ 该腔室全部（非硬化+硬化）」比例（见 ``compute_sclerosis_ratio``）。

**图保存**：``scripts/osteoscierosis.py`` 在 ``--batch-csv-and-figures`` 或单例未指定 ``--out`` 时，
将图写入 ``output_figure_dir``；未配置则用 ``output_csv`` 所在目录。
"""

from pathlib import Path
from typing import Any, Dict

SCLEROSIS_CONFIGS: Dict[str, Dict[str, Any]] = {
    "sclerosis": {
      "label_mapping": {
        "Femur_R": [1,5],
        "Tibia_R": [2,6], 
        "Femur_L": [3,7],
        "Tibia_L": [4,8],
        "Femur_R_Osteosclerosis": 5,
        "Tibia_R_Osteosclerosis": 6, 
        "Femur_L_Osteosclerosis": 7,
        "Tibia_L_Osteosclerosis": 8,
      },   
      "image_dir": Path("/path/to/your/sclerosis_image"),
      "mask_dir": Path("/path/to/your/sclerosis_mask"),
      "output_csv": Path("/path/to/your/koa_outputs/sclerosis/sclerosis_results.csv"),
      "output_figure_dir": Path("/path/to/your/koa_outputs/sclerosis/figures"),
      "meta_data_csv": Path("/path/to/your/sclerosis_image/image_metadata_w_machine_info.csv"),
      "file_type": ".nii.gz",
    },
}

CURRENT_SCLEROSIS_CONFIG_KEY = "sclerosis"

__all__ = [
    "SCLEROSIS_CONFIGS",
    "CURRENT_SCLEROSIS_CONFIG_KEY",
]
