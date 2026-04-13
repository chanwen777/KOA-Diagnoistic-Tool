# -*- coding: utf-8 -*-
"""
骨刺（髌骨 / 髌股）测量：数据路径、标签与批处理选项。

**默认数据布局**：同一受试者 **左、右各一张** 影像与标签，命名为 ``{case_id}_L`` / ``{case_id}_R``
（后缀可配置）。

**比例定义**：``label_mapping`` 中 ``Patella``（如 ``[1,2]``）与 ``Patella_Osteophyte``（如 ``1``）；
分母 = 两类 ID 的 **并集** 像素，分子 = ``Patella_Osteophyte`` 像素（与硬化「腔室∪骨赘」一致）。
无上述键时回退 ``patella_label_ids``（两个整数）+ 显式 ``osteophyte_label_id`` 或「少像素=骨刺」
（平局由 ``tie_osteophyte_is_higher_id`` 决定）。命令行 ``--osteophyte-label`` 可覆盖分子所用单一 ID。

测量与可视化均使用 **左、右各一张** volume；**出图仅叠色与框选骨刺类**（非髌骨主体）。

**图保存**：``scripts/osteophyte.py`` 在提供 ``--case-id`` 且未指定 ``--out`` 时，将并排图保存为
``{output_figure_dir}/{pair_id}.{ext}``；若未配置 ``output_figure_dir``，则退化为 ``output_csv`` 所在目录。
"""

from pathlib import Path
from typing import Any, Dict

OSTEOPHYTE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "OSTEOPHYTE": {
        "label_mapping": {
            "background": 0,
            "Patella": [1, 2],
            "Patella_Osteophyte": 1,
        },
        # 可选：无 Patella/Patella_Osteophyte 时回退双类逻辑
        "patella_label_ids": [1, 2],
        "osteophyte_bbox_label_mapping_keys": ["Patella_Osteophyte"],
        "tie_osteophyte_is_higher_id": True,
        "osteophyte_left_suffix": "_L",
        "osteophyte_right_suffix": "_R",
        "image_dir": Path("/path/to/your/osteophyte_image"),
        "mask_dir": Path("/path/to/your/osteophyte_mask"),
        "output_csv": Path("/path/to/your/koa_outputs/osteophyte/osteophyte_results.csv"),
        # 可视化自动命名时的输出目录（--case-id 且未 --out）；不设则用 output_csv 同目录
        "output_figure_dir": Path("/path/to/your/koa_outputs/osteophyte/figures"),
        "meta_data_csv": Path(
            "/path/to/your/osteophyte_image/image_metadata_w_machine_info.csv"
        ),
        "file_type": ".nii.gz",
    },
}

CURRENT_OSTEOPHYTE_CONFIG_KEY = "OSTEOPHYTE"

__all__ = [
    "OSTEOPHYTE_CONFIGS",
    "CURRENT_OSTEOPHYTE_CONFIG_KEY",
]
