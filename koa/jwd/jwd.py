"""
单例 **JSN** 评估流水线：读 mask、推断方向、骨缘、内外侧划分；输出的 mm 量为 **JWD**，用于变窄等判定。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import SimpleITK as sitk

from koa.utils.sitk_utils import load_sitk_image

from . import direction
from .compartments import split_medial_lateral
from .edges import extract_femur_tibia_edges
from .jsn import aggregate_jsn_results

__all__ = ["measure_knee_joint_space"]


def measure_knee_joint_space(
    *,
    mask_sitk_or_path: Union[str, Path, sitk.Image],
    config: Dict[str, Any],
    case_id: Optional[str] = None,
    meta_data_csv_path: Optional[Path] = None,
    dicom_path_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    单例 JSN 评估：读 multi-label mask（NRRD 等），取首张切片，跑与 ``notebooks/jsn.ipynb`` 相同的核心步骤（JWD 数值见返回 dict 中 ``*_mm`` 等键）。
    """
    label_mapping = config["label_mapping"]
    if isinstance(mask_sitk_or_path, sitk.Image):
        mask_sitk = mask_sitk_or_path
    else:
        mask_sitk = load_sitk_image(str(mask_sitk_or_path))
    mask_arr = sitk.GetArrayFromImage(mask_sitk)
    mask_2d = mask_arr[0] if mask_arr.ndim == 3 else mask_arr
    spacing_lps = mask_sitk.GetSpacing()
    axis_info = direction.infer_axis(
        config,
        mask_2d,
        spacing_lps,
        mask_sitk_or_path=mask_sitk,
        case_id=case_id,
        meta_data_csv_path=meta_data_csv_path,
        dicom_path_column=dicom_path_column,
    )
    edges_per_side = extract_femur_tibia_edges(
        mask_2d, spacing_lps, label_mapping, axis_info, config
    )
    edges_by_compartment = split_medial_lateral(
        edges_per_side,
        mask_2d.shape[1],
        spacing=tuple(spacing_lps) if spacing_lps else None,
        exclude_near_midline_ratio=config.get("exclude_near_midline_ratio"),
        exclude_near_midline_mm=config.get("exclude_near_midline_mm"),
        midline_method=config.get("midline_method", "median"),
    )
    return aggregate_jsn_results(edges_by_compartment, spacing_lps, config)
