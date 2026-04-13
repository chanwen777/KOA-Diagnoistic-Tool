"""
步骤一：确定图像轴与方向。
支持三种来源：mask、nrrd、dicom；返回 axis_info 供 edges 使用。
"""
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from koa.utils.orientation import get_si_axis_with_sign
from koa.utils.sitk_utils import load_sitk_image

__all__ = ["infer_axis", "infer_axis_from_mask", "infer_axis_from_nrrd", "infer_axis_from_dicom"]


def infer_axis(
    config: Dict[str, Any],
    mask_2d: np.ndarray,
    spacing: tuple,
    mask_sitk_or_path: Optional[Union[Any, str, Path]] = None,
    case_id: Optional[str] = None,
    meta_data_csv_path: Optional[Path] = None,
    dicom_path_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    按 config 的 direction_source 分发到 mask / nrrd / dicom 三种实现之一。
    返回 axis_info: axis_si（纵轴 0 或 1）, head_is_low（头侧是否对应较小 index）。
    """
    source = config.get("direction_source", "mask")
    if source == "mask":
        return infer_axis_from_mask(mask_2d, spacing, config)
    if source == "nrrd":
        return infer_axis_from_nrrd(mask_2d, spacing, config, mask_sitk_or_path)
    if source == "dicom":
        return infer_axis_from_dicom(
            mask_2d,
            spacing,
            config,
            case_id=case_id,
            meta_data_csv_path=meta_data_csv_path,
            dicom_path_column=dicom_path_column,
        )
    raise ValueError(f"Unknown direction_source: {source}")


def infer_axis_from_mask(
    mask_2d: np.ndarray,
    spacing: tuple,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    从 mask 几何推断：左右由标签区分，上下由「股骨在胫骨上面」从两骨质心/bbox 推断。
    mask_2d: (H, W) 多标签 mask，标签由 config['label_mapping'] 给出。
    返回 axis_info: axis_si (0 或 1), head_is_low (bool)。
    """
    label_mapping = config.get("label_mapping", {})
    # 需要 Femur_* 和 Tibia_* 的 value
    femur_labels = {k: v for k, v in label_mapping.items() if k.startswith("Femur") and v != 0}
    tibia_labels = {k: v for k, v in label_mapping.items() if k.startswith("Tibia") and v != 0}
    if not femur_labels or not tibia_labels:
        # 无法区分，默认 axis_si=0, head_is_low=True（头在上 = 小 index）
        return {"axis_si": 0, "head_is_low": True}

    # 取所有股骨像素和所有胫骨像素的质心
    femur_mask = np.zeros_like(mask_2d, dtype=bool)
    tibia_mask = np.zeros_like(mask_2d, dtype=bool)
    for v in femur_labels.values():
        femur_mask |= mask_2d == v
    for v in tibia_labels.values():
        tibia_mask |= mask_2d == v

    yx_femur = np.argwhere(femur_mask)  # (N, 2) -> row, col -> axis 0, axis 1
    yx_tibia = np.argwhere(tibia_mask)
    if len(yx_femur) == 0 or len(yx_tibia) == 0:
        return {"axis_si": 0, "head_is_low": True}

    cen_femur = yx_femur.mean(axis=0)  # (2,) axis0 mean, axis1 mean
    cen_tibia = yx_tibia.mean(axis=0)

    # 股骨在胫骨上面：沿纵轴，股骨质心更靠「上」；上 = 头侧。
    # 纵轴：两骨质心连线的主方向，或先试 axis 0 再试 axis 1，看哪个方向分离更明显
    diff = cen_femur - cen_tibia  # 股 - 胫
    if abs(diff[0]) >= abs(diff[1]):
        axis_si = 0  # 行方向为纵轴
        head_is_low = cen_femur[0] < cen_tibia[0]  # 头在上 -> 股骨 row 更小 -> head 对应小 index
    else:
        axis_si = 1  # 列方向为纵轴
        head_is_low = cen_femur[1] < cen_tibia[1]

    return {"axis_si": int(axis_si), "head_is_low": bool(head_is_low)}


def infer_axis_from_nrrd(
    mask_2d: np.ndarray,
    spacing: tuple,
    config: Dict[str, Any],
    mask_sitk_or_path: Optional[Union[Any, str, Path]] = None,
) -> Dict[str, Any]:
    """
    从 NRRD GetDirection() 得到方向矩阵，映射出纵轴及头足侧。
    若 mask_sitk_or_path 为 SITK 图像则直接用；为路径则读取。2D 时从 (1,H,W) 的第三维对应 numpy axis。
    """
    if mask_sitk_or_path is None:
        return infer_axis_from_mask(mask_2d, spacing, config)

    if hasattr(mask_sitk_or_path, "GetDirection"):
        img = mask_sitk_or_path
        nrrd_path = None
    else:
        img = load_sitk_image(mask_sitk_or_path)
        nrrd_path = str(mask_sitk_or_path)

    # get_si_axis_with_sign 返回 numpy_axis（3D 下 0/1/2），superior_is_high_index
    si_info = get_si_axis_with_sign(img, nrrd_path)
    numpy_axis_3d = si_info["numpy_axis"]  # 0=z, 1=y, 2=x
    superior_is_high_index = si_info["superior_is_high_index"]

    # 2D 切片为 (H, W) = 3D 的 (axis1, axis2)，即去掉 axis0
    # 3D shape (1,H,W) -> numpy (1, H, W) -> 2D (H,W) 对应 3D axis 1, 2
    # 所以 2D 的 axis 0 = 3D axis 1, 2D 的 axis 1 = 3D axis 2
    if numpy_axis_3d == 0:
        axis_si = 0  # 3D 的 z 在 2D 被压掉，通常取行作为纵轴
    elif numpy_axis_3d == 1:
        axis_si = 0
    else:
        axis_si = 1

    # head_is_low: 头侧对应较小 index。superior_is_high_index=True 表示 superior 在高 index，所以 head_is_low=False
    head_is_low = not superior_is_high_index

    return {"axis_si": axis_si, "head_is_low": head_is_low}


def infer_axis_from_dicom(
    mask_2d: np.ndarray,
    spacing: tuple,
    config: Dict[str, Any],
    case_id: Optional[str] = None,
    meta_data_csv_path: Optional[Path] = None,
    dicom_path_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    从 RAW_DATA 下 KOA 原始 DICOM 解析方向：通过 meta_data_csv 按 case_id 查 DICOM 路径，用 pydicom 读取。
    """
    try:
        import pandas as pd
        import pydicom
    except ImportError:
        return infer_axis_from_mask(mask_2d, spacing, config)

    path_col = dicom_path_column or config.get("dicom_source_path_column") or "dicom_path"
    csv_path = meta_data_csv_path or config.get("meta_data_csv")
    if not csv_path or not case_id:
        return infer_axis_from_mask(mask_2d, spacing, config)

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return infer_axis_from_mask(mask_2d, spacing, config)

    try:
        df = pd.read_csv(csv_path)
        if path_col not in df.columns:
            return infer_axis_from_mask(mask_2d, spacing, config)
        case_row = df[df[config.get("case_id_column", "case_id")] == case_id]
        if case_row.empty:
            return infer_axis_from_mask(mask_2d, spacing, config)
        dicom_path = Path(case_row[path_col].iloc[0])
        if not dicom_path.exists():
            return infer_axis_from_mask(mask_2d, spacing, config)
        dcm = pydicom.dcmread(str(dicom_path))
    except Exception:
        return infer_axis_from_mask(mask_2d, spacing, config)

    # Image Orientation (Patient) 0020,0037: 行、列方向余弦 (x,y,z) 各两个
    try:
        orient = dcm.get("ImageOrientationPatient", dcm.get(0x0020, 0x0037))
        if orient is None:
            return infer_axis_from_mask(mask_2d, spacing, config)
        # 通常 6 个浮点数: row_x, row_y, row_z, col_x, col_y, col_z
        row_cos = np.array([float(orient[0]), float(orient[1]), float(orient[2])])
        col_cos = np.array([float(orient[3]), float(orient[4]), float(orient[5])])
    except Exception:
        return infer_axis_from_mask(mask_2d, spacing, config)

    # Superior 方向在患者坐标系中为 (0,0,1) 或 (0,0,-1)
    superior_ref = np.array([0.0, 0.0, 1.0])
    # 2D 图像行、列对应 row_cos, col_cos；哪个更接近 superior 方向则哪个是纵轴
    dot_row = np.abs(np.dot(row_cos, superior_ref))
    dot_col = np.abs(np.dot(col_cos, superior_ref))
    if dot_row >= dot_col:
        axis_si = 0  # 行方向更接近 SI
        si_dot = np.dot(row_cos, superior_ref)
    else:
        axis_si = 1
        si_dot = np.dot(col_cos, superior_ref)

    # 索引增加时沿该轴是朝 superior 还是 inferior：si_dot > 0 表示朝 superior
    superior_is_high_index = si_dot > 0
    head_is_low = not superior_is_high_index

    return {"axis_si": axis_si, "head_is_low": head_is_low}
