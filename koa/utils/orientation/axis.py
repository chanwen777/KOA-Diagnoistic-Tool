"""
解剖轴检测：根据图像方向矩阵与 NRRD space 判断 AP/SI/LR 轴及「正侧」方向。

- 通用逻辑：_get_axis_with_sign(image_sitk, nrrd_path, axis_name) 使用
  constants 中的参考向量与符号约定，返回 sitk_axis / numpy_axis / positive_is_high_index 等。
- 对外 API：get_ap_axis_with_sign, get_si_axis_with_sign, get_lr_axis_with_sign
  返回键名含 anterior_is_high_index、superior_is_high_index 等。
"""
import numpy as np

from . import nrrd as _nrrd
from .constants import (
    DEFAULT_COORDINATE_SYSTEM,
    SITK_TO_NUMPY_AXIS,
    SPACE_REFERENCE_VECTORS,
)


def _get_axis_with_sign(image_sitk, nrrd_path, axis_name):
    """
    通用「带符号轴」检测：根据 axis_name (ap/si/lr) 与 space 得到参考向量，
    用图像 direction 列向量与参考向量点积找到对应轴，并判断正侧在高/低索引。

    参数:
        image_sitk: SimpleITK.Image
        nrrd_path: NRRD 路径，用于解析坐标系（可为 None，则用 LPS）
        axis_name: 'ap' | 'si' | 'lr'

    返回:
        dict: sitk_axis, numpy_axis, positive_is_high_index, dot_product,
              confidence, coordinate_system
    """
    direction = np.array(image_sitk.GetDirection()).reshape(3, 3)
    coord_system = _nrrd.get_coordinate_system_name(nrrd_path)
    ref_vec, high_if_dot_positive = SPACE_REFERENCE_VECTORS.get(
        coord_system, SPACE_REFERENCE_VECTORS[DEFAULT_COORDINATE_SYSTEM]
    )[axis_name]

    dot_products = np.array(
        [
            np.dot(direction[:, 0], ref_vec),
            np.dot(direction[:, 1], ref_vec),
            np.dot(direction[:, 2], ref_vec),
        ]
    )
    sitk_axis = int(np.argmax(np.abs(dot_products)))
    dot_val = dot_products[sitk_axis]
    positive_is_high_index = (dot_val > 0) == high_if_dot_positive
    numpy_axis = SITK_TO_NUMPY_AXIS[sitk_axis]

    return {
        "sitk_axis": sitk_axis,
        "numpy_axis": numpy_axis,
        "positive_is_high_index": positive_is_high_index,
        "dot_product": float(dot_val),
        "confidence": float(np.abs(dot_val)),
        "coordinate_system": coord_system,
    }


def get_ap_axis_with_sign(image_sitk, nrrd_path=None):
    """
    判断图像中哪根轴对应 AP（前后），以及 Anterior 在哪一侧。

    参数:
        image_sitk: SimpleITK.Image 对象
        nrrd_path: NRRD 文件路径（可选，用于检测 space）

    返回:
        dict: sitk_axis, numpy_axis, anterior_is_high_index, dot_product,
              confidence, coordinate_system
    """
    out = _get_axis_with_sign(image_sitk, nrrd_path, "ap")
    out["anterior_is_high_index"] = out.pop("positive_is_high_index")
    return out


def get_si_axis_with_sign(image_sitk, nrrd_path=None):
    """
    判断图像中哪根轴对应 SI（上下），以及 Superior 在哪一侧。

    参数:
        image_sitk: SimpleITK.Image 对象
        nrrd_path: NRRD 文件路径（可选）

    返回:
        dict: sitk_axis, numpy_axis, superior_is_high_index, dot_product,
              confidence, coordinate_system
    """
    out = _get_axis_with_sign(image_sitk, nrrd_path, "si")
    out["superior_is_high_index"] = out.pop("positive_is_high_index")
    return out


def get_lr_axis_with_sign(image_sitk, nrrd_path=None):
    """
    判断图像中哪根轴对应 LR（左右），以及 Right 在哪一侧。

    参数:
        image_sitk: SimpleITK.Image 对象
        nrrd_path: NRRD 文件路径（可选）

    返回:
        dict: sitk_axis, numpy_axis, right_is_high_index, dot_product,
              confidence, coordinate_system
    """
    out = _get_axis_with_sign(image_sitk, nrrd_path, "lr")
    out["right_is_high_index"] = out.pop("positive_is_high_index")
    return out
