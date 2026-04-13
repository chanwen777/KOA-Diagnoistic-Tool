"""
方向子包 (orientation)：医学影像坐标系与解剖轴检测。

结构：
- constants: 轴索引映射 (SITK<->NumPy)、各坐标系下 AP/SI/LR 参考向量
- nrrd:      NRRD 头部与 space 解析、坐标系名称
- axis:      解剖轴检测（get_ap/si/lr_axis_with_sign）
- physical:  物理方向映射、物理->图像轴映射、完整方向信息
"""
from .axis import (
    get_ap_axis_with_sign,
    get_lr_axis_with_sign,
    get_si_axis_with_sign,
)
from .constants import (
    NUMPY_TO_SITK_AXIS,
    SITK_TO_NUMPY_AXIS,
)
from .nrrd import (
    get_coordinate_system_name,
    get_space_basis_vectors,
    get_space_coordinate_system,
    read_nrrd_header,
)
from .physical import (
    get_direction_info,
    get_physical_directions,
    map_physical_to_image_axes,
)

__all__ = [
    "read_nrrd_header",
    "get_space_coordinate_system",
    "get_space_basis_vectors",
    "get_coordinate_system_name",
    "get_physical_directions",
    "map_physical_to_image_axes",
    "get_direction_info",
    "SITK_TO_NUMPY_AXIS",
    "NUMPY_TO_SITK_AXIS",
    "get_ap_axis_with_sign",
    "get_si_axis_with_sign",
    "get_lr_axis_with_sign",
]
