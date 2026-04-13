"""
NRRD 与坐标系：读取 NRRD 头部、解析 space、获取标准基向量。

职责：
- 读取 NRRD 文件头部（read_nrrd_header）
- 从头部得到 space 名称（get_space_coordinate_system）
- 根据 space 名称得到物理空间标准基向量（get_space_basis_vectors）
- 将 space 名称规范为 LPS/RAS/LAS 供轴检测使用（get_coordinate_system_name）
"""
from pathlib import Path
import numpy as np
import SimpleITK as sitk


def read_nrrd_header(nrrd_path):
    """
    读取 NRRD 文件的头部信息。

    参数:
        nrrd_path: NRRD 文件路径

    返回:
        dict: NRRD 头部信息字典
    """
    if not nrrd_path:
        return {}

    header: dict = {}
    nrrd_path = Path(nrrd_path)

    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(nrrd_path))
        # 只读头部信息，不加载像素数据
        reader.ReadImageInformation()

        # NRRD 头部字段通常会进入 metadata dictionary
        for key in reader.GetMetaDataKeys():
            try:
                header[key] = reader.GetMetaData(key)
            except Exception:
                continue
    except Exception as e:
        print(f"读取NRRD头部失败(SingleITK): {e}")

    return header


def get_space_coordinate_system(nrrd_path):
    """
    从 NRRD 文件头部读取 space 参数，确定坐标系。

    参数:
        nrrd_path: NRRD 文件路径

    返回:
        str: 坐标系名称，如 'left-posterior-superior', 'right-anterior-superior' 等
    """
    header = read_nrrd_header(nrrd_path)
    return header.get("space", None)


def get_space_basis_vectors(space_name):
    """
    根据 space 名称获取标准基向量（用于 get_physical_directions 等）。

    参数:
        space_name: 空间名称，如 'left-posterior-superior', 'right-anterior-superior' 等

    返回:
        dict: 包含 'x'/'y'/'z' 的标准方向向量，未匹配时返回 None
    """
    space_name = space_name.lower() if space_name else ""

    space_definitions = {
        "left-posterior-superior": {
            "x": np.array([-1, 0, 0]),
            "y": np.array([0, -1, 0]),
            "z": np.array([0, 0, 1]),
        },
        "lps": {
            "x": np.array([-1, 0, 0]),
            "y": np.array([0, -1, 0]),
            "z": np.array([0, 0, 1]),
        },
        "right-anterior-superior": {
            "x": np.array([1, 0, 0]),
            "y": np.array([0, 1, 0]),
            "z": np.array([0, 0, 1]),
        },
        "ras": {
            "x": np.array([1, 0, 0]),
            "y": np.array([0, 1, 0]),
            "z": np.array([0, 0, 1]),
        },
        "left-anterior-superior": {
            "x": np.array([-1, 0, 0]),
            "y": np.array([0, 1, 0]),
            "z": np.array([0, 0, 1]),
        },
        "las": {
            "x": np.array([-1, 0, 0]),
            "y": np.array([0, 1, 0]),
            "z": np.array([0, 0, 1]),
        },
    }

    for key, vectors in space_definitions.items():
        if key in space_name:
            return vectors
    return None


def get_coordinate_system_name(nrrd_path):
    """
    从 NRRD 路径解析得到规范化的坐标系名称，供轴检测使用。

    参数:
        nrrd_path: NRRD 文件路径（可为 None，此时返回默认 'LPS'）

    返回:
        str: 'LPS' | 'RAS' | 'LAS'
    """
    from .constants import DEFAULT_COORDINATE_SYSTEM

    if not nrrd_path:
        return DEFAULT_COORDINATE_SYSTEM
    try:
        space_name = get_space_coordinate_system(nrrd_path)
        if not space_name:
            return DEFAULT_COORDINATE_SYSTEM
        space_lower = space_name.lower()
        if "ras" in space_lower or "right-anterior-superior" in space_lower:
            return "RAS"
        if "las" in space_lower or "left-anterior-superior" in space_lower:
            return "LAS"
        if "lps" in space_lower or "left-posterior-superior" in space_lower:
            return "LPS"
    except Exception:
        pass
    return DEFAULT_COORDINATE_SYSTEM
