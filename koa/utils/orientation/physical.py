"""
物理方向与映射：图像轴对应的物理方向、物理方向到图像轴的映射、完整方向信息。

职责：
- get_physical_directions: 根据图像 direction 与 NRRD space 得到 x/y/z 轴对应的物理方向名
- map_physical_to_image_axes: 将 lr/ap/si 的腐蚀距离（mm）映射到图像 x/y/z
- get_direction_info: 汇总 direction / spacing / size / origin / 物理方向，便于调试
"""
import numpy as np

from . import nrrd as _nrrd


def get_physical_directions(image, nrrd_path=None):
    """
    根据图像的方向矩阵和（可选）NRRD 的 space 参数，
    识别图像坐标轴 x/y/z 对应的物理方向。

    参数:
        image: SimpleITK.Image 对象
        nrrd_path: NRRD 文件路径（可选）

    返回:
        dict: 'x'/'y'/'z' 物理方向名, 'spacing', 'direction_matrix',
              'similarity', 以及有 nrrd_path 时的 'space'
    """
    direction = np.array(image.GetDirection()).reshape(3, 3)
    spacing = np.array(image.GetSpacing())

    space_name = None
    space_basis = None
    if nrrd_path:
        try:
            space_name = _nrrd.get_space_coordinate_system(nrrd_path)
            if space_name:
                space_basis = _nrrd.get_space_basis_vectors(space_name)
        except Exception:
            pass

    standard_dirs = {
        "Right": np.array([1, 0, 0]),
        "Left": np.array([-1, 0, 0]),
        "Anterior": np.array([0, 1, 0]),
        "Posterior": np.array([0, -1, 0]),
        "Superior": np.array([0, 0, 1]),
        "Inferior": np.array([0, 0, -1]),
    }

    if space_basis:
        x_dir = direction[:, 0]
        y_dir = direction[:, 1]
        z_dir = direction[:, 2]

        def find_best_match_in_space(vec, space_basis, standard_dirs):
            best_name = None
            best_sim = -1
            for axis, basis_vec in space_basis.items():
                sim = np.abs(np.dot(vec, basis_vec))
                if sim > best_sim:
                    best_sim = sim
                    if axis == "x":
                        best_name = "Left" if basis_vec[0] < 0 else "Right"
                    elif axis == "y":
                        best_name = "Posterior" if basis_vec[1] < 0 else "Anterior"
                    elif axis == "z":
                        best_name = "Inferior" if basis_vec[2] < 0 else "Superior"
            return best_name, best_sim

        x_phys, x_sim = find_best_match_in_space(x_dir, space_basis, standard_dirs)
        y_phys, y_sim = find_best_match_in_space(y_dir, space_basis, standard_dirs)
        z_phys, z_sim = find_best_match_in_space(z_dir, space_basis, standard_dirs)
    else:
        x_dir = direction[:, 0]
        y_dir = direction[:, 1]
        z_dir = direction[:, 2]

        def find_best_match(vec, standard_dirs):
            best_name = None
            best_sim = -1
            for name, std_vec in standard_dirs.items():
                sim = np.abs(np.dot(vec, std_vec))
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
            return best_name, best_sim

        x_phys, x_sim = find_best_match(x_dir, standard_dirs)
        y_phys, y_sim = find_best_match(y_dir, standard_dirs)
        z_phys, z_sim = find_best_match(z_dir, standard_dirs)

    result = {
        "x": x_phys,
        "y": y_phys,
        "z": z_phys,
        "spacing": tuple(spacing),
        "direction_matrix": direction,
        "similarity": {"x": x_sim, "y": y_sim, "z": z_sim},
    }
    if space_name:
        result["space"] = space_name
    return result


def map_physical_to_image_axes(physical_erosion_dict, image):
    """
    将物理方向（lr/ap/si）的腐蚀距离（毫米）映射到图像坐标轴（x/y/z）。

    参数:
        physical_erosion_dict: 如 {'lr': 5, 'ap': 3, 'si': 2}
        image: SimpleITK.Image 对象

    返回:
        dict: {'x': ..., 'y': ..., 'z': ...} 图像轴上的腐蚀距离
    """
    phys_dirs = get_physical_directions(image)

    lr_axis = "x" if ("Right" in phys_dirs["x"] or "Left" in phys_dirs["x"]) else None
    ap_axis = "y" if ("Anterior" in phys_dirs["y"] or "Posterior" in phys_dirs["y"]) else None
    si_axis = "z" if ("Superior" in phys_dirs["z"] or "Inferior" in phys_dirs["z"]) else None

    if lr_axis is None and phys_dirs["similarity"]["x"] > 0.9:
        lr_axis = "x"
    elif lr_axis is None and phys_dirs["similarity"]["y"] > 0.9:
        lr_axis = "y"
    elif lr_axis is None and phys_dirs["similarity"]["z"] > 0.9:
        lr_axis = "z"

    if ap_axis is None:
        if "Anterior" in phys_dirs["y"] or "Posterior" in phys_dirs["y"]:
            ap_axis = "y"
        elif "Anterior" in phys_dirs["x"] or "Posterior" in phys_dirs["x"]:
            ap_axis = "x"
        elif "Anterior" in phys_dirs["z"] or "Posterior" in phys_dirs["z"]:
            ap_axis = "z"

    if si_axis is None:
        if "Superior" in phys_dirs["z"] or "Inferior" in phys_dirs["z"]:
            si_axis = "z"
        elif "Superior" in phys_dirs["y"] or "Inferior" in phys_dirs["y"]:
            si_axis = "y"
        elif "Superior" in phys_dirs["x"] or "Inferior" in phys_dirs["x"]:
            si_axis = "x"

    image_erosion = {"x": 0, "y": 0, "z": 0}
    if "lr" in physical_erosion_dict and lr_axis:
        image_erosion[lr_axis] = physical_erosion_dict["lr"]
    if "ap" in physical_erosion_dict and ap_axis:
        image_erosion[ap_axis] = physical_erosion_dict["ap"]
    if "si" in physical_erosion_dict and si_axis:
        image_erosion[si_axis] = physical_erosion_dict["si"]
    return image_erosion


def get_direction_info(image):
    """
    获取图像的完整方向信息（用于调试和显示）。

    参数:
        image: SimpleITK.Image 对象

    返回:
        dict: direction_matrix, spacing_mm, size_voxels, physical_size_mm,
              origin_mm, physical_directions, summary
    """
    direction = np.array(image.GetDirection()).reshape(3, 3)
    spacing = np.array(image.GetSpacing())
    size = np.array(image.GetSize())
    origin = np.array(image.GetOrigin())
    phys_dirs = get_physical_directions(image)
    physical_size = size * spacing

    return {
        "direction_matrix": direction,
        "spacing_mm": spacing,
        "size_voxels": size,
        "physical_size_mm": physical_size,
        "origin_mm": origin,
        "physical_directions": phys_dirs,
        "summary": {
            f"图像X轴 ({phys_dirs['x']})": f"{physical_size[0]:.2f} mm (spacing: {spacing[0]:.3f} mm)",
            f"图像Y轴 ({phys_dirs['y']})": f"{physical_size[1]:.2f} mm (spacing: {spacing[1]:.3f} mm)",
            f"图像Z轴 ({phys_dirs['z']})": f"{physical_size[2]:.2f} mm (spacing: {spacing[2]:.3f} mm)",
        },
    }
