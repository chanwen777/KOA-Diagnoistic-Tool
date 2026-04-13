"""
方向子包常量：轴索引映射、坐标系与解剖轴参考向量。

- 轴映射：SimpleITK (x,y,z) 与 NumPy (z,y,x) 的对应关系。
- 参考向量：各坐标系下 AP/SI/LR 的物理参考向量及「正侧」与点积符号的约定，
  供 axis 模块统一计算「某轴对应哪根图像轴、正侧在高索引还是低索引」。
"""
import numpy as np

# -----------------------------------------------------------------------------
# SimpleITK 与 NumPy 轴索引映射
# -----------------------------------------------------------------------------
# sitk (x, y, z) -> numpy (z, y, x)
SITK_TO_NUMPY_AXIS = {0: 2, 1: 1, 2: 0}
NUMPY_TO_SITK_AXIS = {2: 0, 1: 1, 0: 2}

# -----------------------------------------------------------------------------
# 坐标系 → 解剖轴参考向量与符号约定
# -----------------------------------------------------------------------------
# 每个解剖轴 (ap, si, lr) 的配置：
#   (reference_vector, positive_side_high_if_dot_positive)
# - reference_vector: 该轴在物理坐标系下的参考方向（与图像 direction 列向量点积）
# - positive_side_high_if_dot_positive: 若点积 > 0（索引增加朝参考方向），
#   则「正侧」是否在高索引一侧。用于算 anterior_is_high_index 等。
#
# AP: 参考方向为 Posterior；dot>0 表示朝 Posterior → Anterior 在低索引 → False
# SI: 参考方向为 Superior；dot>0 表示朝 Superior → Superior 在高索引 → True
# LR: 参考方向为 Right；dot>0 表示朝 Right → Right 在高索引 → True
#
SPACE_REFERENCE_VECTORS = {
    "LPS": {
        "ap": (np.array([0.0, 1.0, 0.0], dtype=float), False),   # Posterior
        "si": (np.array([0.0, 0.0, 1.0], dtype=float), True),    # Superior
        "lr": (np.array([-1.0, 0.0, 0.0], dtype=float), True),   # Right (Left=+X)
    },
    "RAS": {
        "ap": (np.array([0.0, -1.0, 0.0], dtype=float), False),   # Posterior = -Y
        "si": (np.array([0.0, 0.0, 1.0], dtype=float), True),
        "lr": (np.array([1.0, 0.0, 0.0], dtype=float), True),    # Right = +X
    },
    "LAS": {
        "ap": (np.array([0.0, -1.0, 0.0], dtype=float), False),
        "si": (np.array([0.0, 0.0, 1.0], dtype=float), True),
        "lr": (np.array([-1.0, 0.0, 0.0], dtype=float), True),
    },
}

DEFAULT_COORDINATE_SYSTEM = "LPS"
