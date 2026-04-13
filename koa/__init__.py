# -*- coding: utf-8 -*-
"""膝关节 OA 结构征象：关节间隙、硬化、骨刺、汇总看板等。

**关节间隙命名**：**JWD**（joint width distance）指可计算的**数值量**（如间隙宽度 mm）；**JSN**
（joint space narrowing）指基于间隙的**临床/评估语境**（是否变窄等）。历史 API 中部分函数名或
结果列仍含 ``jsn``，其 mm 字段语义上对应 **JWD**。

**推荐**按子包显式导入，依赖边界更清晰::

    from koa.jwd import measure_knee_joint_space
    from koa.osteophyte import osteophyte_ratios_lr_files_auto
    from koa.osteosclerosis import sclerosis_results_dataframe_from_config
    from koa.dashboard import plot_clinical_koa_dashboard

``from koa import <名称>`` 为惰性加载，等价于从对应子包导入该符号。
"""

from __future__ import annotations

import importlib
from typing import Any

_JWD_EXPORTS = (
    "direction",
    "edges",
    "compartments",
    "jsn",
    "split_medial_lateral",
    "extract_femur_tibia_edges",
    "aggregate_jsn_results",
    "compute_jsn_mm",
    "get_min_distance_pair",
    "measure_knee_joint_space",
)
_OSTEOPHYTE_EXPORTS = (
    "OsteophyteSideAutoResult",
    "osteophyte_bbox_label_ids_from_config",
    "osteophyte_label_sets_from_config",
    "osteophyte_ratio_full_field",
    "osteophyte_ratios_lr_files_auto",
    "resolve_osteophyte_label_id",
)
_OSTEOSCLEROSIS_EXPORTS = (
    "SclerosisCompartmentRatios",
    "SclerosisLabelMapping",
    "default_knee_scl_bone_label_sets",
    "sclerosis_bilateral_overlay_figure",
    "sclerosis_case_metrics_row",
    "sclerosis_label_sets_from_mapping",
    "sclerosis_named_bbox_layers_from_label_mapping",
    "sclerosis_ratios_bilateral_vs_bone",
    "sclerosis_results_dataframe_from_config",
)
_DASHBOARD_EXPORTS = (
    "merge_koa_result_csvs",
    "plot_clinical_koa_dashboard",
    "save_figure",
)

_EXPORT_INDEX: dict[str, str] = {}
for _names, _mod in (
    (_JWD_EXPORTS, "koa.jwd"),
    (_OSTEOPHYTE_EXPORTS, "koa.osteophyte"),
    (_OSTEOSCLEROSIS_EXPORTS, "koa.osteosclerosis"),
    (_DASHBOARD_EXPORTS, "koa.dashboard"),
):
    for _n in _names:
        if _n in _EXPORT_INDEX:
            raise RuntimeError(f"duplicate koa top-level export: {_n!r}")
        _EXPORT_INDEX[_n] = _mod

__all__ = tuple(sorted(_EXPORT_INDEX))


def __getattr__(name: str) -> Any:
    mod = _EXPORT_INDEX.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    submodule = importlib.import_module(mod)
    return getattr(submodule, name)


def __dir__() -> list[str]:
    # 避免把 importlib / typing 等实现细节混进 tab 补全
    _skip = frozenset({"importlib", "Any"})
    extra = {k for k in globals() if not k.startswith("_") and k not in _skip}
    return sorted(set(__all__) | extra)
