# -*- coding: utf-8 -*-
"""
关节间隙（JSN）批量测量。与 ``notebooks/jsn.ipynb`` 核心流程一致：读 config、遍历 mask、输出 CSV。
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

KNEE_OA_ROOT = Path(__file__).resolve().parents[1]
if str(KNEE_OA_ROOT) not in sys.path:
    sys.path.insert(0, str(KNEE_OA_ROOT))

from koa.configs.jsn_config import (
    CURRENT_CONFIG_KEY,
    JOINT_SPACE_MEASUREMENT_CONFIGS,
)
from koa.jwd import measure_knee_joint_space
from koa.utils.case_list import (
    find_mask_path,
    find_volume_path,
    list_case_ids_from_config,
    resolve_volume_extensions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_mask_and_image_paths(case_id: str, config: dict) -> tuple:
    """返回 (mask_path, image_path)，可能为 None；扩展名由 config ``file_type`` / ``volume_extensions`` 决定。"""
    exts = resolve_volume_extensions(config)
    mask_dir = Path(config["mask_dir"])
    image_dir = Path(config.get("image_dir", mask_dir))
    mask_path = find_mask_path(mask_dir, case_id, exts)
    image_path = find_volume_path(
        image_dir, case_id, exts, require_channel_suffix=True
    )
    return (mask_path, image_path)


def run_pipeline(config: dict) -> pd.DataFrame:
    """遍历 case_id，读 mask、调 measure_knee_joint_space、汇总为 DataFrame。"""
    case_ids = list_case_ids_from_config(config)
    logger.info("共 %d 个 case_id", len(case_ids))
    direction_source = config.get("direction_source", "mask")
    meta_data_csv = config.get("meta_data_csv")
    case_id_col = config.get("case_id_column", "case_id")
    dicom_path_col = config.get("dicom_source_path_column", "dicom_path")
    meta_df = None
    if direction_source == "dicom" and meta_data_csv and Path(meta_data_csv).exists():
        meta_df = pd.read_csv(meta_data_csv)

    rows = []
    for i, case_id in enumerate(case_ids):
        mask_path, image_path = find_mask_and_image_paths(case_id, config)
        if mask_path is None:
            logger.warning("[%s] 未找到 mask，跳过", case_id)
            row = {"case_id": case_id}
            for side in ["left", "right"]:
                for part in ["medial", "lateral"]:
                    row[f"jsn_{side}_{part}_mm"] = float("nan")
                    row[f"jsn_{side}_{part}_status"] = "no_mask"
                    row[f"jsn_{side}_{part}_narrow"] = False
            for compartment in (
                "left_medial",
                "left_lateral",
                "right_medial",
                "right_lateral",
            ):
                side, part = compartment.split("_", 1)
                row[f"{compartment}_mm"] = row[f"jsn_{side}_{part}_mm"]
                row[f"{compartment}_narrow"] = row[f"jsn_{side}_{part}_narrow"]
            rows.append(row)
            continue

        try:
            kwargs = {"mask_sitk_or_path": str(mask_path), "config": config}
            if direction_source == "dicom" and meta_df is not None and case_id_col in meta_df.columns and dicom_path_col in meta_df.columns:
                r = meta_df[meta_df[case_id_col].astype(str) == str(case_id)]
                if not r.empty:
                    kwargs["case_id"] = case_id
                    kwargs["meta_data_csv_path"] = Path(meta_data_csv)
                    kwargs["dicom_path_column"] = dicom_path_col
            result = measure_knee_joint_space(**kwargs)
            row = {"case_id": case_id, **result}
            for compartment in (
                "left_medial",
                "left_lateral",
                "right_medial",
                "right_lateral",
            ):
                side, part = compartment.split("_", 1)
                row[f"{compartment}_mm"] = result.get(f"jsn_{side}_{part}_mm", float("nan"))
                row[f"{compartment}_narrow"] = bool(
                    result.get(f"jsn_{side}_{part}_narrow", False)
                )
            rows.append(row)
        except Exception as e:
            logger.exception("[%s] 测量失败: %s", case_id, e)
            row = {"case_id": case_id}
            for side in ["left", "right"]:
                for part in ["medial", "lateral"]:
                    row[f"jsn_{side}_{part}_mm"] = float("nan")
                    row[f"jsn_{side}_{part}_status"] = "error"
                    row[f"jsn_{side}_{part}_narrow"] = False
            for compartment in (
                "left_medial",
                "left_lateral",
                "right_medial",
                "right_lateral",
            ):
                side, part = compartment.split("_", 1)
                row[f"{compartment}_mm"] = row[f"jsn_{side}_{part}_mm"]
                row[f"{compartment}_narrow"] = row[f"jsn_{side}_{part}_narrow"]
            rows.append(row)

        if (i + 1) % 50 == 0:
            logger.info("已处理 %d / %d", i + 1, len(case_ids))

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="关节间隙 JSN 批量测量（对应 jsn.ipynb）")
    parser.add_argument("--output", type=str, default=None, help="输出 CSV；默认用 config 内 output_csv")
    args = parser.parse_args()

    config = dict(JOINT_SPACE_MEASUREMENT_CONFIGS[CURRENT_CONFIG_KEY])
    df = run_pipeline(config)
    out_path = Path(args.output) if args.output else Path(config["output_csv"])
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_path), index=False)
    logger.info("结果已写入 %s，共 %d 行", out_path, len(df))


if __name__ == "__main__":
    main()
