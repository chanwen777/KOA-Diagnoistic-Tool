# -*- coding: utf-8 -*-
"""
双膝软骨下骨硬化（SCL）：单例叠图，或按配置 **仅写出 CSV**，或 **批量 CSV + 每例图**
（``--batch-csv-and-figures``，对齐 ``scripts/osteophyte.py``）。

CSV 中每例包含：**右股骨、右胫骨、左股骨、左胫骨** 四分腔硬化比（硬化像素、分母为腔室 plain∪硬化总像素，列名仍含 ``*_bone_pixels``），标签见 ``label_mapping``。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

KNEE_OA_ROOT = Path(__file__).resolve().parents[1]
if str(KNEE_OA_ROOT) not in sys.path:
    sys.path.insert(0, str(KNEE_OA_ROOT))

from koa.configs.sclerosis_config import (  # noqa: E402
    CURRENT_SCLEROSIS_CONFIG_KEY,
    SCLEROSIS_CONFIGS,
)
from koa.osteosclerosis import (  # noqa: E402
    SclerosisCompartmentRatios,
    SclerosisLabelMapping,
    sclerosis_bilateral_overlay_figure,
    sclerosis_case_metrics_row,
    sclerosis_results_dataframe_from_config,
)
from koa.utils.case_list import (  # noqa: E402
    find_mask_path,
    find_volume_path,
    list_case_ids_from_config,
    resolve_volume_extensions,
)
from koa.utils.plot_text import sanitize_plot_text  # noqa: E402
from koa.utils.sitk_utils import load_sitk_image  # noqa: E402

logger = logging.getLogger(__name__)


def sitk_to_2d(img) -> np.ndarray:
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim == 3:
        arr = arr[0]
    return np.asarray(arr)


def run_csv_only(cfg: Dict[str, Any]) -> pd.DataFrame:
    return sclerosis_results_dataframe_from_config(cfg)


def run_batch_csv_and_figures(
    cfg: Dict[str, Any],
    *,
    figure_ext: str,
    quiet: bool,
) -> None:
    """全部病例：写 CSV；有 ``{case}_0000`` 影像则保存双膝叠图。"""
    mask_dir = Path(cfg["mask_dir"])
    imd = Path(cfg["image_dir"])
    exts = tuple(resolve_volume_extensions(cfg))
    fig_dir = cfg.get("output_figure_dir")
    if fig_dir is not None:
        out_dir = Path(fig_dir).resolve()
    else:
        out_dir = Path(cfg["output_csv"]).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = (figure_ext or "png").strip().lstrip(".") or "png"
    lm = cfg["label_mapping"]

    rows: list[Dict[str, Any]] = []
    for cid in list_case_ids_from_config(cfg):
        mp = find_mask_path(mask_dir, cid, exts)
        if mp is None:
            continue
        mask_2d = sitk_to_2d(load_sitk_image(str(mp))).astype(np.int32)
        rows.append(sclerosis_case_metrics_row(cid, mask_2d, lm))

        ip = find_volume_path(imd, cid, exts, require_channel_suffix=True)
        if ip is None:
            warn = f"跳过出图（缺 _0000 影像）: {cid}"
            print(warn)
            logger.warning(warn)
            continue

        image_2d = sitk_to_2d(load_sitk_image(str(ip)))
        fig, comp = sclerosis_bilateral_overlay_figure(image_2d, mask_2d, lm)
        if not quiet:
            print(
                f"  [{cid}] 右股骨% {comp.pct_right_femur} 右胫骨% {comp.pct_right_tibia} | "
                f"左股骨% {comp.pct_left_femur} 左胫骨% {comp.pct_left_tibia}"
            )
            logger.info(
                "[%s] RF%% %s RT%% %s LF%% %s LT%% %s",
                cid,
                comp.pct_right_femur,
                comp.pct_right_tibia,
                comp.pct_left_femur,
                comp.pct_left_tibia,
            )

        tag = sanitize_plot_text(cid, fallback="case")
        out_path = (out_dir / f"{tag}.{ext}").resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        done = f"已保存图: {out_path}"
        print(done)
        logger.info(done)

    df = pd.DataFrame(rows)
    out_csv = Path(cfg["output_csv"]).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    summary = f"已写入 CSV: {out_csv}，共 {len(df)} 行"
    print(summary)
    logger.info(summary)


def main():
    parser = argparse.ArgumentParser(
        description="硬化占比：单例叠图、--csv-only 批量 CSV，或 --batch-csv-and-figures 批量 CSV+图"
    )
    parser.add_argument(
        "--batch-csv-and-figures",
        action="store_true",
        help="扫描 mask_dir 全部病例：写 output_csv，并为每例保存叠图（影像须为 {case}_0000）",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="仅根据 koa.configs.sclerosis_config 批量写 CSV",
    )
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None, help="覆盖 config 的 image_dir")
    parser.add_argument("--mask-dir", type=Path, default=None, help="覆盖 config 的 mask_dir")
    parser.add_argument("--output-csv", type=Path, default=None, help="覆盖 config 的 output_csv")
    parser.add_argument(
        "--output-figure-dir",
        type=Path,
        default=None,
        help="覆盖 config 的 output_figure_dir（批量出图或单例自动命名目录）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="单例输出图像路径；省略时若配置了 output_figure_dir 则写入其下 {mask_stem}.figure-ext",
    )
    parser.add_argument(
        "--figure-ext",
        type=str,
        default="png",
        help="自动命名时的扩展名（默认 png，与体积 file_type 无关）",
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="--batch-csv-and-figures 时不逐例打印四个腔室百分比",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="logging 级别（默认 INFO）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )

    cfg = dict(SCLEROSIS_CONFIGS[CURRENT_SCLEROSIS_CONFIG_KEY])
    if args.mask_dir is not None:
        cfg["mask_dir"] = args.mask_dir.resolve()
    if args.image_dir is not None:
        cfg["image_dir"] = args.image_dir.resolve()
    if args.output_csv is not None:
        cfg["output_csv"] = args.output_csv.resolve()
    if args.output_figure_dir is not None:
        cfg["output_figure_dir"] = args.output_figure_dir.resolve()

    if args.batch_csv_and_figures and args.csv_only:
        parser.error("--batch-csv-and-figures 与 --csv-only 不要同时使用")

    if args.batch_csv_and_figures:
        run_batch_csv_and_figures(
            cfg,
            figure_ext=args.figure_ext,
            quiet=args.quiet,
        )
        return

    if args.csv_only:
        df = run_csv_only(cfg)
        out_csv = Path(cfg["output_csv"]).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"已写入 {out_csv}，共 {len(df)} 行")
        return

    if args.image is None or args.mask is None:
        parser.error(
            "单例模式需要 --image 与 --mask，或改用 --csv-only / --batch-csv-and-figures"
        )

    img = load_sitk_image(str(args.image))
    msk = load_sitk_image(str(args.mask))
    image_2d = sitk_to_2d(img)
    mask_2d = sitk_to_2d(msk).astype(np.int32)

    fig, r = sclerosis_bilateral_overlay_figure(image_2d, mask_2d, cfg["label_mapping"])
    print("右股骨硬化%", r.pct_right_femur, "右胫骨硬化%", r.pct_right_tibia)
    print("左股骨硬化%", r.pct_left_femur, "左胫骨硬化%", r.pct_left_tibia)

    ext = (args.figure_ext or "png").strip().lstrip(".") or "png"
    if args.out is not None:
        out_path = args.out.resolve()
    elif cfg.get("output_figure_dir") is not None:
        out_dir = Path(cfg["output_figure_dir"]).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        stem_path = Path(args.mask.name)
        while stem_path.suffix:
            stem_path = stem_path.with_suffix("")
        tag = sanitize_plot_text(stem_path.name, fallback="case")
        out_path = (out_dir / f"{tag}.{ext}").resolve()
    else:
        out_path = None

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"已保存图像: {out_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
