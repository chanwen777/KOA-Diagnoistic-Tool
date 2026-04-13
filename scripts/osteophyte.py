# -*- coding: utf-8 -*-
"""
骨刺（OST）：左 / 右膝 **各一张** 影像与标签（``case_id_L`` / ``case_id_R``）。
骨刺标签由配置 / 命令行显式指定，或回退为全图内「少像素 = 骨刺」；可 **仅写 CSV**、**单例图** 或
**批量 CSV + 每例图**（``--batch-csv-and-figures``）。出图 **仅** 对每侧判定的骨刺标签叠色与画 bbox（不含髌骨主体）。
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

KNEE_OA_ROOT = Path(__file__).resolve().parents[1]
if str(KNEE_OA_ROOT) not in sys.path:
    sys.path.insert(0, str(KNEE_OA_ROOT))

from koa.configs.osteophyte_config import (  # noqa: E402
    CURRENT_OSTEOPHYTE_CONFIG_KEY,
    OSTEOPHYTE_CONFIGS,
)
from koa.osteophyte import (  # noqa: E402
    OsteophyteSideAutoResult,
    osteophyte_ratios_lr_files_auto,
    resolve_osteophyte_label_id,
)
from koa.utils.bilateral_viz import plot_lr_knee_images_overlay  # noqa: E402
from koa.utils.case_list import (  # noqa: E402
    find_mask_path,
    find_volume_path,
    list_osteophyte_lr_pairs_from_config,
    resolve_volume_extensions,
)
from koa.utils.plot_text import sanitize_plot_text  # noqa: E402
from koa.utils.sitk_utils import load_sitk_image  # noqa: E402

logger = logging.getLogger(__name__)


def _fmt_pct(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return str(x)


def _log_bilateral_pct(
    rr: OsteophyteSideAutoResult,
    rl: OsteophyteSideAutoResult,
    *,
    tag: str = "",
) -> None:
    """右膝 / 左膝骨刺占髌骨区域百分比（与 ``osteophyte_ratios_lr_files_auto`` 返回顺序一致）。"""
    suffix = f" [{tag}]" if tag else ""
    msg = (
        f"OST 骨刺/髌骨区域 %{suffix}: "
        f"右 {_fmt_pct(rr.percentage)}% | 左 {_fmt_pct(rl.percentage)}"
    )
    print(msg)
    logger.info(msg)


def sitk_to_2d(img) -> np.ndarray:
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim == 3:
        arr = arr[0]
    return np.asarray(arr)


def _row_lr(
    case_id: str,
    mask_left: np.ndarray,
    mask_right: np.ndarray,
    cfg: Dict[str, Any],
    *,
    osteophyte_label_id: Optional[int] = None,
) -> Dict[str, Any]:
    rr, rl = osteophyte_ratios_lr_files_auto(
        mask_left,
        mask_right,
        cfg,
        tie_osteophyte_is_higher_id=cfg.get("tie_osteophyte_is_higher_id", True),
        osteophyte_label_id=osteophyte_label_id,
    )
    return {
        "case_id": case_id,
        "right_osteophyte_label": rr.osteophyte_label,
        "right_patella_body_label": rr.patella_label_other,
        "right_osteophyte_pixels": rr.osteophyte_pixels,
        "right_patella_pixels": rr.patella_pixels,
        "right_osteophyte_pct_of_patella": rr.percentage,
        "left_osteophyte_label": rl.osteophyte_label,
        "left_patella_body_label": rl.patella_label_other,
        "left_osteophyte_pixels": rl.osteophyte_pixels,
        "left_patella_pixels": rl.patella_pixels,
        "left_osteophyte_pct_of_patella": rl.percentage,
    }


def run_csv_only(
    cfg: Dict[str, Any],
    *,
    osteophyte_label_id: Optional[int] = None,
) -> pd.DataFrame:
    mask_dir = Path(cfg["mask_dir"])
    exts = tuple(resolve_volume_extensions(cfg))
    rows = []
    for pair in list_osteophyte_lr_pairs_from_config(cfg):
        pl = find_mask_path(mask_dir, pair["left_stem"], exts)
        pr = find_mask_path(mask_dir, pair["right_stem"], exts)
        if pl is None or pr is None:
            continue
        ml = sitk_to_2d(load_sitk_image(str(pl))).astype(np.int32)
        mr = sitk_to_2d(load_sitk_image(str(pr))).astype(np.int32)
        rows.append(
            _row_lr(pair["case_id"], ml, mr, cfg, osteophyte_label_id=osteophyte_label_id)
        )
    return pd.DataFrame(rows)


def _print_csv_bilateral_rows(df: pd.DataFrame) -> None:
    if df.empty:
        return
    for _, row in df.iterrows():
        tag = str(row.get("case_id", ""))
        rp = row.get("right_osteophyte_pct_of_patella")
        lp = row.get("left_osteophyte_pct_of_patella")
        line = f"  [{tag}] 右 {_fmt_pct(rp)}% | 左 {_fmt_pct(lp)}%"
        print(line)
        logger.info(line)


def run_batch_csv_and_figures(
    cfg: Dict[str, Any],
    *,
    osteophyte_label_id: Optional[int],
    figure_ext: str,
    quiet: bool,
) -> None:
    """全部 L/R 配对：写 CSV；有影像则保存并排图。"""
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

    rows: list[Dict[str, Any]] = []
    for pair in list_osteophyte_lr_pairs_from_config(cfg):
        pl = find_mask_path(mask_dir, pair["left_stem"], exts)
        pr = find_mask_path(mask_dir, pair["right_stem"], exts)
        if pl is None or pr is None:
            continue
        ml = sitk_to_2d(load_sitk_image(str(pl))).astype(np.int32)
        mr = sitk_to_2d(load_sitk_image(str(pr))).astype(np.int32)
        rr, rl = osteophyte_ratios_lr_files_auto(
            ml,
            mr,
            cfg,
            tie_osteophyte_is_higher_id=cfg.get("tie_osteophyte_is_higher_id", True),
            osteophyte_label_id=osteophyte_label_id,
        )
        rows.append(
            _row_lr(pair["case_id"], ml, mr, cfg, osteophyte_label_id=osteophyte_label_id)
        )
        pid = str(pair.get("pair_id", pair["case_id"]))
        if not quiet:
            _log_bilateral_pct(rr, rl, tag=pid)

        il_p = find_volume_path(
            imd, pair["left_stem"], exts, require_channel_suffix=True
        )
        ir_p = find_volume_path(
            imd, pair["right_stem"], exts, require_channel_suffix=True
        )
        if il_p is None or ir_p is None:
            warn = f"跳过出图（缺 _0000 影像）: {pid}"
            print(warn)
            logger.warning(warn)
            continue

        image_l = sitk_to_2d(load_sitk_image(str(il_p)))
        image_r = sitk_to_2d(load_sitk_image(str(ir_p)))
        pair_tag = sanitize_plot_text(pid, fallback="subject")
        suptitle_en = f"OST (patella) — right | left — {pair_tag}"
        fig = plot_lr_knee_images_overlay(
            image_l,
            ml,
            rl.percentage,
            image_r,
            mr,
            rr.percentage,
            [],
            overlay_label_values_right=[rr.osteophyte_label],
            overlay_label_values_left=[rl.osteophyte_label],
            suptitle=suptitle_en,
            bbox_label_ids_right=[rr.osteophyte_label],
            bbox_label_ids_left=[rl.osteophyte_label],
            legend_overlay_label="Osteophyte overlay",
        )
        out_path = (out_dir / f"{pair['pair_id']}.{ext}").resolve()
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
        description="骨刺：患者左 / 右各一张影像与标签（base_L / base_R），并排可视化或批量 CSV"
    )
    parser.add_argument(
        "--batch-csv-and-figures",
        action="store_true",
        help="扫描全部 L/R 配对：写 output_csv，并为每例保存并排图（影像须为 *_0000）",
    )
    parser.add_argument("--csv-only", action="store_true", help="批量仅写 CSV（读 config）")
    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="受试者 base id（读取 image/mask 目录下 base_L / base_R）",
    )
    parser.add_argument("--image-left", type=Path, default=None)
    parser.add_argument("--image-right", type=Path, default=None)
    parser.add_argument("--mask-left", type=Path, default=None)
    parser.add_argument("--mask-right", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None, help="覆盖 config 的 image_dir")
    parser.add_argument("--mask-dir", type=Path, default=None, help="覆盖 config 的 mask_dir")
    parser.add_argument("--output-csv", type=Path, default=None, help="覆盖 config 的 output_csv")
    parser.add_argument(
        "--output-figure-dir",
        type=Path,
        default=None,
        help="覆盖 config 的 output_figure_dir（仅 --case-id 且未指定 --out 时）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="双图合一输出路径；省略且 --case-id 时保存为 output_figure_dir（或 output_csv 同目录）下 {pair_id}.{figure-ext}",
    )
    parser.add_argument(
        "--figure-ext",
        type=str,
        default="png",
        help="自动命名时的扩展名（默认 png，与体积 file_type 无关）",
    )
    parser.add_argument(
        "--osteophyte-label",
        type=int,
        default=None,
        help="显式骨刺标签 ID（覆盖配置）；省略则用配置中 osteophyte_label_id 或 label_mapping['Patella_Osteophyte']，再无则自动少像素规则",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="在单例可视化结束后仍写入完整 output_csv（与 --csv-only / --batch 互斥）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="--csv-only 或 --batch 时不逐行打印百分比（仍会打汇总）",
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="logging 级别（默认 INFO，与 print 同时输出关键行）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )

    cfg = dict(OSTEOPHYTE_CONFIGS[CURRENT_OSTEOPHYTE_CONFIG_KEY])
    if args.mask_dir is not None:
        cfg["mask_dir"] = args.mask_dir.resolve()
    if args.image_dir is not None:
        cfg["image_dir"] = args.image_dir.resolve()
    if args.output_csv is not None:
        cfg["output_csv"] = args.output_csv.resolve()
    if args.output_figure_dir is not None:
        cfg["output_figure_dir"] = args.output_figure_dir.resolve()

    ost_lab = resolve_osteophyte_label_id(cfg, cli_override=args.osteophyte_label)

    if args.batch_csv_and_figures and args.csv_only:
        parser.error("--batch-csv-and-figures 与 --csv-only 不要同时使用")
    if args.batch_csv_and_figures and args.case_id:
        parser.error("--batch-csv-and-figures 不要与 --case-id 同时使用")
    if args.batch_csv_and_figures and args.write_csv:
        parser.error("--batch-csv-and-figures 已包含写 CSV，无需再加 --write-csv")

    if args.batch_csv_and_figures:
        run_batch_csv_and_figures(
            cfg,
            osteophyte_label_id=ost_lab,
            figure_ext=args.figure_ext,
            quiet=args.quiet,
        )
        return

    if args.csv_only:
        df = run_csv_only(cfg, osteophyte_label_id=ost_lab)
        out_csv = Path(cfg["output_csv"]).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"已写入 {out_csv}，共 {len(df)} 行")
        if not args.quiet:
            _print_csv_bilateral_rows(df)
        return

    exts = tuple(resolve_volume_extensions(cfg))
    pair: Optional[Dict[str, Any]] = None

    if args.case_id:
        case_id = args.case_id
        md = Path(cfg["mask_dir"])
        imd = Path(cfg["image_dir"])
        match_pairs = [
            p
            for p in list_osteophyte_lr_pairs_from_config(cfg)
            if p["case_id"] == case_id or p["pair_id"] == case_id
        ]
        if not match_pairs:
            parser.error(f"未找到 case_id={case_id} 的 L/R 配对样本")
        if len(match_pairs) > 1:
            parser.error(
                f"case_id={case_id} 命中 {len(match_pairs)} 个配对样本，请改用 --mask-left/right 与 --image-left/right 显式指定"
            )
        pair = match_pairs[0]
        if args.mask_left and args.mask_right:
            ml_p, mr_p = args.mask_left, args.mask_right
        else:
            ml_p = find_mask_path(md, pair["left_stem"], exts)
            mr_p = find_mask_path(md, pair["right_stem"], exts)
        if ml_p is None or mr_p is None:
            parser.error(f"未找到 {pair['left_stem']} / {pair['right_stem']} 的 mask")
        if args.image_left and args.image_right:
            il_p, ir_p = args.image_left, args.image_right
        else:
            il_p = find_volume_path(
                imd, pair["left_stem"], exts, require_channel_suffix=True
            )
            ir_p = find_volume_path(
                imd, pair["right_stem"], exts, require_channel_suffix=True
            )
        if il_p is None or ir_p is None:
            parser.error(f"未找到 {pair['left_stem']} / {pair['right_stem']} 的 image")

        image_l = sitk_to_2d(load_sitk_image(str(il_p)))
        image_r = sitk_to_2d(load_sitk_image(str(ir_p)))
        mask_l = sitk_to_2d(load_sitk_image(str(ml_p))).astype(np.int32)
        mask_r = sitk_to_2d(load_sitk_image(str(mr_p))).astype(np.int32)
    else:
        if not (
            args.image_left
            and args.image_right
            and args.mask_left
            and args.mask_right
        ):
            parser.error(
                "请提供 --case-id（配合 config 路径），或同时提供 "
                "--image-left/right 与 --mask-left/right"
            )
        if args.out is None:
            parser.error("显式 L/R 路径模式必须指定 --out（无 pair_id 可供自动命名）")
        image_l = sitk_to_2d(load_sitk_image(str(args.image_left)))
        image_r = sitk_to_2d(load_sitk_image(str(args.image_right)))
        mask_l = sitk_to_2d(load_sitk_image(str(args.mask_left))).astype(np.int32)
        mask_r = sitk_to_2d(load_sitk_image(str(args.mask_right)))

    rr, rl = osteophyte_ratios_lr_files_auto(
        mask_l,
        mask_r,
        cfg,
        tie_osteophyte_is_higher_id=cfg.get("tie_osteophyte_is_higher_id", True),
        osteophyte_label_id=ost_lab,
    )
    _log_bilateral_pct(
        rr,
        rl,
        tag=str(pair["pair_id"]) if pair is not None else "custom_paths",
    )
    pair_tag = ""
    if pair is not None:
        pair_tag = sanitize_plot_text(
            str(pair.get("pair_id", "")),
            fallback="subject",
        )
    suptitle_en = "OST (patella) — right | left"
    if pair_tag:
        suptitle_en = f"{suptitle_en} — {pair_tag}"
    fig = plot_lr_knee_images_overlay(
        image_l,
        mask_l,
        rl.percentage,
        image_r,
        mask_r,
        rr.percentage,
        [],
        overlay_label_values_right=[rr.osteophyte_label],
        overlay_label_values_left=[rl.osteophyte_label],
        suptitle=suptitle_en,
        bbox_label_ids_right=[rr.osteophyte_label],
        bbox_label_ids_left=[rl.osteophyte_label],
        legend_overlay_label="Osteophyte overlay",
    )

    ext = (args.figure_ext or "png").strip().lstrip(".") or "png"
    if args.out is not None:
        out_path = args.out.resolve()
    else:
        assert pair is not None
        fig_dir = cfg.get("output_figure_dir")
        if fig_dir is not None:
            out_dir = Path(fig_dir).resolve()
        else:
            out_dir = Path(cfg["output_csv"]).resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (out_dir / f"{pair['pair_id']}.{ext}").resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"已保存（双图合一）: {out_path}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

    detail = (
        f"明细: 左 标签={rl.osteophyte_label} 骨刺像素={rl.osteophyte_pixels} "
        f"髌骨总像素={rl.patella_pixels}; "
        f"右 标签={rr.osteophyte_label} 骨刺像素={rr.osteophyte_pixels} "
        f"髌骨总像素={rr.patella_pixels}"
    )
    print(detail)
    logger.info(detail)

    if args.write_csv:
        df = run_csv_only(cfg, osteophyte_label_id=ost_lab)
        out_csv = Path(cfg["output_csv"]).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"已写入完整 CSV: {out_csv}，共 {len(df)} 行")
        logger.info("已写入完整 CSV %s，共 %d 行", out_csv, len(df))
        if not args.quiet:
            _print_csv_bilateral_rows(df)


if __name__ == "__main__":
    main()
