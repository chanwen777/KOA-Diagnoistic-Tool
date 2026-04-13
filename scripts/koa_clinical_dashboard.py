# -*- coding: utf-8 -*-
"""
Aggregate JSN + osteophyte + osteosclerosis for one subject and plot a 2×2 clinical board,
or merge batch CSVs on ``case_id``.

Examples::

    python scripts/koa_clinical_dashboard.py merge \\
        --jsn-csv /path/to/your/koa_outputs/jsn/jsn_results.csv \\
        --ost-csv /path/to/your/koa_outputs/osteophyte/osteophyte_results.csv \\
        --scl-csv /path/to/your/koa_outputs/sclerosis/sclerosis_results.csv \\
        --output /path/to/your/koa_outputs/koa_merged.csv

    python scripts/koa_clinical_dashboard.py plot \\
        --jsn-stem CASE_AP_STEM \\
        --scl-stem CASE_SCL_STEM \\
        --ost-left-stem CASE_L_STEM --ost-right-stem CASE_R_STEM \\
        --out /path/to/your/koa_outputs/board.png --no-show
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import SimpleITK as sitk

KNEE_OA_ROOT = Path(__file__).resolve().parents[1]
if str(KNEE_OA_ROOT) not in sys.path:
    sys.path.insert(0, str(KNEE_OA_ROOT))

from koa.configs.jsn_config import (  # noqa: E402
    CURRENT_CONFIG_KEY as JSN_KEY,
    JOINT_SPACE_MEASUREMENT_CONFIGS,
)
from koa.configs.osteophyte_config import (  # noqa: E402
    CURRENT_OSTEOPHYTE_CONFIG_KEY as OST_KEY,
    OSTEOPHYTE_CONFIGS,
)
from koa.configs.sclerosis_config import (  # noqa: E402
    CURRENT_SCLEROSIS_CONFIG_KEY as SCL_KEY,
    SCLEROSIS_CONFIGS,
)
from koa.dashboard import merge_koa_result_csvs, plot_clinical_koa_dashboard, save_figure  # noqa: E402
from koa.jwd import measure_knee_joint_space  # noqa: E402
from koa.osteophyte import resolve_osteophyte_label_id  # noqa: E402
from koa.osteosclerosis import sclerosis_label_sets_from_mapping  # noqa: E402
from koa.utils.case_list import find_mask_path, find_volume_path, resolve_volume_extensions  # noqa: E402
from koa.utils.sitk_utils import load_sitk_image  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def sitk_to_2d(img: sitk.Image) -> np.ndarray:
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim == 3:
        arr = arr[0]
    return np.asarray(arr)


def _scl_overlay_label_ids(cfg: Dict[str, Any]) -> List[int]:
    lbl = sclerosis_label_sets_from_mapping(cfg["label_mapping"])
    ids: List[int] = []
    for k in (
        "labels_sclerosis_right_femur",
        "labels_sclerosis_right_tibia",
        "labels_sclerosis_left_femur",
        "labels_sclerosis_left_tibia",
    ):
        ids.extend(int(x) for x in lbl[k])
    return sorted(set(ids))


def cmd_merge(args: argparse.Namespace) -> None:
    df = merge_koa_result_csvs(
        Path(args.jsn_csv),
        Path(args.ost_csv),
        Path(args.scl_csv),
        Path(args.output),
        how=args.how,
    )
    logger.info("Merged %d rows -> %s", len(df), args.output)


def cmd_plot(args: argparse.Namespace) -> None:
    jcfg = dict(JOINT_SPACE_MEASUREMENT_CONFIGS[JSN_KEY])
    ocfg = dict(OSTEOPHYTE_CONFIGS[OST_KEY])
    scfg = dict(SCLEROSIS_CONFIGS[SCL_KEY])

    j_ext = tuple(resolve_volume_extensions(jcfg))
    o_ext = tuple(resolve_volume_extensions(ocfg))
    s_ext = tuple(resolve_volume_extensions(scfg))

    j_md = Path(jcfg["mask_dir"])
    j_imd = Path(jcfg.get("image_dir", j_md))
    o_md = Path(ocfg["mask_dir"])
    o_imd = Path(ocfg.get("image_dir", o_md))
    s_md = Path(scfg["mask_dir"])

    jsn_stem = args.jsn_stem
    scl_stem = args.scl_stem
    ost_l = args.ost_left_stem
    ost_r = args.ost_right_stem

    j_mask_p = find_mask_path(j_md, jsn_stem, j_ext)
    j_img_p = find_volume_path(j_imd, jsn_stem, j_ext, require_channel_suffix=True)
    if j_mask_p is None or j_img_p is None:
        raise SystemExit(f"JSN: missing mask or image for stem {jsn_stem!r}")

    s_mask_p = find_mask_path(s_md, scl_stem, s_ext)
    ml_p = find_mask_path(o_md, ost_l, o_ext)
    mr_p = find_mask_path(o_md, ost_r, o_ext)
    il_p = find_volume_path(o_imd, ost_l, o_ext, require_channel_suffix=True)
    ir_p = find_volume_path(o_imd, ost_r, o_ext, require_channel_suffix=True)
    if ml_p is None or mr_p is None or il_p is None or ir_p is None:
        raise SystemExit("Osteophyte: missing one of L/R mask or image paths")

    ap_img = sitk_to_2d(load_sitk_image(str(j_img_p)))

    jsn_result = measure_knee_joint_space(mask_sitk_or_path=str(j_mask_p), config=jcfg)

    scl_mask: np.ndarray | None = None
    scl_ids: List[int] = []
    if s_mask_p is not None and not args.no_scl:
        scl_mask = sitk_to_2d(load_sitk_image(str(s_mask_p))).astype(np.int32)
        scl_ids = _scl_overlay_label_ids(scfg)
        if scl_mask.shape != ap_img.shape:
            logger.warning(
                "SCL mask shape %s != AP image %s; skipping SCL overlay",
                scl_mask.shape,
                ap_img.shape,
            )
            scl_mask = None
            scl_ids = []

    m_l = sitk_to_2d(load_sitk_image(str(ml_p))).astype(np.int32)
    m_r = sitk_to_2d(load_sitk_image(str(mr_p))).astype(np.int32)
    i_l = sitk_to_2d(load_sitk_image(str(il_p)))
    i_r = sitk_to_2d(load_sitk_image(str(ir_p)))

    ost_lab = resolve_osteophyte_label_id(ocfg, cli_override=args.osteophyte_label)
    fig = plot_clinical_koa_dashboard(
        ap_img,
        jsn_result,
        scl_mask=scl_mask,
        scl_label_ids=scl_ids if scl_ids else None,
        image_ost_r=i_r,
        image_ost_l=i_l,
        mask_ost_r=m_r,
        mask_ost_l=m_l,
        osteophyte_cfg=ocfg,
        tie_osteophyte_is_higher_id=ocfg.get("tie_osteophyte_is_higher_id", True),
        osteophyte_label_id=ost_lab,
        study_title=args.title or "",
    )
    out = Path(args.out)
    save_figure(fig, out, dpi=args.dpi)
    logger.info("Saved figure -> %s", out.resolve())
    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="KOA merged CSV + clinical 2×2 dashboard")
    sub = p.add_subparsers(dest="command", required=True)

    pm = sub.add_parser("merge", help="Merge JSN / OST / SCL CSVs on case_id")
    pm.add_argument("--jsn-csv", type=Path, required=True)
    pm.add_argument("--ost-csv", type=Path, required=True)
    pm.add_argument("--scl-csv", type=Path, required=True)
    pm.add_argument("--output", type=Path, required=True)
    pm.add_argument("--how", type=str, default="outer", choices=("outer", "inner", "left", "right"))
    pm.set_defaults(func=cmd_merge)

    pp = sub.add_parser("plot", help="2×2 figure (AP JSN/SCL + patella OST)")
    pp.add_argument("--jsn-stem", type=str, required=True, help="JSN bilateral mask/image stem")
    pp.add_argument("--scl-stem", type=str, required=True, help="Sclerosis mask stem (same shape as AP if overlay)")
    pp.add_argument("--ost-left-stem", type=str, required=True)
    pp.add_argument("--ost-right-stem", type=str, required=True)
    pp.add_argument("--out", type=Path, required=True)
    pp.add_argument("--title", type=str, default="")
    pp.add_argument("--dpi", type=int, default=150)
    pp.add_argument("--no-show", action="store_true")
    pp.add_argument("--no-scl", action="store_true", help="Do not draw sclerosis overlay")
    pp.add_argument(
        "--osteophyte-label",
        type=int,
        default=None,
        help="显式骨刺标签 ID（覆盖骨刺配置中的解析）",
    )
    pp.set_defaults(func=cmd_plot)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
