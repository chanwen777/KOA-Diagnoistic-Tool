# -*- coding: utf-8 -*-
"""
JSN 狭窄预测 vs 人工标注评估 + 各区室 / 内外侧最优 jsn_narrow_mm（阈值）搜索。
与 ``notebooks/jsn_eval.ipynb`` 逻辑一致。需安装 scikit-learn。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

KNEE_OA_ROOT = Path(__file__).resolve().parents[1]
if str(KNEE_OA_ROOT) not in sys.path:
    sys.path.insert(0, str(KNEE_OA_ROOT))

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


def resolve_mm_column(df: pd.DataFrame, compartment: str) -> Optional[str]:
    """解析毫米列：优先 ``jsn_*``，其次历史 ``jsw_*``，再 ``{compartment}_mm``。"""
    for cand in (
        f"jsn_{compartment}_mm",
        f"jsw_{compartment}_mm",
        f"{compartment}_mm",
    ):
        if cand in df.columns:
            return cand
    return None


def load_labeled_table(
    label_dir: Path,
    prefer_csv_name: Optional[str] = None,
) -> pd.DataFrame:
    """优先 CSV，其次 Excel；默认文件名兼顾 JSN 新名与旧 JSW 表名。"""
    if prefer_csv_name:
        csv_path = label_dir / prefer_csv_name
        if csv_path.exists():
            return pd.read_csv(csv_path, encoding="utf-8")
    csv_path = label_dir / "jwd_result_w_label.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path, encoding="utf-8")
    for xlsx_name in ("jsn_results_w_labels.xlsx", "jsw_results_w_labels.xlsx"):
        xlsx_path = label_dir / xlsx_name
        if xlsx_path.exists():
            return pd.read_excel(xlsx_path)
    raise FileNotFoundError(
        f"未在 {label_dir} 找到 jwd_result_w_label.csv 或 jsn_results_w_labels.xlsx / jsw_results_w_labels.xlsx"
    )


def read_ignore_case_ids(path: Optional[Path]) -> set:
    if path is None or not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")}


def multilabel_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame, labels: List[str]) -> dict:
    out: Dict = {}
    for i, name in enumerate(labels):
        out[name] = {
            "precision": precision_score(y_true.iloc[:, i], y_pred.iloc[:, i], zero_division=0),
            "recall": recall_score(y_true.iloc[:, i], y_pred.iloc[:, i], zero_division=0),
            "f1": f1_score(y_true.iloc[:, i], y_pred.iloc[:, i], zero_division=0),
            "accuracy": accuracy_score(y_true.iloc[:, i], y_pred.iloc[:, i]),
        }
    out["macro"] = {
        "precision": float(np.mean([out[n]["precision"] for n in labels])),
        "recall": float(np.mean([out[n]["recall"] for n in labels])),
        "f1": float(np.mean([out[n]["f1"] for n in labels])),
        "accuracy": float(np.mean([out[n]["accuracy"] for n in labels])),
    }
    y_t = y_true.values.ravel()
    y_p = y_pred.values.ravel()
    out["micro"] = {
        "precision": precision_score(y_t, y_p, zero_division=0),
        "recall": recall_score(y_t, y_p, zero_division=0),
        "f1": f1_score(y_t, y_p, zero_division=0),
        "accuracy": accuracy_score(y_t, y_p),
    }
    out["hamming_loss"] = hamming_loss(y_true, y_pred)
    out["subset_accuracy"] = (np.asarray(y_true) == np.asarray(y_pred)).all(axis=1).mean()
    return out


def find_best_threshold(
    jsn_mm: np.ndarray,
    y_label: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float, dict]:
    jsn_mm = np.asarray(jsn_mm, dtype=float)
    y_label = np.asarray(y_label, dtype=int)
    valid = ~np.isnan(jsn_mm)
    jsn_mm = jsn_mm[valid]
    y_label = y_label[valid]
    if len(jsn_mm) == 0:
        return float("nan"), 0.0, {}
    if thresholds is None:
        th = np.unique(np.percentile(jsn_mm, np.linspace(0, 100, 101)))
        th = np.sort(np.r_[th, np.linspace(jsn_mm.min(), jsn_mm.max(), 200)])
        th = np.unique(th)
    else:
        th = np.asarray(thresholds)
    best_f1 = -1.0
    best_t = float("nan")
    for t in th:
        y_pred_t = (jsn_mm < t).astype(int)
        f1v = f1_score(y_label, y_pred_t, zero_division=0)
        if f1v > best_f1:
            best_f1 = f1v
            best_t = float(t)
    y_pred_best = (jsn_mm < best_t).astype(int)
    metrics_at_best = {
        "precision": precision_score(y_label, y_pred_best, zero_division=0),
        "recall": recall_score(y_label, y_pred_best, zero_division=0),
        "f1": best_f1,
        "accuracy": accuracy_score(y_label, y_pred_best),
    }
    return best_t, float(best_f1), metrics_at_best


def pool_jsn_and_label(
    df: pd.DataFrame,
    compartments: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    jsn_list, lab_list = [], []
    for c in compartments:
        mk = resolve_mm_column(df, c)
        lk = f"{c}_narrow_label"
        if mk is None or lk not in df.columns:
            continue
        jsn_list.append(df[mk].values)
        lab_list.append((df[lk] == 1).fillna(0).astype(int).values)
    if not jsn_list:
        return np.array([]), np.array([])
    return np.concatenate(jsn_list), np.concatenate(lab_list)


def main():
    parser = argparse.ArgumentParser(description="JSN 评估与最优狭窄阈值搜索（对应 jsn_eval.ipynb）")
    parser.add_argument("--label-dir", type=Path, required=True, help="含标注表与预测列的目录")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="写出 CSV 的目录，默认与 --label-dir 相同",
    )
    parser.add_argument(
        "--ignore-cases",
        type=Path,
        default=None,
        help="文本文件：每行一个 case_id，评估时排除",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="指定 CSV 文件名（在 label-dir 下）；不指定则按 jwd_result_w_label.csv → xlsx 顺序",
    )
    args = parser.parse_args()

    label_dir = args.label_dir.resolve()
    out_dir = (args.output_dir or label_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df_eval = load_labeled_table(label_dir, prefer_csv_name=args.input_csv)
    ignore = read_ignore_case_ids(args.ignore_cases)
    case_id_col = "case_id"
    if ignore and case_id_col in df_eval.columns:
        df_eval = df_eval[~df_eval[case_id_col].isin(ignore)].copy()
        print(f"已排除 {len(ignore)} 个 case_id 规则，剩余 {len(df_eval)} 行")

    compartments = ["left_medial", "left_lateral", "right_medial", "right_lateral"]
    pred_cols = [f"{c}_narrow" for c in compartments]
    label_cols = [f"{c}_narrow_label" for c in compartments]
    for pc, lc in zip(pred_cols, label_cols):
        if pc not in df_eval.columns or lc not in df_eval.columns:
            raise KeyError(f"缺少列：需要 {pred_cols} 与 {label_cols}")

    y_pred = df_eval[pred_cols].astype(int)
    y_true = (df_eval[label_cols] == 1).fillna(0).astype(int)

    print(f"评估样本数: {len(df_eval)}")
    metrics = multilabel_metrics(y_true, y_pred, compartments)
    print("\n=== 各区室 ===")
    for name in compartments:
        m = metrics[name]
        print(f"  {name}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}, Acc={m['accuracy']:.4f}")
    print("\n=== Macro / Micro ===")
    print(
        f"  Macro: P={metrics['macro']['precision']:.4f}, R={metrics['macro']['recall']:.4f}, "
        f"F1={metrics['macro']['f1']:.4f}, Acc={metrics['macro']['accuracy']:.4f}"
    )
    print(
        f"  Micro: P={metrics['micro']['precision']:.4f}, R={metrics['micro']['recall']:.4f}, "
        f"F1={metrics['micro']['f1']:.4f}, Acc={metrics['micro']['accuracy']:.4f}"
    )
    print(f"  Subset accuracy: {metrics['subset_accuracy']:.4f}")

    df_metrics = pd.DataFrame([{**{"compartment": c}, **metrics[c]} for c in compartments])
    df_metrics = pd.concat(
        [
            df_metrics,
            pd.DataFrame([{"compartment": "macro", **metrics["macro"]}]),
            pd.DataFrame([{"compartment": "micro", **metrics["micro"]}]),
        ],
        ignore_index=True,
    )
    metrics_path = out_dir / "jsn_evaluation_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False, encoding="utf-8")
    print(f"\n已写入 {metrics_path}")

    print("\n=== 各区室最优 jsn_narrow_mm（由 JSN mm 与 narrow 标签搜索）===")
    best_per_compartment = {}
    for c in compartments:
        mm_col = resolve_mm_column(df_eval, c)
        label_col = f"{c}_narrow_label"
        if mm_col is None or label_col not in df_eval.columns:
            print(f"  {c}: 缺列（mm 或 {label_col}）")
            continue
        y_lab = (df_eval[label_col] == 1).fillna(0).astype(int)
        best_t, best_f1, m = find_best_threshold(df_eval[mm_col].values, y_lab.values)
        best_per_compartment[c] = {"best_jsn_narrow_mm": best_t, "f1": best_f1, **m}
        print(
            f"  {c}: best_jsn_narrow_mm = {best_t:.3f} mm  "
            f"(F1={best_f1:.4f}, P={m['precision']:.4f}, R={m['recall']:.4f}, Acc={m['accuracy']:.4f})"
        )

    df_best_4 = pd.DataFrame(
        [{"compartment": c, **best_per_compartment[c]} for c in compartments if c in best_per_compartment]
    )

    medial_comps = ["left_medial", "right_medial"]
    lateral_comps = ["left_lateral", "right_lateral"]

    jsn_med, lab_med = pool_jsn_and_label(df_eval, medial_comps)
    jsn_lat, lab_lat = pool_jsn_and_label(df_eval, lateral_comps)

    print("\n=== 内侧合并（左内 + 右内）最优 jsn_narrow_mm ===")
    best_t_med, best_f1_med, m_med = find_best_threshold(jsn_med, lab_med)
    print(
        f"  best_jsn_narrow_mm = {best_t_med:.3f} mm  "
        f"(F1={best_f1_med:.4f}, P={m_med['precision']:.4f}, R={m_med['recall']:.4f}, Acc={m_med['accuracy']:.4f})"
    )

    print("\n=== 外侧合并（左外 + 右外）最优 jsn_narrow_mm ===")
    best_t_lat, best_f1_lat, m_lat = find_best_threshold(jsn_lat, lab_lat)
    print(
        f"  best_jsn_narrow_mm = {best_t_lat:.3f} mm  "
        f"(F1={best_f1_lat:.4f}, P={m_lat['precision']:.4f}, R={m_lat['recall']:.4f}, Acc={m_lat['accuracy']:.4f})"
    )

    df_best_med_lat = pd.DataFrame(
        [
            {"group": "medial", "best_jsn_narrow_mm": best_t_med, "f1": best_f1_med, **m_med},
            {"group": "lateral", "best_jsn_narrow_mm": best_t_lat, "f1": best_f1_lat, **m_lat},
        ]
    )

    if not df_best_4.empty:
        df_best_4.to_csv(out_dir / "jsn_best_threshold_per_compartment.csv", index=False, encoding="utf-8")
    df_best_med_lat.to_csv(out_dir / "jsn_best_threshold_medial_lateral.csv", index=False, encoding="utf-8")
    print(f"\n阈值结果已写入 {out_dir}")


if __name__ == "__main__":
    main()
