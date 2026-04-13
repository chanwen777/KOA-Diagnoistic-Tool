# -*- coding: utf-8 -*-
"""从 ``mask_dir`` 按 ``file_type`` 列举 case_id / 骨刺 L-R 配对（不依赖 ``meta_data_csv`` 过滤）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

def _normalize_case_id(raw: Any) -> str:
    """
    统一 case_id：去空白；若末尾是 nnU-Net 通道后缀 ``_0000`` 则去掉。
    """
    cid = str(raw).strip()
    if cid.endswith("_0000"):
        cid = cid[: -len("_0000")]
    return cid


def _parse_osteophyte_stem(stem: str) -> Optional[Tuple[str, str, str, str]]:
    """
    严格解析骨刺命名：``<case_id>_<SIDE>_<AGE>_<GENDER>``（可由上游先去 ``_0000``）。
    SIDE 仅支持 L/R，GENDER 仅支持 M/F。
    """
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    side = parts[-3].upper()
    age = parts[-2]
    gender = parts[-1].upper()
    case_id = "_".join(parts[:-3])
    if side not in {"L", "R"} or gender not in {"M", "F"} or not case_id:
        return None
    return case_id, side, age, gender


def resolve_volume_extensions(cfg: Dict[str, Any]) -> List[str]:
    """
    与用户配置对齐：仅使用 ``file_type``（如 ``.nrrd``、``.nii.gz``）。
    未提供时回退 ``.nrrd``。
    """
    file_type = cfg.get("file_type")
    if not file_type:
        return [".nrrd"]
    ext = str(file_type).strip()
    if not ext.startswith("."):
        ext = f".{ext}"
    return [ext]


def list_case_ids_from_config(cfg: Dict[str, Any]) -> List[str]:
    """
    仅从 ``mask_dir`` 按 ``resolve_volume_extensions``（即 ``file_type``）glob，
    stem 以 ``_0000`` 结尾的跳过。不再用 ``meta_data_csv`` 决定病例列表。
    """
    mask_dir = Path(cfg["mask_dir"])

    exts = resolve_volume_extensions(cfg)
    seen: set[str] = set()
    out: List[str] = []
    for ext in exts:
        if not ext.startswith("."):
            ext = f".{ext}"
        for f in sorted(mask_dir.glob(f"*{ext}")):
            if not f.is_file() or not f.name.endswith(ext):
                continue
            stem = f.name[: -len(ext)]
            if stem.endswith("_0000"):
                continue
            if stem not in seen:
                seen.add(stem)
                out.append(stem)
    return sorted(out)


def find_mask_path(
    mask_dir: Path,
    case_id: str,
    extensions: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    """在 ``mask_dir`` 下查找 ``{case_id}{ext}``（mask 不追加 ``_0000``）。"""
    exts: Sequence[str] = extensions if extensions is not None else (".nrrd",)
    mask_dir = Path(mask_dir)
    candidates = [_normalize_case_id(case_id), str(case_id).strip()]
    for cid in candidates:
        if not cid:
            continue
        hit = find_volume_path(mask_dir, cid, exts, allow_channel_suffix=False)
        if hit is not None:
            return hit
    return None


def find_volume_path(
    dir_path: Path,
    stem: str,
    extensions: Sequence[str],
    *,
    allow_channel_suffix: bool = True,
    require_channel_suffix: bool = False,
) -> Optional[Path]:
    """
    在目录下查找体积文件。``extensions`` 如 ``[".nrrd"]`` 或 ``[".nrrd", ".nii.gz"]``。

    - ``require_channel_suffix=True``（常用于 ``image_dir``）：**只**接受 ``{stem}_0000{ext}``，
      不会匹配无 ``_0000`` 的文件。
    - ``require_channel_suffix=False`` 且 ``allow_channel_suffix=True``：先试 ``{stem}{ext}``，
      再试 ``{stem}_0000{ext}``。
    - ``allow_channel_suffix=False``（如 mask）：仅 ``{stem}{ext}``。
    """
    for ext in extensions:
        if require_channel_suffix:
            cand = dir_path / f"{stem}_0000{ext}"
            if cand.is_file():
                return cand
            continue
        names = [f"{stem}{ext}"]
        if allow_channel_suffix:
            names.append(f"{stem}_0000{ext}")
        for name in names:
            cand = dir_path / name
            if cand.is_file():
                return cand
    return None


def list_osteophyte_lr_base_cases(
    mask_dir: Path,
    *,
    left_suffix: str = "_L",
    right_suffix: str = "_R",
    extensions: Sequence[str] = (".nrrd",),
) -> List[str]:
    """
    枚举同时存在 ``{base}{left_suffix}`` 与 ``{base}{right_suffix}`` 的 base case_id。

    例：``KOA01_L.nrrd`` 与 ``KOA01_R.nrrd`` → base ``KOA01``。
    """
    # 兼容旧接口：返回 case_id（不含 side/age/gender）。
    pairs = list_osteophyte_lr_pairs(
        Path(mask_dir),
        left_suffix=left_suffix,
        right_suffix=right_suffix,
        extensions=extensions,
    )
    return sorted({p["case_id"] for p in pairs})


def list_osteophyte_lr_pairs(
    mask_dir: Path,
    *,
    left_suffix: str = "_L",
    right_suffix: str = "_R",
    extensions: Sequence[str] = (".nrrd",),
) -> List[Dict[str, str]]:
    """
    按严格命名 ``<case_id>_<L/R>_<AGE>_<M/F>`` 枚举 L/R 成对样本。
    返回元素含：case_id、age、gender、left_stem、right_stem。
    """
    mask_dir = Path(mask_dir)
    exts = tuple(extensions)
    index: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for ext in exts:
        for f in mask_dir.glob(f"*{ext}"):
            if not f.is_file() or not f.name.endswith(ext):
                continue
            stem = _normalize_case_id(f.name[: -len(ext)])
            parsed = _parse_osteophyte_stem(stem)
            if parsed is None:
                continue
            case_id, side, age, gender = parsed
            key = (case_id, age, gender)
            slot = index.setdefault(key, {})
            slot[side] = stem

    out: List[Dict[str, str]] = []
    for (case_id, age, gender), slot in sorted(index.items()):
        if "L" not in slot or "R" not in slot:
            continue
        out.append(
            {
                "case_id": case_id,
                "age": age,
                "gender": gender,
                "left_stem": slot["L"],
                "right_stem": slot["R"],
                "pair_id": f"{case_id}_{age}_{gender}",
            }
        )
    return out


def list_osteophyte_bases_from_config(cfg: Dict[str, Any]) -> List[str]:
    """
    骨刺 L/R 成对：仅扫描 ``mask_dir`` 下符合严格命名的文件；扩展名由 ``file_type`` 决定。
    """
    mask_dir = Path(cfg["mask_dir"])
    left_s = cfg.get("osteophyte_left_suffix", "_L")
    right_s = cfg.get("osteophyte_right_suffix", "_R")
    exts = tuple(resolve_volume_extensions(cfg))

    pairs = list_osteophyte_lr_pairs(
        mask_dir, left_suffix=left_s, right_suffix=right_s, extensions=exts
    )
    return sorted({p["case_id"] for p in pairs})

def list_osteophyte_lr_pairs_from_config(cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    返回 L/R 成对列表（case_id/age/gender/left_stem/right_stem/pair_id），
    仅从 ``mask_dir`` 扫描，不按 ``meta_data_csv`` 过滤。
    """
    mask_dir = Path(cfg["mask_dir"])
    left_s = cfg.get("osteophyte_left_suffix", "_L")
    right_s = cfg.get("osteophyte_right_suffix", "_R")
    exts = tuple(resolve_volume_extensions(cfg))
    return list_osteophyte_lr_pairs(
        mask_dir, left_suffix=left_s, right_suffix=right_s, extensions=exts
    )
