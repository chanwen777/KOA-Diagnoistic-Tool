<p align="right">
<a href="README.zh-CN.md"><img src="https://img.shields.io/badge/lang-简体中文-555555?style=flat-square" alt="Simplified Chinese README"></a>
</p>

# KOA (Knee OA structural analysis)

This repository provides the Python package **`koa`** for extracting **structural** knee osteoarthritis (OA) measures from images and segmentations. There are three main families:

- **Joint space**: measurable gap widths are **JWD** (joint width distance, typically in mm); the **evaluation framing** (e.g. narrowing) is **JSN**. Literature often uses JSW; in this repo it is the same idea as JWD. Legacy scripts/columns may still say ``jsn`` while ``*_mm`` means JWD.
- **Subchondral sclerosis (SCL)**: ratios from sclerosis segmentations.
- **Osteophytes (OST)**: ratios from osteophyte-related segmentations.

Together these align with common K–L imaging elements (narrowing, osteophytes, sclerosis). **Combined K–L grading / automated diagnosis is not implemented here.**

If you use **nnU-Net** for training and inference, keep those scripts in a **separate pipeline repository**; this repo covers **post-segmentation** measurement and evaluation only.

---

## Environment and dependencies

**Note:** Some notebooks were saved with a kernel named **`image_analysis_env`** and Python **3.11**. That name is optional; any venv/conda env is fine.

**Recommended (Conda):**

```bash
conda create -n koa python=3.11 -y
conda activate koa

# Replace with your local clone path
cd /path/to/your/KOA-Diagnoistic-Tool
pip install -r requirements.txt
```

**pip only:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

Then set **`PYTHONPATH`** as below.

---

## Package layout (`koa/`)

| Module | Role |
|--------|------|
| **`jwd`** | **JSN** assessment from 2D femur–tibia label maps; mm outputs are **JWD** (`measure_knee_joint_space`, orientation, edges, medial/lateral splits, etc.) |
| **`osteosclerosis`** | Sclerosis: `compute_sclerosis_ratio.py` (four compartments: compartment-specific sclerosis ÷ bone in that compartment; legacy single-ratio API kept) |
| **`osteophyte`** | Osteophytes: `koa/osteophyte/compute_osteophyte_ratio.py` (OST vs full patella, OST/PAT; vertical midline split or L/R files—see file header) |
| **`configs`** | e.g. `jsn_config.py` (batch paths for joint space) |
| **`utils`** | `sitk_utils`, `orientation` (NRRD + anatomical axes), `bilateral_viz` (bilateral overlays) |

CLI scripts live under **`scripts/`**; demos: **`notebooks/`**.

---

## `PYTHONPATH` and imports

Add the **project root** (this README’s directory, containing the `koa` package folder) to `PYTHONPATH`:

```bash
cd /path/to/your/KOA-Diagnoistic-Tool
export PYTHONPATH="$PWD:$PYTHONPATH"

# Joint space batch (see jsn.ipynb)
python scripts/jsn.py --output /path/to/your/koa_outputs/jsn/jsn_results.csv

# JSN evaluation / thresholds (jsn_eval.ipynb; needs scikit-learn)
python scripts/jsn_eval.py --label-dir /path/to/your/jsn_eval_labels

# Sclerosis: CSV only (paths in koa/configs/sclerosis_config.py)
python scripts/osteoscierosis.py --csv-only

# Sclerosis single-case overlay (osteoscierosis.ipynb)
python scripts/osteoscierosis.py --image /path/to/your/case_0000.nrrd --mask /path/to/your/case.nrrd --out /path/to/your/scl.png --no-show

# Osteophyte: CSV only (paired L/R masks; see osteophyte_config)
python scripts/osteophyte.py --csv-only

# Osteophyte: subject base id; reads base_L / base_R from config image_dir / mask_dir
python scripts/osteophyte.py --case-id KOA01 --out /path/to/your/ost.png --no-show
```

`scripts/measure_jsw.py` forwards to `jsn.py` for backward-compatible commands.

Preferred imports:

```python
from koa.jwd import measure_knee_joint_space, direction, edges, jsn, compartments
from koa.osteophyte import osteophyte_ratios_lr_files_auto
from koa.osteosclerosis import sclerosis_results_dataframe_from_config
from koa.dashboard import plot_clinical_koa_dashboard
```

You can also use lazy top-level re-exports (see `koa/__init__.py`, `__all__`).

In **Jupyter**, set `KNEE_PKG_ROOT` to the same project root (sibling of `notebooks/`, parent of `koa/`), then `sys.path.insert(0, str(KNEE_PKG_ROOT))` before `from koa...`.

---

## Configuration

Default paths in the repo are **illustrative placeholders** (e.g. `/path/to/your/jsn_image`, `/path/to/your/jsn_mask`). Point them at your data:

- **`koa/configs/jsn_config.py`** — joint-space batch paths, etc.
- **`koa/configs/osteophyte_config.py`** — `mask_dir` / `image_dir` / `output_csv`; `osteophyte_left_suffix` / `osteophyte_right_suffix` (default `_L` / `_R`); `file_type`; `label_mapping` with `Patella` / `Patella_Osteophyte` (sync with nnU-Net label JSON if needed); `patella_label_ids`; `meta_data_csv` (base ids without `_L`/`_R`; column from `case_id_column` or `case_id` / `patient_name`).
- **`koa/configs/sclerosis_config.py`** — same IO keys; maintain `label_mapping` only; femur/tibia/sclerosis groups come from `sclerosis_label_sets_from_mapping`. CSV columns include right/left femoral and tibial sclerosis ratios (and counts).

Notebooks read these configs by default; edit them to match your `dataset.json` naming.

---

## Dependencies (see `requirements.txt`)

| Package | Use |
|---------|-----|
| numpy, scipy | Arrays and distances |
| pandas | Tables and `jsn_eval` labels |
| matplotlib | Notebooks and osteophyte / sclerosis scripts |
| opencv-python-headless | Contours in `jwd` (headless-friendly) |
| SimpleITK | NRRD/NIfTI I/O |
| pydicom | DICOM orientation when `direction_source="dicom"` |
| scikit-learn | Metrics / threshold search in `jsn_eval.py` |
| openpyxl | `.xlsx` labels in `jsn_eval` |

---

## Notebooks

| File | Content |
|------|---------|
| `jsn.ipynb` | Joint space (JWD / JSN) walkthrough |
| `jsn_eval.ipynb` | JSN evaluation and threshold search |
| `osteophyte.ipynb` | Osteophytes: one volume per side (`base_L` / `base_R`), side-by-side overlay + batch CSV |
| `osteoscierosis.ipynb` | Sclerosis: overlays vs femur/tibia/F+T + batch CSV |

---

## Ratio conventions (defaults)

- **Sclerosis**: four compartments—RF sclerosis/RF bone, RT sclerosis/RT bone, LF, LT (from `label_mapping` names). Legacy CLI flags like `--scl-r`/`--bone-r` still use the old single-ratio path.
- **Osteophytes**: one volume per side (`{base}_L` / `{base}_R`). With two `patella_label_ids`, the **smaller** label count is treated as osteophyte; sum of both = patella denominator; ties default to **higher label id = osteophyte** (`tie_osteophyte_is_higher_id`).

If orientation or label definitions differ, adjust **configs** or CLI args.
