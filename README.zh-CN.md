<p align="right">
<a href="README.md"><img src="https://img.shields.io/badge/lang-English-2ea44f?style=flat-square" alt="English README"></a>
</p>

# KOA（膝关节 OA 结构测量）

本目录包含 Python 包 **`koa`**，用于从影像与分割结果中提取膝关节骨关节炎（OA）相关的**结构**信息，主要分为三类：

- **关节间隙**：可测的间隙宽度等**数值**称 **JWD**（joint width distance，常用 mm）；基于间隙的**评估语境**（是否变窄等）称 **JSN**。文献常称 JSW，与本包 JWD 同指宽度；历史脚本/列名仍多含 ``jsn``，其中 ``*_mm`` 语义为 JWD。
- **软骨下骨硬化（SCL）**：由硬化区域分割计算占比等；
- **骨刺（OST）**：由骨刺相关分割计算占比等。

三者共同对应影像学 K–L 分级常用的要素（间隙变窄、骨刺、硬化）。**K–L 综合分级 / 自动诊断尚未在本包中实现。**

若使用 **nnU-Net** 做训练与推理，相关脚本通常放在**独立的流水线仓库**中；本仓库仅包含**分割之后**的测量与评估代码。

---

## 环境与依赖（先前使用记录 + 推荐搭建）

**先前上下文：** 仓库内 `jsn_eval.ipynb` 等保存的内核名为 **`image_analysis_env`**，Python **3.11**。该名称仅为当时的 Conda/venv 习惯命名，并非必须；你可任意命名新环境。

**推荐步骤（Conda 示例）：**

```bash
# 1) 新建环境（Python 3.11 与先前笔记本一致；3.10/3.12 通常也可）
conda create -n koa python=3.11 -y
conda activate koa

# 2) 进入本仓库根目录并安装依赖（将路径换为你本机克隆位置）
cd /path/to/your/KOA-Diagnoistic-Tool
pip install -r requirements.txt
```

**仅用 pip（无 Conda）时：**

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

安装完成后，继续下一节设置 **`PYTHONPATH`**。

---

## 目录结构（`koa/`）

| 模块 | 作用 |
|------|------|
| **`jwd`** | 根据 2D 股骨–胫骨标签图做 **JSN** 评估；输出中的 mm 量为 **JWD**（`measure_knee_joint_space`、方向、骨缘、内外侧划分等） |
| **`osteosclerosis`** | 硬化：``compute_sclerosis_ratio.py``（四分腔：各腔室 **专用硬化类** ÷ **该腔室骨段**；另保留旧版「单侧硬化总和÷骨并集」API） |
| **`osteophyte`** | 骨刺：``koa/osteophyte/compute_osteophyte_ratio.py``（骨刺相对全髌像素，记作 OST/PAT；竖直中线分幅或左 / 右文件；记法见该文件顶部） |
| **`configs`** | 例如 `jsn_config.py`（关节间隙批量路径等） |
| **`utils`** | `sitk_utils`、`orientation`（NRRD 与解剖轴）、`bilateral_viz`（双侧结果叠图） |

命令行脚本（与各笔记本对应）见 **`scripts/`**；示例与交互：`notebooks/`。

---

## 环境变量与导入

将 **`KOA`** 项目根目录（本 README 所在目录，且其下包含 `koa` 包目录）加入 `PYTHONPATH`：

```bash
cd /path/to/your/KOA-Diagnoistic-Tool
export PYTHONPATH="$PWD:$PYTHONPATH"

# 关节间隙批量测量（对应 jsn.ipynb）
python scripts/jsn.py --output /path/to/your/koa_outputs/jsn/jsn_results.csv

# JSN 评估与最优阈值（对应 jsn_eval.ipynb，需 scikit-learn）
python scripts/jsn_eval.py --label-dir /path/to/your/jsn_eval_labels

# 硬化：仅批量写 CSV（路径见 koa/configs/sclerosis_config.py；与笔记本相同入口 ``sclerosis_results_dataframe_from_config``）
python scripts/osteoscierosis.py --csv-only

# 硬化单例叠图（对应 osteoscierosis.ipynb）
python scripts/osteoscierosis.py --image /path/to/your/case_0000.nrrd --mask /path/to/your/case.nrrd --out /path/to/your/scl.png --no-show

# 骨刺：仅批量写 CSV（L/R 成对 mask，见 osteophyte_config）
python scripts/osteophyte.py --csv-only

# 骨刺：受试者 base id，从 config 的 image_dir / mask_dir 读 base_L、base_R
python scripts/osteophyte.py --case-id KOA01 --out /path/to/your/ost.png --no-show

```

`scripts/measure_jsw.py` 仅转发到 `jsn.py`，便于沿用旧命令。

推荐子包导入（依赖清晰）：

```python
from koa.jwd import measure_knee_joint_space, direction, edges, jsn, compartments
from koa.osteophyte import osteophyte_ratios_lr_files_auto
from koa.osteosclerosis import sclerosis_results_dataframe_from_config
from koa.dashboard import plot_clinical_koa_dashboard
```

也可惰性从顶层导入等价符号，例如 ``from koa import measure_knee_joint_space``（见 ``koa/__init__.py`` 中 ``__all__``）。

在 **Jupyter** 中：将 `KNEE_PKG_ROOT` 设为上述 **`KOA` 项目根目录**（与 `notebooks/` 同级、`koa/` 所在目录），`sys.path.insert(0, str(KNEE_PKG_ROOT))` 后再 `from koa...`。

---

## 配置

仓库内默认路径均为**示意占位路径**（如 `/path/to/your/jsn_image`、`/path/to/your/jsn_mask`），请按你的数据布局修改：

- **`koa/configs/jsn_config.py`**：关节间隙批量路径等。
- **`koa/configs/osteophyte_config.py`**：`mask_dir` / `image_dir` / `output_csv`；`osteophyte_left_suffix` / `osteophyte_right_suffix`（默认 ``_L`` / ``_R``）；`volume_extensions` / `file_type`；`label_mapping` 推荐键 ``Patella`` / ``Patella_Osteophyte``（与 nnU-Net 标签表一致时请同步改名）；`patella_label_ids`（默认 `[1,2]`，**每张图全图**内少像素标签 = 骨刺，两类之和 = 髌骨）；`meta_data_csv`（列为 **base** id，不含 `_L`/`_R` 后缀；列名见 `case_id_column` 或回退 `case_id` / `patient_name`）。
- **`koa/configs/sclerosis_config.py`**：同上 IO 字段；**仅**维护 `label_mapping`（类名 → ID）。股骨 / 胫骨 / 硬化分组由 `sclerosis_label_sets_from_mapping` 从类名解析。CSV 中为 **右/左股骨硬化比、右/左胫骨硬化比**（及像素计数）。

笔记本会从上述 config **读默认路径**；改 config 即可与 nnU-Net `dataset.json` 对齐。

---

## 依赖一览（与 requirements.txt 一致）

| 包 | 用途 |
|----|------|
| numpy, scipy | 数组与距离计算 |
| pandas | 表格与 `jsn_eval` 读入标注 |
| matplotlib | 笔记本与 `osteophyte` / `osteoscierosis` 脚本出图 |
| opencv-python-headless | `jwd` 轮廓（无 GUI 依赖，服务器友好） |
| SimpleITK | NRRD/NIfTI 读写 |
| pydicom | `direction_source="dicom"` 时读 DICOM 朝向 |
| scikit-learn | `jsn_eval.py` 指标与阈值搜索 |
| openpyxl | `jsn_eval` 读取 `.xlsx` 标注表 |

---

## 可视化笔记本说明

|  文件名 | 内容 |
|-----------|------|
| `jsn.ipynb` | 关节间隙（JWD/JSN）流程演示 |
| `jsn_eval.ipynb` | JSN 与狭窄阈值评估 |
| `osteophyte.ipynb` | **骨刺**：左/右 **各一张** 图（`base_L`/`base_R`），并排叠图 + 占比 + **批量 CSV** |
| `osteoscierosis.ipynb` | **硬化**：vs 股骨/胫骨/F+T，叠图 + **批量 CSV** |

---

## 比例含义（默认约定）

- **硬化（SCL）**：四分腔——**右股骨硬化/右股骨**、**右胫骨硬化/右胫骨**、左股骨、左胫骨（各腔室独立分子；`label_mapping` 通过类名解析）。CLI 若传入 `--scl-r`/`--bone-r` 等，则仍走旧的「单侧硬化总和 vs 骨并集」单比例逻辑。
- **骨刺（OST）**：**左、右各一张** 图（文件名 `{base}_L` / `{base}_R`）。每张 **全图** 内用配置中的 `patella_label_ids`（两标签）统计：**较少像素 = 骨刺**，两类之和 = 髌骨分母；`plot_lr_knee_images_overlay` 等在 **同一 figure** 并排显示两侧占比。打平手时默认 **较大标签 ID 视为骨刺**（`tie_osteophyte_is_higher_id`）。

若体位或标签定义不同，请改对应 **config** 或传参。
