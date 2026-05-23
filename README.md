# STS 2024 Task 2 牙齿 CBCT 半监督实例分割训练指南（nnU-Net v2）

本项目用于牙齿 CBCT 图像半监督实例分割。当前代码已经迁移到 **nnU-Net v2**，不再使用 nnU-Net v1 的 `TaskXXX`、`nnUNet_train`、`nnUNet_predict`。

本文档按“刚入门也能照着做”的方式整理，从服务器环境创建、Miniconda 安装、PyTorch 安装、数据检查、nnU-Net v2 数据准备、预处理、训练、推理到半监督伪标签迭代，完整记录一套推荐流程。

官方参考：

- nnU-Net v2: https://github.com/MIC-DKFZ/nnUNet
- nnU-Net v2 数据格式: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

## 1. 项目流程总览

本项目采用两阶段分割：

```text
原始 CBCT 图像
  -> 阶段 1：全图象限分割，预测 4 个牙齿象限
  -> 根据象限预测裁剪 4 个局部 ROI
  -> 阶段 2：象限内牙齿实例分割，预测局部牙齿 1..8
  -> 将 4 个象限结果合并回原图空间
  -> 后处理去除小连通域
```

训练时会生成两个 nnU-Net v2 数据集：

- `Dataset313_STS2024_ToothQuadrants`：全图象限分割，标签为 `0..4`。
- `Dataset312_STS2024_QuadrantTeeth`：象限 crop 后的牙齿分割，标签为 `0..9`，其中 `1..8` 是当前象限内牙齿，`9` 是其他象限牙齿。

半监督迭代时会额外生成学生数据集，默认编号：

- `Dataset323_STS2024_ToothQuadrants`
- `Dataset322_STS2024_QuadrantTeeth`

## 2. 服务器硬件检查

登录服务器后先检查 GPU、Python、Conda、磁盘：

```bash
nvidia-smi
python --version
conda --version
pwd
df -h
```

本项目当前验证过的服务器环境示例：

```text
GPU: 8 x NVIDIA GeForce RTX 4090
Driver: 580.126.09
CUDA shown by nvidia-smi: 13.0
Disk: /data1 has several TB free
```

注意：`nvidia-smi` 显示的 CUDA 版本是驱动支持的最高版本，不代表 PyTorch 必须安装相同 CUDA 版本。我们实际使用的是 PyTorch `2.4.0+cu121`，已经验证能正常识别 8 张 4090。

## 3. 安装 Miniconda

如果服务器没有 `conda`，需要先安装 Miniconda。假设你已经把安装包放在：

```text
/data1/Miniconda3-latest-Linux-x86_64.sh
```

推荐安装到个人目录：

```bash
bash /data1/Miniconda3-latest-Linux-x86_64.sh -b -p /data1/zhkang/miniconda3
```

让当前终端识别 conda：

```bash
source /data1/zhkang/miniconda3/etc/profile.d/conda.sh
conda --version
```

为了以后重新登录也能直接使用 conda：

```bash
echo 'source /data1/zhkang/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
```

以后重新登录服务器后，执行：

```bash
source ~/.bashrc
```

## 4. 创建 Python 环境

创建并激活环境：

```bash
conda create -n nnunetv2 python=3.10 -y
conda activate nnunetv2
```

检查：

```bash
python --version
pip --version
```

建议看到 Python `3.10.x`。

## 5. 安装 PyTorch

当前已验证可用的安装命令：

```bash
pip install --upgrade pip

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

验证 GPU 是否可用：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu 0:", torch.cuda.get_device_name(0))
PY
```

理想输出：

```text
torch: 2.4.0+cu121
cuda available: True
torch cuda: 12.1
gpu count: 8
gpu 0: NVIDIA GeForce RTX 4090
```

如果 `cuda available` 是 `False`，先不要继续训练，需要重新检查 PyTorch 安装版本。

## 6. 安装项目依赖

进入项目目录：

```bash
cd ~/STS2024Task2-nnUnet-main
```

安装依赖：

```bash
pip install -r requirements-linux.txt
```

`requirements-linux.txt` 中包含：

```text
nnunetv2
SimpleITK
numpy
```

验证 Python 包和命令：

```bash
python - <<'PY'
import torch
import SimpleITK
import nnunetv2
import numpy as np

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("SimpleITK ok")
print("nnunetv2 ok")
print("numpy:", np.__version__)
PY

which nnUNetv2_plan_and_preprocess
which nnUNetv2_train
which nnUNetv2_predict
```

能找到三个 `nnUNetv2_*` 命令后，环境就准备好了。

## 7. 上传和检查数据

训练至少需要：

```text
data/
  Train-Labeled/
    Images/*.nii.gz
    Masks/*_Mask.nii.gz
  Train-Unlabeled/*.nii.gz
  Validation-Public/*.nii.gz
```

`dental_CBCT_test_set` 不是训练必需目录，后面需要测试集推理或提交结果时再上传即可。

服务器项目目录应类似：

```text
~/STS2024Task2-nnUnet-main/
  README.md
  requirements-linux.txt
  process/
  pipeline/
  scripts/
  data/
    Train-Labeled/
    Train-Unlabeled/
    Validation-Public/
```

统计数量：

```bash
cd ~/STS2024Task2-nnUnet-main

echo "Train images:" && find data/Train-Labeled/Images -name "*.nii.gz" | wc -l
echo "Train masks:" && find data/Train-Labeled/Masks -name "*.nii.gz" | wc -l
echo "Unlabeled:" && find data/Train-Unlabeled -name "*.nii.gz" | wc -l
echo "Validation:" && find data/Validation-Public -name "*.nii.gz" | wc -l
```

当前服务器数据示例：

```text
Train images: 106
Train masks: 106
Unlabeled: 300
Validation: 20
```

## 8. 检查图像和标签是否匹配

先检查有标签图像是否都能找到对应 mask：

```bash
cd ~/STS2024Task2-nnUnet-main

python - <<'PY'
from pathlib import Path

image_dir = Path("data/Train-Labeled/Images")
mask_dir = Path("data/Train-Labeled/Masks")

images = sorted(image_dir.glob("*.nii.gz"))
masks = sorted(mask_dir.glob("*.nii.gz"))

missing = []
for img in images:
    case_id = img.name[:-7]
    candidates = [
        mask_dir / f"{case_id}_Mask.nii.gz",
        mask_dir / f"{case_id}.nii.gz",
        mask_dir / f"{case_id}_mask.nii.gz",
    ]
    if not any(p.exists() for p in candidates):
        missing.append(img.name)

print("images:", len(images))
print("masks:", len(masks))
print("missing masks:", len(missing))
if missing:
    print("\n".join(missing[:20]))
PY
```

理想输出：

```text
images: 106
masks: 106
missing masks: 0
```

再检查标签编号：

```bash
python - <<'PY'
from pathlib import Path
import SimpleITK as sitk
import numpy as np

mask_dir = Path("data/Train-Labeled/Masks")
masks = sorted(mask_dir.glob("*.nii.gz"))

global_labels = set()
for i, p in enumerate(masks[:10], 1):
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
    labels = sorted(int(x) for x in np.unique(arr))
    global_labels.update(labels)
    print(f"{i}. {p.name}")
    print("   labels:", labels[:80], "..." if len(labels) > 80 else "")

print("sampled global min/max:", min(global_labels), max(global_labels))
PY
```

当前数据是 FDI 标签，例如：

```text
0, 11, 12, ..., 18, 21, ..., 28, 31, ..., 38, 41, ..., 48
```

因此训练时明确使用：

```bash
export LABEL_SCHEME=fdi
```

## 9. 设置 nnU-Net 工作目录

推荐把 nnU-Net 工作目录放在大容量磁盘：

```bash
cd ~/STS2024Task2-nnUnet-main

export DATA_ROOT="$PWD/data"
export NNUNET_BASE="/data1/zhkang/nnunet_work"

export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
export nnUNet_results="$NNUNET_BASE/nnUNet_results"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

echo "DATA_ROOT=$DATA_ROOT"
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"
```

后续每次新开终端训练或推理，都建议重新执行这一段 `export`。

为了以后不用每次手动 export，可以把这三行写入 ~/.bashrc

cat >> ~/.bashrc <<'EOF'
export NNUNET_BASE="/data1/zhkang/nnunet_work"
export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
export nnUNet_results="$NNUNET_BASE/nnUNet_results"
EOF

写完后当前终端执行一次：  source ~/.bashrc

## 10. 生成 nnU-Net v2 数据集

当前数据是 FDI 标签，所以使用 `--label-scheme fdi`：

```bash
cd ~/STS2024Task2-nnUnet-main

python process/prepare_nnunetv2_datasets.py \
  --data-root "$DATA_ROOT" \
  --nnunet-raw "$nnUNet_raw" \
  --label-scheme fdi \
  --overwrite
```

这个脚本会生成：

```text
$nnUNet_raw/
  Dataset313_STS2024_ToothQuadrants/
    dataset.json
    imagesTr/*_0000.nii.gz
    labelsTr/*.nii.gz
  Dataset312_STS2024_QuadrantTeeth/
    dataset.json
    imagesTr/*_0000.nii.gz
    labelsTr/*.nii.gz
```

检查数量：

```bash
echo "Dataset313 images:"
find "$nnUNet_raw/Dataset313_STS2024_ToothQuadrants/imagesTr" -name "*.nii.gz" | wc -l

echo "Dataset313 labels:"
find "$nnUNet_raw/Dataset313_STS2024_ToothQuadrants/labelsTr" -name "*.nii.gz" | wc -l

echo "Dataset312 images:"
find "$nnUNet_raw/Dataset312_STS2024_QuadrantTeeth/imagesTr" -name "*.nii.gz" | wc -l

echo "Dataset312 labels:"
find "$nnUNet_raw/Dataset312_STS2024_QuadrantTeeth/labelsTr" -name "*.nii.gz" | wc -l
```

当前数据示例结果：

```text
Dataset313 images: 106
Dataset313 labels: 106
Dataset312 images: 422
Dataset312 labels: 422
```

`Dataset312` 不一定正好是 `106 * 4 = 424`，因为有些病例可能缺牙或某些象限为空。

查看 `dataset.json`：

```bash
cat "$nnUNet_raw/Dataset313_STS2024_ToothQuadrants/dataset.json"
cat "$nnUNet_raw/Dataset312_STS2024_QuadrantTeeth/dataset.json"
```

## 11. 重要说明：mask spacing mismatch

如果预处理时报：

```text
Error: Spacing mismatch between segmentation and corresponding images.
Spacing images: 0.2, 0.2, 0.2
Spacing seg: 1.0, 1.0, 1.0
```

说明原始 mask 的 NIfTI 文件头信息不正确。当前代码已经修复：生成 nnU-Net label 时会继承对应 image 的 spacing/origin/direction，而不是继承原始 mask 的错误信息。

可以用下面命令检查一个病例：

```bash
python - <<'PY'
import SimpleITK as sitk
from pathlib import Path

case = "STS24_Train_X2313838"
raw = Path("/data1/zhkang/nnunet_work/nnUNet_raw/Dataset313_STS2024_ToothQuadrants")

img = sitk.ReadImage(str(raw / "imagesTr" / f"{case}_0000.nii.gz"))
seg = sitk.ReadImage(str(raw / "labelsTr" / f"{case}.nii.gz"))

print("image spacing:", img.GetSpacing())
print("seg spacing:  ", seg.GetSpacing())
print("image origin:", img.GetOrigin())
print("seg origin:  ", seg.GetOrigin())
print("image size:", img.GetSize())
print("seg size:  ", seg.GetSize())
PY
```

理想情况是 image 和 seg 的 spacing、origin、size 都一致。

## 12. nnU-Net v2 预处理

建议把输出保存到日志：

```bash
cd ~/STS2024Task2-nnUnet-main
mkdir -p logs

nnUNetv2_plan_and_preprocess -d 313 --verify_dataset_integrity \
  2>&1 | tee logs/preprocess_313.log

nnUNetv2_plan_and_preprocess -d 312 --verify_dataset_integrity \
  2>&1 | tee logs/preprocess_312.log
```

检查预处理结果：

```bash
find "$nnUNet_preprocessed/Dataset313_STS2024_ToothQuadrants" -maxdepth 2 -type d | sort
find "$nnUNet_preprocessed/Dataset312_STS2024_QuadrantTeeth" -maxdepth 2 -type d | sort
```

当前数据已验证的 planner 结果：

```text
Dataset313_STS2024_ToothQuadrants:
  nnUNetPlans_2d
  nnUNetPlans_3d_fullres
  nnUNetPlans_3d_lowres

Dataset312_STS2024_QuadrantTeeth:
  nnUNetPlans_2d
  nnUNetPlans_3d_fullres
```

因此推荐训练配置：

```bash
Dataset313: 3d_lowres
Dataset312: 3d_fullres
```

## 13. 使用 tmux 防止训练中断

训练时间很长，强烈建议使用 `tmux`，避免 SSH 断开导致训练停止。

检查 tmux：

```bash
tmux -V
```

新建训练会话：

```bash
tmux new -s sts_train
```

在 tmux 中启动训练后，如需退出但保持训练继续：

```text
Ctrl+b
松开
d
```

重新进入：

```bash
tmux attach -t sts_train
```

查看日志：

```bash
tail -f logs/train_313_3d_lowres_fold0.log
```

查看 GPU：

```bash
nvidia-smi
```

## 14. 训练策略

nnU-Net v2 默认训练约 `1000 epochs`。当前阶段建议：

```text
先训练 fold 0，完整跑通两阶段流程。
不要一开始就训练 all 或 5 折。
```

推荐顺序：

```text
1. 训练 Dataset313 3d_lowres fold 0
2. 训练 Dataset312 3d_fullres fold 0
3. 用两个 fold 0 模型跑 Validation-Public 推理
4. 确认结果正常后，再考虑 all、5 折或半监督迭代
```

训练中如果看到 PyTorch 的 warning，例如：

```text
torch/fx/experimental/symbolic_shapes.py
unknown range
```

一般不是错误。只要没有 `Traceback`、`RuntimeError`、`CUDA out of memory`、`loss nan`，就先让训练继续。

如果下一次想减少 `torch.compile` 相关 warning，可以在训练前设置：

```bash
export nnUNet_compile=f
```

## 15. 训练 Dataset313 象限模型

在 tmux 中执行：

```bash
cd ~/STS2024Task2-nnUnet-main

export NNUNET_BASE="/data1/zhkang/nnunet_work"
export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
export nnUNet_results="$NNUNET_BASE/nnUNet_results"

mkdir -p logs

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 313 3d_lowres 0 --npz \
  2>&1 | tee logs/train_313_3d_lowres_fold0.log
```

正常启动时会看到类似：

```text
Creating new 5-fold cross-validation split...
Desired fold for training: 0
This split has 84 training and 22 validation cases.
using pin_memory on device 0
```

训练完成后检查 checkpoint：

```bash
find "$nnUNet_results" -path "*Dataset313*" -name "checkpoint_final.pth"
find "$nnUNet_results" -path "*Dataset313*" -name "checkpoint_best.pth"
```

## 16. 训练 Dataset312 牙齿模型

`Dataset313` 完成后，再训练第二阶段模型：

```bash
cd ~/STS2024Task2-nnUnet-main

export NNUNET_BASE="/data1/zhkang/nnunet_work"
export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
export nnUNet_results="$NNUNET_BASE/nnUNet_results"

mkdir -p logs

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 312 3d_fullres 0 --npz \
  2>&1 | tee logs/train_312_3d_fullres_fold0.log
```

完成后检查：

```bash
find "$nnUNet_results" -path "*Dataset312*" -name "checkpoint_final.pth"
find "$nnUNet_results" -path "*Dataset312*" -name "checkpoint_best.pth"
```

## 17. 两阶段验证集推理

如果你训练的是 fold 0，推理时必须设置：

```bash
export FOLD=0
```

验证集推理：

```bash
cd ~/STS2024Task2-nnUnet-main

export NNUNET_BASE="/data1/zhkang/nnunet_work"
export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
export nnUNet_results="$NNUNET_BASE/nnUNet_results"

export QUADRANT_DATASET_ID=313
export TOOTH_DATASET_ID=312
export QUADRANT_CONFIG=3d_lowres
export TOOTH_CONFIG=3d_fullres
export FOLD=0
export QUADRANT_CHECKPOINT=checkpoint_final.pth
export TOOTH_CHECKPOINT=checkpoint_final.pth
export OUTPUT_LABEL_SCHEME=fdi
export DISABLE_TTA=1

bash scripts/predict_v2.sh data/Validation-Public runs/val_pred_fold0
```

最终结果：

```text
runs/val_pred_fold0/final/*_Mask.nii.gz
```

推理目录结构：

```text
runs/val_pred_fold0/
  nnunet_inputs/
  quadrant_predictions/
  quadrant_resizer/
  quadrant_cropped_inputs/
  tooth_predictions/
  final/
```

## 18. 后续正式训练建议

fold 0 跑通后，可以选择：

```text
方案 A：继续训练 fold 1-4，做更完整的交叉验证。
方案 B：训练 fold all，用全部有标签数据训练最终模型。
方案 C：先用 fold 0 teacher 生成伪标签，跑一轮 student，验证半监督流程。
```

训练 `all` 示例：

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 313 3d_lowres all --npz \
  2>&1 | tee logs/train_313_3d_lowres_all.log

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 312 3d_fullres all --npz \
  2>&1 | tee logs/train_312_3d_fullres_all.log
```

如果使用 `all` 模型推理：

```bash
export FOLD=all
```

## 19. 半监督伪标签迭代

无标签数据目录：

```text
data/Train-Unlabeled
```

当前服务器示例数量：

```text
300
```

一次 teacher -> pseudo label -> student 迭代：

```bash
cd ~/STS2024Task2-nnUnet-main

export NNUNET_BASE="/data1/zhkang/nnunet_work"
export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
export nnUNet_results="$NNUNET_BASE/nnUNet_results"

export LABEL_SCHEME=fdi
export PSEUDO_OUTPUT_LABEL_SCHEME=fdi
export TEACHER_QUADRANT_DATASET_ID=313
export TEACHER_TOOTH_DATASET_ID=312
export STUDENT_QUADRANT_DATASET_ID=323
export STUDENT_TOOTH_DATASET_ID=322
export QUADRANT_CONFIG=3d_lowres
export TOOTH_CONFIG=3d_fullres
export FOLD=0
export CHECKPOINTS="checkpoint_best.pth checkpoint_final.pth"
export PSEUDO_THRESHOLD=0.90
export PSEUDO_TOP_K=30

bash scripts/pseudo_iteration_v2.sh
```

该脚本会：

1. 用 teacher 对 `data/Train-Unlabeled` 做两阶段推理。
2. 对多个 checkpoint 的预测结果计算 pairwise multi-class Dice。
3. 把一致性 Dice 高于阈值的伪标签复制到 `runs/pseudo_iter1/selected_pseudo_labels/`。
4. 用原始标注数据 + 选中的伪标签生成学生数据集。
5. 训练学生象限模型和学生牙齿模型。

筛选报告：

```text
runs/pseudo_iter1/pseudo_selection.csv
```

第二轮伪标签迭代可改成：

```bash
export TEACHER_QUADRANT_DATASET_ID=323
export TEACHER_TOOTH_DATASET_ID=322
export STUDENT_QUADRANT_DATASET_ID=333
export STUDENT_TOOTH_DATASET_ID=332
export PSEUDO_WORK_DIR=runs/pseudo_iter2

bash scripts/pseudo_iteration_v2.sh
```

## 20. 常用监控命令

查看 GPU：

```bash
nvidia-smi
```

实时看训练日志：

```bash
tail -f logs/train_313_3d_lowres_fold0.log
tail -f logs/train_312_3d_fullres_fold0.log
```

看最近 50 行：

```bash
tail -n 50 logs/train_313_3d_lowres_fold0.log
```

查 checkpoint：

```bash
find /data1/zhkang/nnunet_work/nnUNet_results -name "checkpoint_final.pth"
find /data1/zhkang/nnunet_work/nnUNet_results -name "checkpoint_best.pth"
```

查预处理目录：

```bash
find /data1/zhkang/nnunet_work/nnUNet_preprocessed -maxdepth 2 -type d | sort
```

## 21. 文件说明

- `process/prepare_nnunetv2_datasets.py`：生成 nnU-Net v2 raw datasets，是训练入口的核心数据转换脚本。
- `process/prepare_inference_inputs.py`：把普通 NIfTI 目录复制成 nnU-Net v2 推理输入命名 `*_0000.nii.gz`。
- `process/FDI2Qua.py`：单独把实例牙齿标签转成象限标签。
- `process/preparefor2.py`：单独生成第二阶段象限 crop 图像和标签。
- `process/select_pseudo_dice.py`：基于多 checkpoint 预测一致性 Dice 筛选伪标签。
- `pipeline/quadrant_locate.py`：根据阶段 1 象限预测裁剪阶段 2 ROI。
- `pipeline/quadrant_merge.py`：将阶段 2 象限内牙齿预测合并回原图。
- `pipeline/postprocess_small_components.py`：按每个标签的连通域移除小组件。
- `pipeline/predict.sh`：兼容入口，实际调用 `scripts/predict_v2.sh`。
- `scripts/train_v2_supervised.sh`：监督训练入口，默认会训练 `all`，新手建议先手动按本文档训练 `fold 0`。
- `scripts/predict_v2.sh`：两阶段推理入口。
- `scripts/pseudo_iteration_v2.sh`：半监督伪标签迭代入口。

## 22. 常见问题

如果 `nnUNetv2_train 313 3d_lowres ...` 报配置不存在，说明 planner 没有为该数据集生成 `3d_lowres`，把 `QUADRANT_CONFIG` 改成 `3d_fullres`。

如果伪标签训练时标签不对，优先检查 `LABEL_SCHEME` 和 `PSEUDO_OUTPUT_LABEL_SCHEME` 是否一致。当前数据是 FDI 标签，所以推荐都设置为 `fdi`。

如果推理结果要提交评测，通常使用 `OUTPUT_LABEL_SCHEME=fdi`。

如果推理结果要进入下一轮训练，使用：

```bash
export OUTPUT_LABEL_SCHEME="$LABEL_SCHEME"
```

如果训练中出现 `CUDA out of memory`，先确认是否只使用了一个 GPU：

```bash
echo $CUDA_VISIBLE_DEVICES
```

也可以换一张空闲 GPU：

```bash
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 313 3d_lowres 0 --npz
```

如果 SSH 断开导致训练中断，说明没有使用 tmux 或 nohup。以后训练前先进入：

```bash
tmux new -s sts_train
```

如果训练过程中只有 warning，没有 `Traceback` 或 `RuntimeError`，通常先不要中断，让它继续跑。
