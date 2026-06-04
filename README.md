# STS2024 Task 2 牙齿 CBCT 半监督实例分割复现指南（nnU-Net v2）

本项目用于牙齿 CBCT 图像的半监督实例分割。代码已经迁移到 **nnU-Net v2**，不再使用 nnU-Net v1 的 `TaskXXX`、`nnUNet_train`、`nnUNet_predict`。

本文档只保留复现实验所需的正确主流程：环境安装、数据准备、nnU-Net v2 数据集生成、预处理、五折训练、测试集推理、Dice/HD95 评价、无标签伪标签推理。按顺序执行即可复现当前项目的主要结果。

## 1. 最终推荐方案

本项目采用两阶段分割：

```text
CBCT 原图
  -> 阶段 1：四象限分割，得到 1、2、3、4 四个牙齿象限
  -> 根据四象限预测裁剪 4 个局部 ROI
  -> 阶段 2：在每个象限 ROI 内做牙齿实例分割
  -> 合并 4 个象限的局部实例预测，恢复到原图空间
  -> 去除小连通域，得到最终 FDI 牙齿实例标签
```

最终主模型：

```text
四象限模型：Dataset423，3d_lowres，5-fold ensemble
牙齿实例模型：Dataset412，3d_fullres，5-fold ensemble
推理策略：checkpoint_final.pth + TTA
输出标签：标准 FDI 标签，11-18、21-28、31-38、41-48
```

当前测试集结果：

```text
Test cases: 50
Instance-level mean Dice: 0.9562
Instance-level mean HD95: 0.5754 mm
Foreground Dice: 0.9684
Foreground HD95: 0.2645 mm
```

`checkpoint_best.pth + TTA` 也可作为对照：

```text
Instance-level mean Dice: 0.9247
Instance-level mean HD95: 1.5498 mm
Foreground Dice: 0.9677
Foreground HD95: 0.2686 mm
```

因此论文主结果建议使用 `checkpoint_final.pth + TTA`。

## 2. 服务器环境

当前项目验证过的环境示例：

```text
GPU: NVIDIA GeForce RTX 4090
Driver: 580.126.09
Python: 3.10
PyTorch: 2.4.0+cu121
nnU-Net v2: 2.7.0
```

先检查服务器：

```bash
nvidia-smi
pwd
df -h
```

`nvidia-smi` 显示的 CUDA 版本是驱动支持的最高版本，不要求 PyTorch 安装完全相同的 CUDA 版本。当前项目使用 `torch==2.4.0+cu121` 已经验证可用。

## 3. 安装 Miniconda

如果服务器还没有 conda，先安装 Miniconda。假设安装包在：

```text
/data1/Miniconda3-latest-Linux-x86_64.sh
```

安装：

```bash
bash /data1/Miniconda3-latest-Linux-x86_64.sh -b -p /data1/zhkang/miniconda3
```

让当前终端识别 conda：

```bash
source /data1/zhkang/miniconda3/etc/profile.d/conda.sh
conda --version
```

写入 `~/.bashrc`：

```bash
echo 'source /data1/zhkang/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
source ~/.bashrc
```

## 4. 创建环境并安装依赖

```bash
conda create -n nnunetv2 python=3.10 -y
conda activate nnunetv2
```

安装 PyTorch：

```bash
pip install --upgrade pip

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

进入项目并安装依赖：

```bash
cd ~/STS2024Task2-nnUnet-main
pip install -r requirements-linux.txt
```

验证：

```bash
python - <<'PY'
import torch
import SimpleITK
import nnunetv2
import numpy as np

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu 0:", torch.cuda.get_device_name(0))
print("SimpleITK ok")
print("nnunetv2 ok")
print("numpy:", np.__version__)
PY

which nnUNetv2_plan_and_preprocess
which nnUNetv2_train
which nnUNetv2_predict
```

理想输出包含：

```text
cuda available: True
gpu count: 大于 0
```

## 5. 固定 nnU-Net v2 工作目录

推荐把 nnU-Net 的 raw、preprocessed、results 放在大容量磁盘：

```bash
cat >> ~/.bashrc <<'EOF'
export NNUNET_BASE="/data1/zhkang/nnunet_work"
export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
export nnUNet_results="$NNUNET_BASE/nnUNet_results"
EOF

source ~/.bashrc
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
```

之后每次进入项目：

```bash
conda activate nnunetv2
cd ~/STS2024Task2-nnUnet-main
```

检查：

```bash
echo "$nnUNet_raw"
echo "$nnUNet_preprocessed"
echo "$nnUNet_results"
```

## 6. 数据目录结构

项目要求的数据目录：

```text
data/
  Train-Labeled/
    Images/*.nii.gz
    Masks/*.nii.gz

  Train-Unlabeled/*.nii.gz

  Validation-Public/*.nii.gz

  Manual-Pseudo-Quadrant-Labels/*.nii.gz

  dental_CBCT_test_set/
    images/*.nii.gz
    labels/*.nii.gz
```

各目录用途：

```text
Train-Labeled
  有金标准训练数据。Mask 是 FDI 标签：11-18、21-28、31-38、41-48。

Train-Unlabeled
  无标签训练数据，用于生成伪标签。

Validation-Public
  无金标准，只用于可视化检查或比赛提交，不能计算 Dice/HD95。

Manual-Pseudo-Quadrant-Labels
  手工筛选的四象限伪标签，标签应为 0、1、2、3、4。

dental_CBCT_test_set
  有金标准测试集，用于论文定量评价和可视化展示。
```

检查数量：

```bash
cd ~/STS2024Task2-nnUnet-main

echo "Train images:"
find data/Train-Labeled/Images -maxdepth 1 -name "*.nii.gz" | wc -l

echo "Train masks:"
find data/Train-Labeled/Masks -maxdepth 1 -name "*.nii.gz" | wc -l

echo "Unlabeled:"
find data/Train-Unlabeled -maxdepth 1 -name "*.nii.gz" | wc -l

echo "Validation:"
find data/Validation-Public -maxdepth 1 -name "*.nii.gz" | wc -l

echo "Test images:"
find data/dental_CBCT_test_set/images -maxdepth 1 -name "*.nii.gz" | wc -l

echo "Test labels:"
find data/dental_CBCT_test_set/labels -maxdepth 1 -name "*.nii.gz" | wc -l
```

## 7. 检查训练标签

检查 `Train-Labeled` 的标签值：

```bash
python - <<'PY'
from pathlib import Path
import SimpleITK as sitk
import numpy as np

mask_dir = Path("data/Train-Labeled/Masks")
for i, p in enumerate(sorted(mask_dir.glob("*.nii.gz"))[:10], 1):
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
    print(i, p.name)
    print("labels:", sorted(np.unique(arr).astype(int).tolist()))
PY
```

如果输出包含：

```text
0, 11, 12, ..., 18, 21, ..., 28, 31, ..., 38, 41, ..., 48
```

说明数据是 FDI 标签。后续统一使用：

```text
--label-scheme fdi
```

检查手工四象限伪标签：

```bash
python - <<'PY'
import SimpleITK as sitk
import numpy as np
from pathlib import Path

label_dir = Path("data/Manual-Pseudo-Quadrant-Labels")
for p in sorted(label_dir.glob("*.nii.gz"))[:10]:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
    labels = sorted(np.unique(arr).astype(int).tolist())
    zero_ratio = float((arr == 0).mean())
    print(p.name, "labels=", labels, "zero_ratio=", round(zero_ratio, 4))
PY
```

正常应看到：

```text
labels = [0, 1, 2, 3, 4]
zero_ratio 大约 0.99
```

## 8. 数据集编号

最终复现主流程使用：

```text
Dataset412_STS2024_QuadrantTeethLabeledV2
  来源：新版 Train-Labeled。
  任务：象限 crop 内牙齿实例分割。
  配置：3d_fullres。

Dataset423_STS2024_ToothQuadrantsManualPseudoV2
  来源：新版 Train-Labeled + Manual-Pseudo-Quadrant-Labels。
  任务：全图四象限分割。
  配置：3d_lowres。
```

辅助对照：

```text
Dataset413_STS2024_ToothQuadrantsLabeledV2
  只使用新版 Train-Labeled 的四象限数据集，可用于对照。

Dataset443_STS2024_ToothQuadrantsAllPseudoV2
  加入了实例伪标签转四象限后的数据，可用于对照。
  当前实验没有优于 423，不作为最终主线。
```

## 9. 生成 Dataset412 和 Dataset413

执行：

```bash
cd ~/STS2024Task2-nnUnet-main

python process/prepare_nnunetv2_datasets.py \
  --data-root data \
  --nnunet-raw "$nnUNet_raw" \
  --quadrant-dataset-id 413 \
  --tooth-dataset-id 412 \
  --quadrant-dataset-name STS2024_ToothQuadrantsLabeledV2 \
  --tooth-dataset-name STS2024_QuadrantTeethLabeledV2 \
  --label-scheme fdi \
  --overwrite
```

检查：

```bash
find "$nnUNet_raw" -maxdepth 1 -type d -name "Dataset413*"
find "$nnUNet_raw" -maxdepth 1 -type d -name "Dataset412*"

cat "$nnUNet_raw/Dataset412_STS2024_QuadrantTeethLabeledV2/dataset.json"

ls "$nnUNet_raw/Dataset412_STS2024_QuadrantTeethLabeledV2/imagesTr" | wc -l
ls "$nnUNet_raw/Dataset412_STS2024_QuadrantTeethLabeledV2/labelsTr" | wc -l
```

## 10. 生成 Dataset423

执行：

```bash
python process/prepare_quadrant_pseudo_dataset.py \
  --data-root data \
  --nnunet-raw "$nnUNet_raw" \
  --dataset-id 423 \
  --dataset-name STS2024_ToothQuadrantsManualPseudoV2 \
  --label-scheme fdi \
  --pseudo-image-dir data/Train-Unlabeled \
  --pseudo-label-dir data/Manual-Pseudo-Quadrant-Labels \
  --overwrite
```

检查：

```bash
find "$nnUNet_raw" -maxdepth 1 -type d -name "Dataset423*"

cat "$nnUNet_raw/Dataset423_STS2024_ToothQuadrantsManualPseudoV2/dataset.json"

ls "$nnUNet_raw/Dataset423_STS2024_ToothQuadrantsManualPseudoV2/imagesTr" | wc -l
ls "$nnUNet_raw/Dataset423_STS2024_ToothQuadrantsManualPseudoV2/labelsTr" | wc -l
```

## 11. 预处理

预处理是训练前必须做的步骤。它会检查数据完整性、计算数据指纹、生成训练计划，并把数据转换成 nnU-Net v2 训练时读取的格式。

执行：

```bash
mkdir -p logs

nnUNetv2_plan_and_preprocess -d 423 --verify_dataset_integrity \
  2>&1 | tee logs/preprocess_423_manual_quadrant_v2.log

nnUNetv2_plan_and_preprocess -d 412 --verify_dataset_integrity \
  2>&1 | tee logs/preprocess_412_labeled_v2.log
```

检查：

```bash
find "$nnUNet_preprocessed/Dataset423_STS2024_ToothQuadrantsManualPseudoV2" -maxdepth 2 -type d | sort
find "$nnUNet_preprocessed/Dataset412_STS2024_QuadrantTeethLabeledV2" -maxdepth 2 -type d | sort
```

正常应能看到：

```text
Dataset423:
  nnUNetPlans_2d
  nnUNetPlans_3d_lowres
  nnUNetPlans_3d_fullres

Dataset412:
  nnUNetPlans_2d
  nnUNetPlans_3d_fullres
```

最终只使用：

```text
423 -> 3d_lowres
412 -> 3d_fullres
```

## 12. 训练 Dataset423 四象限模型

建议使用 tmux，避免 SSH 断开影响训练：

```bash
tmux new -s sts_train
```

五折训练：

```bash
cd ~/STS2024Task2-nnUnet-main
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 423 3d_lowres 0 --npz \
  2>&1 | tee logs/train_423_3d_lowres_fold0.log

CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 423 3d_lowres 1 --npz \
  2>&1 | tee logs/train_423_3d_lowres_fold1.log

CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 423 3d_lowres 2 --npz \
  2>&1 | tee logs/train_423_3d_lowres_fold2.log

CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 423 3d_lowres 3 --npz \
  2>&1 | tee logs/train_423_3d_lowres_fold3.log

CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 423 3d_lowres 4 --npz \
  2>&1 | tee logs/train_423_3d_lowres_fold4.log
```

当前实验结果：

```text
fold0: 0.7766
fold1: 0.8845
fold2: 0.8109
fold3: 0.8117
fold4: 0.7653
mean: 0.8096
```

说明：四象限任务更依赖全局上下左右信息，`3d_lowres` 比 `3d_fullres` 更适合作为主模型。

## 13. 训练 Dataset412 牙齿实例模型

五折训练：

```bash
cd ~/STS2024Task2-nnUnet-main
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 412 3d_fullres 0 --npz \
  2>&1 | tee logs/train_412_3d_fullres_fold0.log

CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 412 3d_fullres 1 --npz \
  2>&1 | tee logs/train_412_3d_fullres_fold1.log

CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 412 3d_fullres 2 --npz \
  2>&1 | tee logs/train_412_3d_fullres_fold2.log

CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 412 3d_fullres 3 --npz \
  2>&1 | tee logs/train_412_3d_fullres_fold3.log

CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 412 3d_fullres 4 --npz \
  2>&1 | tee logs/train_412_3d_fullres_fold4.log
```

当前实验结果：

```text
fold0: 0.8744
fold1: 0.9008
fold2: 0.8677
fold3: 0.9267
fold4: 0.9133
mean: 0.8966
```

## 14. 汇总训练指标

```bash
for f in 0 1 2 3 4; do
  echo "423 fold$f"
  grep -h "Mean Validation Dice" logs/train_423_3d_lowres_fold${f}.log | tail -1
done

for f in 0 1 2 3 4; do
  echo "412 fold$f"
  grep -h "Mean Validation Dice" logs/train_412_3d_fullres_fold${f}.log | tail -1
done
```

检查 checkpoint：

```bash
find "$nnUNet_results" -path "*Dataset423*" -name "checkpoint_final.pth" | sort
find "$nnUNet_results" -path "*Dataset412*" -name "checkpoint_final.pth" | sort
find "$nnUNet_results" -path "*Dataset423*" -name "checkpoint_best.pth" | sort
find "$nnUNet_results" -path "*Dataset412*" -name "checkpoint_best.pth" | sort
```

## 15. 推理 Validation-Public

`Validation-Public` 没有金标准，所以只做可视化检查，不计算 Dice/HD95。

推理：

```bash
cd ~/STS2024Task2-nnUnet-main
mkdir -p logs

export QUADRANT_DATASET_ID=423
export TOOTH_DATASET_ID=412
export QUADRANT_CONFIG=3d_lowres
export TOOTH_CONFIG=3d_fullres
export FOLD="0 1 2 3 4"
export QUADRANT_CHECKPOINT=checkpoint_final.pth
export TOOTH_CHECKPOINT=checkpoint_final.pth
export OUTPUT_LABEL_SCHEME=fdi
export DISABLE_TTA=0

CUDA_VISIBLE_DEVICES=3 bash scripts/predict_v2.sh \
  data/Validation-Public \
  runs/val_pred_423_412_final_tta \
  2>&1 | tee logs/predict_val_423_412_final_tta.log
```

输出：

```text
runs/val_pred_423_412_final_tta/final
```

检查数量：

```bash
find runs/val_pred_423_412_final_tta/final -name "*.nii.gz" | wc -l
```

## 16. 推理有金标准测试集

测试集目录：

```text
data/dental_CBCT_test_set/images
data/dental_CBCT_test_set/labels
```

推理 `checkpoint_final.pth + TTA`：

```bash
cd ~/STS2024Task2-nnUnet-main
mkdir -p logs

export QUADRANT_DATASET_ID=423
export TOOTH_DATASET_ID=412
export QUADRANT_CONFIG=3d_lowres
export TOOTH_CONFIG=3d_fullres
export FOLD="0 1 2 3 4"
export QUADRANT_CHECKPOINT=checkpoint_final.pth
export TOOTH_CHECKPOINT=checkpoint_final.pth
export OUTPUT_LABEL_SCHEME=fdi
export DISABLE_TTA=0

CUDA_VISIBLE_DEVICES=2 bash scripts/predict_v2.sh \
  data/dental_CBCT_test_set/images \
  runs/test_pred_423_412_final_tta \
  2>&1 | tee logs/predict_test_423_412_final_tta.log
```

推理 `checkpoint_best.pth + TTA`：

```bash
export QUADRANT_CHECKPOINT=checkpoint_best.pth
export TOOTH_CHECKPOINT=checkpoint_best.pth
export DISABLE_TTA=0

CUDA_VISIBLE_DEVICES=3 bash scripts/predict_v2.sh \
  data/dental_CBCT_test_set/images \
  runs/test_pred_423_412_best_tta \
  2>&1 | tee logs/predict_test_423_412_best_tta.log
```

检查预测数量：

```bash
find runs/test_pred_423_412_final_tta/final -name "*.nii.gz" | wc -l
find runs/test_pred_423_412_best_tta/final -name "*.nii.gz" | wc -l
find data/dental_CBCT_test_set/labels -name "*.nii.gz" | wc -l
```

## 17. 计算测试集 Dice 和 HD95

评价 `final_tta`：

```bash
python process/evaluate_segmentation_metrics.py \
  --pred-dir runs/test_pred_423_412_final_tta/final \
  --gt-dir data/dental_CBCT_test_set/labels \
  --output-csv runs/eval_test_423_412_final_tta/per_case_metrics.csv \
  --summary-csv runs/eval_test_423_412_final_tta/summary_metrics.csv \
  --label-mode fdi
```

查看结果：

```bash
cat runs/eval_test_423_412_final_tta/summary_metrics.csv
```

当前结果：

```text
scope,label,n,mean_dice,mean_hd95_mm
all_teeth,mean,1514,0.9562,0.5754
foreground,foreground,50,0.9684,0.2645
```

评价 `best_tta`：

```bash
python process/evaluate_segmentation_metrics.py \
  --pred-dir runs/test_pred_423_412_best_tta/final \
  --gt-dir data/dental_CBCT_test_set/labels \
  --output-csv runs/eval_test_423_412_best_tta/per_case_metrics.csv \
  --summary-csv runs/eval_test_423_412_best_tta/summary_metrics.csv \
  --label-mode fdi
```

当前结果：

```text
scope,label,n,mean_dice,mean_hd95_mm
all_teeth,mean,1514,0.9246523388145589,1.549844420393415
foreground,foreground,50,0.9677140146969152,0.26863960921764374
```

评价指标解释：

```text
all_teeth
  逐牙标签分别计算 Dice/HD95 后求平均，更适合作为实例分割主指标。

foreground
  把所有牙齿标签合并成一个前景后计算 Dice/HD95，更适合反映整体牙齿区域分割质量。
```

论文主结果建议使用 `final_tta`。

## 18. 历史预测结果的 FDI 编号修正

当前版本的 `pipeline/quadrant_merge.py` 已经输出标准 FDI：

```text
1 -> 11-18
2 -> 21-28
3 -> 31-38
4 -> 41-48
```

如果你是用当前版本重新推理测试集，不需要执行本节。

如果你已经用早期版本生成了预测，可能出现：

```text
foreground Dice 很高
all_teeth Dice 接近 0
```

这通常说明空间分割正确，但 FDI 左右象限编号交换。可用下面命令修正旧预测：

```bash
python process/remap_legacy_fdi_quadrants.py \
  --input-dir runs/test_pred_423_412_final_tta/final \
  --output-dir runs/test_pred_423_412_final_tta/final_fdi_standard \
  --overwrite
```

然后评价修正后的目录：

```bash
python process/evaluate_segmentation_metrics.py \
  --pred-dir runs/test_pred_423_412_final_tta/final_fdi_standard \
  --gt-dir data/dental_CBCT_test_set/labels \
  --output-csv runs/eval_test_423_412_final_tta_fixed/per_case_metrics.csv \
  --summary-csv runs/eval_test_423_412_final_tta_fixed/summary_metrics.csv \
  --label-mode fdi
```

## 19. 推理 Train-Unlabeled 生成伪标签

如果要继续做半监督迭代，可对 `Train-Unlabeled` 推理。

推荐分别跑 `final_tta` 和 `best_tta`，再做一致性筛选。

推理 final：

```bash
export QUADRANT_DATASET_ID=423
export TOOTH_DATASET_ID=412
export QUADRANT_CONFIG=3d_lowres
export TOOTH_CONFIG=3d_fullres
export FOLD="0 1 2 3 4"
export QUADRANT_CHECKPOINT=checkpoint_final.pth
export TOOTH_CHECKPOINT=checkpoint_final.pth
export OUTPUT_LABEL_SCHEME=fdi
export DISABLE_TTA=0

CUDA_VISIBLE_DEVICES=4 bash scripts/predict_v2.sh \
  data/Train-Unlabeled \
  runs/pseudo_teacher_423_412_final_tta \
  2>&1 | tee logs/predict_unlabeled_423_412_final_tta.log
```

推理 best：

```bash
export QUADRANT_CHECKPOINT=checkpoint_best.pth
export TOOTH_CHECKPOINT=checkpoint_best.pth
export DISABLE_TTA=0

CUDA_VISIBLE_DEVICES=5 bash scripts/predict_v2.sh \
  data/Train-Unlabeled \
  runs/pseudo_teacher_423_412_best_tta \
  2>&1 | tee logs/predict_unlabeled_423_412_best_tta.log
```

一致性筛选：

```bash
python process/select_pseudo_dice.py \
  --prediction-dirs \
    runs/pseudo_teacher_423_412_best_tta/final \
    runs/pseudo_teacher_423_412_final_tta/final \
  --output-dir runs/pseudo_selected_423_412_tta_thr095 \
  --threshold 0.95 \
  --report-csv runs/pseudo_selected_423_412_tta_thr095_report.csv
```

检查筛选数量：

```bash
find runs/pseudo_selected_423_412_tta_thr095 -name "*.nii.gz" | wc -l
```

筛出的伪标签仍建议用 3D Slicer 人工检查，删除明显错误病例后再进入下一轮训练。

## 20. 论文可视化

推荐使用有金标准的测试集：

```text
data/dental_CBCT_test_set/images
data/dental_CBCT_test_set/labels
runs/test_pred_423_412_final_tta/final
```

推荐展示方式：

```text
CT 原图
Ground Truth overlay
Prediction overlay
3D surface
```

建议选择：

```text
2-3 个效果好的病例
1 个中等难度病例
1 个失败或困难病例
```

3D Slicer 5.8.1 可用于制作可视化图：

```text
1. 加载 CBCT 图像。
2. 加载 GT label。
3. 加载 prediction label。
4. 转为 Segmentation。
5. 开启 3D 显示。
6. 调整颜色、透明度和视角。
7. 截图保存。
```

`Validation-Public` 没有金标准，只能展示预测结果，不能作为论文定量指标。

## 21. 最终论文结果记录模板

```text
Method:
Two-stage nnU-Net v2 pipeline.
Stage 1: Dataset423, 3d_lowres, 5-fold ensemble.
Stage 2: Dataset412, 3d_fullres, 5-fold ensemble.
Inference: checkpoint_final.pth, TTA enabled.

Test set:
50 labeled dental CBCT scans.

Metrics:
Instance-level mean Dice: 0.9562
Instance-level mean HD95: 0.5754 mm
Foreground Dice: 0.9684
Foreground HD95: 0.2645 mm
```

## 22. 复现顺序总览

```text
1. 安装 Miniconda。
2. 创建 nnunetv2 环境。
3. 安装 PyTorch 和 requirements-linux.txt。
4. 固定 nnU-Net v2 环境变量。
5. 检查 data 目录和标签值。
6. 生成 Dataset412 和 Dataset413。
7. 生成 Dataset423。
8. 预处理 Dataset423 和 Dataset412。
9. 训练 Dataset423 3d_lowres 五折。
10. 训练 Dataset412 3d_fullres 五折。
11. 使用 423 + 412 推理 dental_CBCT_test_set。
12. 计算 Dice 和 HD95。
13. 使用 3D Slicer 制作论文可视化。
14. 如需继续半监督迭代，再推理 Train-Unlabeled 并筛选伪标签。
```
