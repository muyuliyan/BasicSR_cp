# HSI 超光谱图像数据集准备指南

[English Version](HSI_DatasetPreparation.md)

本指南提供了使用 BasicSR 进行超光谱图像(HSI)超分辨率实验的数据集准备说明。

## 数据集结构

按以下结构组织您的 HSI 数据集：

```
datasets/
├── your_hsi_dataset/
│   ├── HR/                 # 高分辨率图像
│   │   ├── image001.mat    # 或 .npy
│   │   ├── image002.mat
│   │   └── ...
│   ├── LR/                 # 低分辨率图像（生成的）
│   │   ├── image001.mat    # 或 .npy
│   │   ├── image002.mat
│   │   └── ...
│   ├── val/               # 验证集（可选）
│   │   ├── HR/
│   │   └── LR/
│   └── test/              # 测试集
│       ├── HR/
│       └── LR/
```

## 支持的文件格式

### MATLAB 文件 (.mat)
- 数据应存储在键 `'gt'`、`'data'` 或任何自定义键中
- 形状：`[H, W, C]`，其中 C 是光谱波段数
- MATLAB 加载示例：`data = load('image.mat'); hsi = data.gt;`

### NumPy 文件 (.npy)
- 直接 numpy 数组存储
- 形状：`[H, W, C]`，其中 C 是光谱波段数
- Python 加载示例：`hsi = np.load('image.npy')`

## 数据预处理

### 步骤1：准备高分辨率(HR)数据

将原始 HSI 数据放置在 `HR` 文件夹中。确保所有图像具有相同的光谱波段数。

### 步骤2：生成低分辨率(LR)数据

使用提供的预处理脚本通过双三次下采样生成 LR 数据：

```bash
# 对于 MATLAB 文件
python scripts/hsi_bicubic_preprocessing.py \
    --input_folder datasets/your_hsi_dataset/HR \
    --output_folder datasets/your_hsi_dataset/LR \
    --scale 4 \
    --file_format mat \
    --data_key gt

# 对于 NumPy 文件
python scripts/hsi_bicubic_preprocessing.py \
    --input_folder datasets/your_hsi_dataset/HR \
    --output_folder datasets/your_hsi_dataset/LR \
    --scale 4 \
    --file_format npy
```

参数说明：
- `--input_folder`：HR 图像路径
- `--output_folder`：保存 LR 图像的路径
- `--scale`：下采样因子（通常为2、3或4）
- `--file_format`：文件格式（`mat` 或 `npy`）
- `--data_key`：MATLAB 文件的键（默认：`data`）

### 步骤3：验证数据

预处理后，验证以下内容：
1. HR 和 LR 文件夹包含相同数量的文件
2. HR 和 LR 之间文件名匹配
3. LR 图像尺寸为 HR 尺寸除以缩放因子
4. 所有图像具有相同的光谱波段数

## 配置

### 训练配置

修改训练配置文件 `options/train/HSI/train_HSI_SRResNet_x4.yml`：

1. **更新数据集路径**：
   ```yaml
   datasets:
     train:
       dataroot_gt: datasets/your_hsi_dataset/HR
       dataroot_lq: datasets/your_hsi_dataset/LR
   ```

2. **设置光谱通道数**：
   ```yaml
   network_g:
     num_in_ch: 31   # 替换为您的光谱波段数
     num_out_ch: 31  # 应与 num_in_ch 匹配
   ```

3. **根据 GPU 内存调整批大小和补丁大小**：
   ```yaml
   datasets:
     train:
       gt_size: 64           # 补丁大小
       batch_size_per_gpu: 8 # 批大小
   ```

### 测试配置

修改测试配置文件 `options/test/HSI/test_HSI_SRResNet_x4.yml`：

1. **更新数据集路径**：
   ```yaml
   datasets:
     test_1:
       dataroot_gt: datasets/your_hsi_dataset/test/HR
       dataroot_lq: datasets/your_hsi_dataset/test/LR
   ```

2. **设置模型路径**：
   ```yaml
   path:
     pretrain_network_g: experiments/pretrained_models/your_model.pth
   ```

## 常见 HSI 数据集

### CAVE 数据集
- **光谱波段**：31 (400-700nm)
- **空间分辨率**：各种（主要是512×512）
- **下载**：[Columbia CAVE dataset](http://www1.cs.columbia.edu/CAVE/databases/multispectral/)

### Harvard 数据集
- **光谱波段**：31 (400-700nm)
- **空间分辨率**：各种
- **下载**：[Harvard dataset](http://vision.seas.harvard.edu/hyperspec/explore.html)

### ICVL 数据集
- **光谱波段**：31 (400-700nm)
- **空间分辨率**：各种
- **下载**：[ICVL dataset](https://icvl.cs.bgu.ac.il/)

### Pavia University/Centre
- **光谱波段**：103/102
- **空间分辨率**：610×340 / 1096×715
- **下载**：[Grupo de Inteligencia Computacional](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

## 训练

使用以下命令开始训练：

```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/train.py \
    -opt options/train/HSI/train_HSI_SRResNet_x4.yml \
    --auto_resume
```

## 测试

测试训练好的模型：

```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py \
    -opt options/test/HSI/test_HSI_SRResNet_x4.yml
```

## 评价指标

HSI 配置包括五个用于超光谱图像评估的关键指标：

1. **PSNR**（峰值信噪比）：整体图像质量
2. **SSIM**（结构相似性指数）：结构相似性
3. **SAM**（光谱角映射）：光谱相似性（HSI专用）
4. **ERGAS**（全局相对误差）：全局相对误差（HSI专用）
5. **RMSE**（均方根误差）：像素级误差

## 技巧和最佳实践

1. **内存管理**：HSI 数据需要更多内存。如果遇到内存不足错误，请减少批大小和补丁大小。

2. **光谱归一化**：考虑将光谱值归一化到 [0, 1] 范围以获得更好的训练稳定性。

3. **数据增强**：HSI 数据集支持水平翻转和旋转。如果光谱顺序重要，请禁用。

4. **验证**：使用单独的验证集监控训练进度并防止过拟合。

5. **模型选择**：从 SRResNet 架构开始，尝试其他模型如 RCAN 或 EDSR 以获得更好的性能。

## 故障排除

### 常见问题

1. **形状不匹配错误**：确保所有 HSI 图像具有相同的光谱波段数
2. **文件格式错误**：验证 .mat 文件的文件格式和数据键
3. **内存错误**：减少批大小和补丁大小
4. **路径错误**：检查配置文件中的所有数据集路径是否正确

### 调试命令

```bash
# 检查 HSI 数据形状和格式
python -c "
import scipy.io as sio
import numpy as np
data = sio.loadmat('path/to/your/file.mat')
print('Keys:', list(data.keys()))
print('Shape:', data['your_key'].shape)
print('Data type:', data['your_key'].dtype)
"
```