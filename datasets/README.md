# Symlink/Put all the datasets here

It is recommended to symlink your dataset root to this folder - `datasets` with the command `ln -s xxx yyy`.

Please refer to [DatasetPreparation.md](../docs/DatasetPreparation.md) for more details about data preparation.

## HSI (Hyperspectral Image) Datasets

For HSI super-resolution experiments, see:
- **Setup Guide**: [HSI_DatasetPreparation.md](../docs/HSI_DatasetPreparation.md)
- **Quick Start**: [HSI_QuickStart.md](../HSI_QuickStart.md)
- **Example Structure**: `example_hsi_dataset/`

### Supported HSI Datasets

| Dataset | Bands | Resolution | Download Link |
|---------|-------|------------|---------------|
| CAVE | 31 | Various | [Columbia CAVE](http://www1.cs.columbia.edu/CAVE/databases/multispectral/) |
| Harvard | 31 | Various | [Harvard](http://vision.seas.harvard.edu/hyperspec/explore.html) |
| ICVL | 31 | Various | [ICVL](https://icvl.cs.bgu.ac.il/) |
| Pavia University | 103 | 610×340 | [Pavia](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |

### Quick HSI Setup

```bash
# 1. Create dataset structure
mkdir -p datasets/my_hsi_dataset/{HR,LR,val/{HR,LR},test/{HR,LR}}

# 2. Copy your HR data to datasets/my_hsi_dataset/HR/

# 3. Generate LR data
python scripts/hsi_bicubic_preprocessing.py \
    --input_folder datasets/my_hsi_dataset/HR \
    --output_folder datasets/my_hsi_dataset/LR \
    --scale 4 --file_format mat
```

---

推荐把数据通过 `ln -s xxx yyy` 软链到当前目录 `datasets` 下.

更多数据准备的细节参见 [DatasetPreparation_CN.md](../docs/DatasetPreparation_CN.md).

## HSI (超光谱图像) 数据集

HSI 超分辨率实验请参考:
- **设置指南**: [HSI_DatasetPreparation_CN.md](../docs/HSI_DatasetPreparation_CN.md)
- **快速开始**: [HSI_QuickStart.md](../HSI_QuickStart.md)
- **示例结构**: `example_hsi_dataset/`
