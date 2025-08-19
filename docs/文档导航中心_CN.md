# BasicSR 文档导航中心

欢迎使用 BasicSR！本页面为您提供完整的文档导航，帮助您快速找到所需信息。

## 🎯 根据您的需求选择

### 🆕 初学者用户
- **立即开始** → [常见任务一键配置指南](常见任务一键配置指南_CN.md)
- **系统学习** → [BasicSR 使用指南 (中文)](BasicSR_使用指南_CN.md)
- **环境搭建** → [安装指南](INSTALL.md)

### 🔧 配置开发者
- **快速配置** → [任务配置快速参考](任务配置快速参考_CN.md)
- **详细说明** → [配置文件说明](Config.md)
- **样例配置** → `options/` 目录下的配置文件

### 🔬 研究人员
- **完整指南** → [BasicSR 使用指南 (English)](BasicSR_Usage_Guide.md)
- **模型库** → [ModelZoo](ModelZoo_CN.md)
- **设计约定** → [设计约定](DesignConvention_CN.md)

### 🚀 高级用户
- **HOWTOs** → [HOWTOs 指南](HOWTOs_CN.md)
- **指标评估** → [评估指标](Metrics_CN.md)
- **日志系统** → [Logging](Logging_CN.md)

---

## 📚 完整文档目录

### 🎪 新增综合指南 (推荐)
| 文档 | 描述 | 适用人群 |
|------|------|----------|
| [BasicSR 使用指南 (中文)](BasicSR_使用指南_CN.md) | 涵盖超分、去噪、修复任务的完整流程 | 所有用户 |
| [BasicSR Usage Guide (English)](BasicSR_Usage_Guide.md) | Complete workflow for SR, denoising, and inpainting | All users |
| [常见任务一键配置指南](常见任务一键配置指南_CN.md) | 三种常见任务的详细配置步骤 | 初学者 |
| [任务配置快速参考](任务配置快速参考_CN.md) | 各种任务的配置模板和参数说明 | 开发者 |

### 🛠️ 基础设置文档
| 文档 | 描述 |
|------|------|
| [INSTALL.md](INSTALL.md) | 环境安装和依赖配置 |
| [DatasetPreparation_CN.md](DatasetPreparation_CN.md) | 数据准备详细说明 |
| [TrainTest_CN.md](TrainTest_CN.md) | 训练测试命令 |
| [Config.md](Config.md) | 配置文件详细说明 |

### 🏗️ 架构和模型
| 文档 | 描述 |
|------|------|
| [Models_CN.md](Models_CN.md) | 支持的模型描述 |
| [ModelZoo_CN.md](ModelZoo_CN.md) | 预训练模型和实验结果 |
| [DesignConvention_CN.md](DesignConvention_CN.md) | 代码库设计约定 |

### 🎯 任务专项指南
| 文档 | 描述 |
|------|------|
| [HOWTOs_CN.md](HOWTOs_CN.md) | 各种任务的具体操作方法 |
| [HSI_DatasetPreparation_CN.md](HSI_DatasetPreparation_CN.md) | 超光谱图像数据准备 |

### 📊 评估和监控
| 文档 | 描述 |
|------|------|
| [Metrics_CN.md](Metrics_CN.md) | 支持的评估指标 |
| [Logging_CN.md](Logging_CN.md) | 日志系统使用 |

---

## 🔍 按任务类型快速导航

### 🖼️ 图像超分辨率 (Super-Resolution)
```
1. 新手入门 → 常见任务一键配置指南 > 任务一：超分辨率
2. 详细配置 → 任务配置快速参考 > 超分辨率任务配置  
3. 完整流程 → BasicSR使用指南 > 超分辨率任务
4. 命令参考 → TrainTest_CN.md
```

**推荐架构**: SRResNet (入门) → EDSR (性能) → SwinIR (最新)

### 🔇 图像去噪 (Denoising)
```
1. 新手入门 → 常见任务一键配置指南 > 任务二：图像去噪
2. 详细配置 → 任务配置快速参考 > 去噪任务配置
3. 完整流程 → BasicSR使用指南 > 去噪任务
```

**推荐架构**: RIDNet (专用) → SwinIR (通用)

### 🖼️ 图像修复 (Restoration/Inpainting)
```
1. 新手入门 → 常见任务一键配置指南 > 任务三：JPEG压缩伪影去除
2. 详细配置 → 任务配置快速参考 > 修复任务配置
3. 完整流程 → BasicSR使用指南 > 修复任务
```

**推荐架构**: SwinIR (JPEG修复) → DFDNet (人脸修复)

---

## 📖 按学习路径导航

### 🌟 快速上手路径 (1-2小时)
1. [INSTALL.md](INSTALL.md) - 安装环境
2. [常见任务一键配置指南](常见任务一键配置指南_CN.md) - 选择一个任务跟做
3. [TrainTest_CN.md](TrainTest_CN.md) - 学习基本命令

### 📚 系统学习路径 (1-2天)
1. [BasicSR 使用指南](BasicSR_使用指南_CN.md) - 完整阅读
2. [Config.md](Config.md) - 深入理解配置
3. [任务配置快速参考](任务配置快速参考_CN.md) - 掌握各种配置
4. [HOWTOs_CN.md](HOWTOs_CN.md) - 学习高级技巧

### 🔬 深度研究路径 (1周+)
1. [Models_CN.md](Models_CN.md) - 理解模型架构
2. [DesignConvention_CN.md](DesignConvention_CN.md) - 了解设计理念
3. [ModelZoo_CN.md](ModelZoo_CN.md) - 研究预训练模型
4. 源码阅读 `basicsr/` 目录

---

## 🆘 问题解决指南

### ❓ 常见问题
1. **环境安装问题** → [INSTALL.md](INSTALL.md) + [FAQ.md](FAQ.md)
2. **配置文件错误** → [Config.md](Config.md) + [任务配置快速参考](任务配置快速参考_CN.md)
3. **训练不收敛** → [BasicSR 使用指南](BasicSR_使用指南_CN.md) > 常见问题解决
4. **显存不足** → [常见任务一键配置指南](常见任务一键配置指南_CN.md) > 常见问题解决

### 🔧 调试流程
```
1. 检查配置文件语法 → Config.md
2. 验证数据路径 → DatasetPreparation_CN.md  
3. 查看训练日志 → Logging_CN.md
4. 调整超参数 → 任务配置快速参考
```

### 💡 获取帮助
- **GitHub Issues**: [提交问题](https://github.com/XPixelGroup/BasicSR/issues)
- **QQ群**: 320960100 (答案: 互帮互助共同进步)
- **Discussions**: [参与讨论](https://github.com/XPixelGroup/BasicSR/discussions)

---

## 📝 文档贡献

如果您发现文档中的错误或有改进建议：

1. **提交Issue**: 在GitHub上创建issue描述问题
2. **Pull Request**: 直接提交文档修改
3. **反馈建议**: 通过QQ群或Discussions提供建议

---

## 🏷️ 文档版本信息

- **最后更新**: 2024年
- **版本**: BasicSR 1.4+
- **语言**: 中文/English
- **维护者**: BasicSR开发团队

---

## 📄 许可证

本文档遵循 Apache 2.0 许可证。详见 [LICENSE](../LICENSE.txt)。

---

**快速开始建议**: 如果您是第一次使用BasicSR，建议从[常见任务一键配置指南](常见任务一键配置指南_CN.md)开始！