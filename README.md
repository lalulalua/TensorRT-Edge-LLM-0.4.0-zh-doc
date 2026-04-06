> **修改说明：** 本文件在基于 NVIDIA TensorRT-Edge-LLM 0.4.0 发行源码整理的本仓库中已作调整。详见 [PROJECT_ORIGIN.md](PROJECT_ORIGIN.md)。英文说明见 [README_EN.md](README_EN.md)。

# TensorRT Edge-LLM

**面向 NVIDIA 边缘平台的高性能大型语言模型推理框架**

---

## 关于本仓库

本仓库是基于上游项目 [NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) **0.4.0** 发行版源码整理的**非官方**修改分发，**不由 NVIDIA 维护**，也不代表 NVIDIA 背书或提供官方支持。后续针对**本 Git 仓库**的维护、改动与发布由当前仓库维护者负责。上游及第三方原有的版权与许可信息予以保留；许可仍以 [LICENSE](LICENSE) 为准，并请参阅 [NOTICE](NOTICE)、[PROJECT_ORIGIN.md](PROJECT_ORIGIN.md)。

---

## 概览

TensorRT Edge-LLM 是 NVIDIA 面向嵌入式平台的大型语言模型（LLM）与视觉语言模型（VLM）的高性能 C++ 推理运行时，支持在 NVIDIA Jetson、NVIDIA DRIVE 等资源受限设备上高效部署前沿语言模型。项目提供便捷的 Python 脚本，将 HuggingFace 检查点转换为 [ONNX](https://onnx.ai)。引擎构建与端到端推理均在边缘平台上完成。

---

## 快速开始

支持的平台、模型与精度见 [**概览**](docs_zh/source/developer_guide/01.1_Overview.md)。约 15 分钟内可上手。完整安装与使用说明见 [**快速入门**](docs_zh/source/developer_guide/01.2_Quick_Start_Guide.md)。

---

## 文档

### 开发者指南

安装、使用与部署的完整说明：

- **[概览](docs_zh/source/developer_guide/01.1_Overview.md)** — TensorRT Edge-LLM 是什么及主要特性
- **[快速入门](docs_zh/source/developer_guide/01.2_Quick_Start_Guide.md)** — 约 15 分钟跑通流程
- **[安装](docs_zh/source/developer_guide/01.3_Installation.md)** — 详细安装说明
- **[支持的模型](docs_zh/source/developer_guide/02_Supported_Models.md)** — 完整模型与精度兼容表
- **[Python 导出流水线](docs_zh/source/developer_guide/03.1_Python_Export_Pipeline.md)** — 模型导出与量化
- **[引擎构建器](docs_zh/source/developer_guide/03.2_Engine_Builder.md)** — 构建 TensorRT 引擎
- **[C++ 运行时概览](docs_zh/source/developer_guide/04.1_C++_Runtime_Overview.md)** — 运行时系统架构
  - [LLM 推理运行时](docs_zh/source/developer_guide/04.2_LLM_Inference_Runtime.md)
  - [LLM SpecDecode 运行时](docs_zh/source/developer_guide/04.3_LLM_Inference_SpecDecode_Runtime.md)
  - [高级运行时特性](docs_zh/source/developer_guide/04.4_Advanced_Runtime_Features.md)
- **[示例](docs_zh/source/developer_guide/05_Examples.md)** — 可运行示例
- **[对话模板格式](docs_zh/source/developer_guide/06_Chat_Template_Format.md)** — 对话模板配置
- **[定制指南](docs_zh/source/developer_guide/07_Customization_Guide.md)** — 扩展与定制框架
- **[TensorRT 插件](docs_zh/source/developer_guide/08_TensorRT_Plugins.md)** — TensorRT 插件说明

> **与英文文档对照：** 英文索引见 [README_EN.md](README_EN.md)。部分上游英文文档中「TensorRT 插件」曾链接到不存在的 `07_TensorRT_Plugins.md`；仓库内实际文件为 `08_TensorRT_Plugins.md`，且包含 `07_Customization_Guide.md`；本仓库的 [README_EN.md](README_EN.md) 已按实际路径整理链接。

### 其他资源

- **[示例目录](examples/)** — LLM、VLM 推理与精度评估相关示例；各子模块的**中文**构建说明、依赖与数据流见 [examples/README.md](examples/README.md)（其下 [llm](examples/llm/README.md)、[multimodal](examples/multimodal/README.md)、[utils](examples/utils/README.md)、[accuracy](examples/accuracy/README.md) 等子目录亦有独立说明；与上游英文对照可参阅 [accuracy 目录内 README_en.md](examples/accuracy/README_en.md) 等文件）。
- **[测试](tests/)** — 贡献者可用的完整测试集

---

## 适用场景

**车载**

- 车内 AI 助手
- 语音控制界面
- 场景理解
- 驾驶辅助

**机器人**

- 自然语言交互
- 任务规划与推理
- 视觉问答
- 人机协作

**工业物联网**

- 基于 NLP 的设备监控
- 自动化检测
- 预测性维护
- 语音控制设备

**边缘设备**

- 端侧聊天机器人
- 离线语言处理
- 隐私保护型 AI
- 低延迟推理

---

## 技术博客

*敬请期待*

后续将发布技术深读、优化指南与部署最佳实践。

---

## 最新动态

*敬请期待*

**NVIDIA 上游项目**的更新与发行说明见 [NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)。若您关注的是**本非官方仓库**的变更，请使用您获取本仓库时所使用的 Git 托管平台上的 Issue 与 Release 信息。

---

## 支持

- **文档**：[开发者指南](docs_zh/source/developer_guide/01.1_Overview.md)
- **上游 NVIDIA 项目**（官方来源）：[GitHub Issues](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues)、[Discussions](https://github.com/NVIDIA/TensorRT-Edge-LLM/discussions)
- **本非官方仓库**：与**本分发**相关的问题，请使用当前仓库所在 Git 托管服务的 Issue（请勿默认当作 NVIDIA 官方支持渠道）。
- **论坛**（一般性 NVIDIA 技术讨论）：[NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

---

## 许可

[Apache License 2.0](LICENSE)

---

## 仓库贡献与修改说明

本 Git 仓库在保留上游 [NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) 0.4.0 发行版结构与许可的前提下，对**学习与维护向**可读性做了增补，主要包括：`examples/` 下各示例子目录的**纯中文** [README.md](examples/README.md) 体系（含 CMake 目标说明、依赖关系与 **Mermaid** 数据流/架构图）、以及对部分示例 **C++/Python** 源码中与 **engine 文件**加载、激活缓存（Activation Buffer）、设备侧数据流相关段落的中文注释。上述增补**不替代**官方开发者指南；部署与排错请以 [文档](#文档) 中的开发者指南及上游仓库为准。

**本仓库的中文文档重构、逻辑流程图绘制及深度代码注释，均由 Cursor 自动编程辅助生成。** 读者应结合源码与官方文档自行核对；发现错误或过时之处，欢迎通过本仓库 Issue 或 PR 反馈。

---

## 贡献

欢迎贡献。请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)，其中说明了如何区分向上游 NVIDIA 项目提交与向本仓库提交。

---
