# TensorRT Edge-LLM

**面向 NVIDIA 边缘平台的高性能大型语言模型推理框架**

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

> **与英文主页对照说明：** 英文根目录 `README.md` 中「TensorRT 插件」曾链接到不存在的 `07_TensorRT_Plugins.md`；仓库内实际文件为 `08_TensorRT_Plugins.md`，且未单独列出 `07_Customization_Guide.md`。中文版已按实际文件命名与目录补全上述链接。

### 其他资源

- **[示例目录](examples/)** — LLM 与 VLM 推理示例
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

关注 [GitHub 仓库](https://github.com/NVIDIA/TensorRT-Edge-LLM) 获取更新、发行说明与公告。

---

## 支持

- **文档**：[开发者指南](docs_zh/source/developer_guide/01.1_Overview.md)
- **问题反馈**：[GitHub Issues](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues)
- **讨论**：[GitHub Discussions](https://github.com/NVIDIA/TensorRT-Edge-LLM/discussions)
- **论坛**：[NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

---

## 许可

[Apache License 2.0](LICENSE)

---

## 贡献

欢迎贡献代码。请参阅 [贡献指南](CONTRIBUTING.md)。

---
