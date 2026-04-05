# 支持的模型

> **代码位置：** `tensorrt_edgellm/`（导出）、`cpp/`（运行时）

## 大型语言模型（LLM）

### Llama 家族

| 模型 | 参数量 | FP16 | FP8 | INT4 | NVFP4 |
|-------|-----------|------|-----|------|-------|
| [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | 8B | ✅ | ✅ | ✅ | ✅ |
| [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | 8B | ✅ | ✅ | ✅ | ✅ |
| [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | 3B | ✅ | ✅ | ✅ | ✅ |

### Qwen2/2.5 家族

| 模型 | 参数量 | FP16 | FP8 | INT4 | NVFP4 |
|-------|-----------|------|-----|------|-------|
| [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) | 0.5B | ✅ | ✅ | ✅ | ✅ |
| [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) | 1.5B | ✅ | ✅ | ✅ | ✅ |
| [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | 7B | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 0.5B | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1.5B | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 3B | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 7B | ✅ | ✅ | ✅ | ✅ |

### Qwen3 家族

| 模型 | 参数量 | FP16 | FP8 | INT4 | NVFP4 |
|-------|-----------|------|-----|------|-------|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 0.6B | ✅ | ✅ | ✅ | ✅ |
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 4B | ✅ | ✅ | ✅ | ✅ |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 8B | ✅ | ✅ | ✅ | ✅ |


### DeepSeek-R1 Distilled 家族

| 模型 | 参数量 | FP16 | FP8 | INT4 | NVFP4 |
|-------|-----------|------|-----|------|-------|
| [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | 1.5B | ✅ | ✅ | ✅ | ✅ |
| [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | 7B | ✅ | ✅ | ✅ | ✅ |

---

## 视觉语言模型（VLM）

| 模型 | 参数量 | FP16 | FP8 | INT4 | NVFP4 |
|-------|-----------|------|-----|------|-------|
| [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | 2B | ✅ | ✅ | ✅ | ✅ |
| [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) | 7B | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 3B | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 7B | ✅ | ✅ | ✅ | ✅ |
| [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | 2B | ✅ | ✅ | ✅ | ✅ |
| [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | 4B | ✅ | ✅ | ✅ | ✅ |
| [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 8B | ✅ | ✅ | ✅ | ✅ |
| [InternVL3-1B](https://huggingface.co/OpenGVLab/InternVL3-1B-hf) | 1B | ✅ | ✅ | ✅ | ✅ |
| [InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B-hf) | 2B | ✅ | ✅ | ✅ | ✅ |
| [Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | 5.6B | ✅ | ✅ | ✅ | ✅ |

---

## 精度支持

| 精度 | 显存 | 计算 | 平台要求 | 适用场景 |
|-----------|--------|---------|----------------------|----------|
| **FP16** | 1×（基准） | FP16 | 全平台 | 精度基准、通用兼容 |
| **FP8** | 约减半 | FP8 GEMM + FP16 | **SM89+**（Ada Lovelace 及更新） | 在新 GPU 上平衡性能 |
| **INT4 AWQ** | 约 1/4 | FP16（AWQ 量化） | 全平台 | 显存受限设备 |
| **INT4 GPTQ** | 约 1/4 | FP16（GPTQ 量化） | 全平台 | 显存受限设备 |
| **NVFP4** | 约 1/4 | NVFP4 GEMM + FP16 | **SM100+**（Blackwell 及更新） | **Thor 平台（推荐）** |

### 其他特性

- **FP8 视觉编码器**：视觉模型（Qwen2-VL、InternVL3）在 SM89+ 上支持
- **FP8/NVFP4 LM Head**：语言模型头按平台要求支持

---

## 平台兼容性

| GPU 架构 | 计算能力 | 支持的精度 |
|-----------------|-------------------|---------------------|
| **全平台** | 任意 | FP16、INT4 AWQ、INT4 GPTQ |
| **Ada Lovelace+** | SM89+ | FP16、FP8、INT4 AWQ、INT4 GPTQ |
| **Blackwell+** | SM100+ | FP16、FP8、INT4 AWQ、INT4 GPTQ、NVFP4 |

**说明：**

- FP16 与 INT4（AWQ/GPTQ）可在所有支持 CUDA 的平台上使用
- FP8 量化需要 SM89+（Ada Lovelace 或更新架构，例如 RTX 40 系列）
- NVFP4 量化需要 SM100+（Blackwell 或更新架构，例如 Thor 平台）
- 平台要求同时适用于模型权重与算子（含 ViT 编码器与 LM head）

**开发用 GPU：**

开发阶段，TensorRT Edge-LLM 支持下列独立 GPU 计算能力：

- **SM80**：Ampere（如 A100、A30、A10）
- **SM86**：Ampere（如 RTX 30 系列、RTX Pro Ampere）
- **SM89**：Ada Lovelace（如 RTX 40 系列、L4、L40、RTX Pro Ada）
- **SM100**：Blackwell（如 GB200）
- **SM120**：Blackwell（如 RTX 50 系列、RTX Pro Blackwell）

> **说明：** 上述 GPU 可用于开发与测试；官方部署平台为 NVIDIA Jetson Thor（JetPack 7.1）与 NVIDIA DRIVE Thor（DriveOS 7）。若需在这些 GPU 上获得更高性能推理方案，请参阅 [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)。

---

## 更多资料

- [概览](01.1_Overview.md)
- [快速入门](01.2_Quick_Start_Guide.md)
- [示例](05_Examples.md)
