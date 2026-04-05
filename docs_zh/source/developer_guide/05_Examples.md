# 示例

> **代码位置：** `examples/` | **构建输出：** `examples/llm/`、`examples/multimodal/`

## 概览

用于构建引擎与运行推理的 C++ 示例。各示例在 `examples/` 中附带 README 与源码。

> **⚠️ 用户责任**：请针对自身场景编写合理、恰当的提示。示例仅展示技术用法，不保证输出质量或适用性。

---

## 示例流程

```
ONNX 模型 → [构建示例] → TensorRT 引擎 → [推理示例] → 结果
```

---

## 构建示例

### `llm_build` — 源码：`examples/llm/llm_build.cpp`

为 LLM 构建 TensorRT 引擎（标准、EAGLE、VLM、LoRA）。

```bash
# 标准 LLM
./build/examples/llm/llm_build \
  --onnxDir onnx_models/qwen3-4b \
  --engineDir engines/qwen3-4b \
  --maxBatchSize 1

# 多模态（VLM）
./build/examples/llm/llm_build \
  --onnxDir onnx_models/qwen2.5-vl-3b \
  --engineDir engines/qwen2.5-vl-3b \
  --maxBatchSize 1 \
  --maxInputLen=1024 \
  --maxKVCacheCapacity=4096 \
  --vlm \
  --minImageTokens 128 \
  --maxImageTokens 512

# EAGLE（推测解码）
./build/examples/llm/llm_build \
  --onnxDir onnx_models/model_eagle_base \
  --engineDir engines/model_eagle \
  --eagleBase

./build/examples/llm/llm_build \
  --onnxDir onnx_models/model_eagle_draft \
  --engineDir engines/model_eagle \
  --eagleDraft
```

### `visual_build` — 源码：`examples/multimodal/visual_build.cpp`

为视觉编码器构建 TensorRT 引擎（Qwen-VL、InternVL、Phi-4-Multimodal）。

```bash
./build/examples/multimodal/visual_build \
  --onnxDir onnx_models/qwen2.5-vl-3b/visual_enc_onnx \
  --engineDir visual_engines/qwen2.5-vl-3b \
  --minImageTokens 128 \
  --maxImageTokens 512 \
  --maxImageTokensPerImage=512
```

---

## 推理示例

### `llm_inference` — 源码：`examples/llm/llm_inference.cpp`

从 JSON 文件批量推理。支持标准、EAGLE、多模态与 LoRA 模式。

```bash
# 标准 LLM
./build/examples/llm/llm_inference \
  --engineDir engines/qwen3-4b \
  --inputFile input.json \
  --outputFile output.json

# EAGLE（推测解码）
./build/examples/llm/llm_inference \
  --engineDir engines/model_eagle \
  --inputFile input.json \
  --outputFile output.json \
  --eagle

# 多模态（VLM）
./build/examples/llm/llm_inference \
  --engineDir engines/qwen2.5-vl-3b \
  --multimodalEngineDir visual_engines/qwen2.5-vl-3b \
  --inputFile input_with_images.json \
  --outputFile output.json

# 性能分析
./build/examples/llm/llm_inference \
  --engineDir engines/qwen3-4b \
  --inputFile input.json \
  --outputFile output.json \
  --dumpProfile \
  --profileOutputFile profile.json
```

---

## 完整工作流

### 标准 LLM（端到端）

```bash
# 1. 导出 ONNX（x86 主机）
tensorrt-edgellm-quantize-llm --model_dir Qwen/Qwen3-4B-Instruct-2507 --quantization fp8 --output_dir quantized/qwen3-4b
tensorrt-edgellm-export-llm --model_dir quantized/qwen3-4b --output_dir onnx_models/qwen3-4b

# 2. 构建引擎（Thor 设备）
./build/examples/llm/llm_build --onnxDir onnx_models/qwen3-4b --engineDir engines/qwen3-4b --maxBatchSize 1

# 3. 运行推理（Thor 设备）
./build/examples/llm/llm_inference --engineDir engines/qwen3-4b --inputFile input.json --outputFile output.json
```

### 多模态 VLM（端到端）

说明：Phi-4 需要额外的 merge-lora 步骤，请按步骤操作。

```bash
# 1. 导出（x86 主机）
tensorrt-edgellm-export-llm --model_dir Qwen/Qwen2.5-VL-3B-Instruct --output_dir onnx_models/qwen2.5-vl-3b
tensorrt-edgellm-export-visual --model_dir Qwen/Qwen2.5-VL-3B-Instruct --output_dir onnx_models/qwen2.5-vl-3b/visual_enc_onnx

# 2. 构建引擎（Thor 设备）
./build/examples/llm/llm_build --onnxDir onnx_models/qwen2.5-vl-3b --engineDir engines/qwen2.5-vl-3b --vlm
./build/examples/multimodal/visual_build --onnxDir onnx_models/qwen2.5-vl-3b/visual_enc_onnx --engineDir visual_engines/qwen2.5-vl-3b

# 3. 运行推理（Thor 设备）
./build/examples/llm/llm_inference --engineDir engines/qwen2.5-vl-3b --multimodalEngineDir visual_engines/qwen2.5-vl-3b --inputFile input_with_images.json --outputFile output.json
```

### Phi-4 与带 LoRA 的多模态 VLM（端到端）

**注意：LoRA 模型与量化流水线不兼容，需先将 LoRA 适配器合并进主模型。**

```bash
# 0. 克隆 Phi-4-multimodal-instruct 到本地
git clone https://huggingface.co/microsoft/Phi-4-multimodal-instruct
cd Phi-4-multimodal-instruct
git lfs pull

# 1. 合并 LoRA（x86 主机）
tensorrt-edgellm-merge-lora --model_dir Phi-4-multimodal-instruct \
                            --lora_dir Phi-4-multimodal-instruct/vision-lora \
                            --output_dir Phi-4-multimodal-instruct-merged-vision

# 2. 量化（x86 主机）
tensorrt-edgellm-quantize-llm --model_dir Phi-4-multimodal-instruct-merged-vision \
                               --output_dir Phi-4-multimodal-instruct-merged-vision-nvfp4 \
                               --quantization=nvfp4

# 3. 导出（x86 主机）
tensorrt-edgellm-export-llm --model_dir Phi-4-multimodal-instruct-merged-vision-nvfp4 --output_dir onnx_models/phi4-mm
# 视觉部分使用原始权重导出
tensorrt-edgellm-export-visual --model_dir Phi-4-multimodal-instruct --output_dir onnx_models/phi4-mm/visual_enc_onnx

# 4. 构建引擎（Thor 设备）
./build/examples/llm/llm_build --onnxDir onnx_models/phi4-mm --engineDir engines/phi4-mm --vlm
./build/examples/multimodal/visual_build --onnxDir onnx_models/phi4-mm/visual_enc_onnx --engineDir visual_engines/phi4-mm

# 5. 运行推理（Thor 设备）
./build/examples/llm/llm_inference --engineDir engines/phi4-mm --multimodalEngineDir visual_engines/phi4-mm --inputFile input_with_images.json --outputFile output.json
```

### 多模态 VLM + EAGLE 推测解码（端到端）

```bash
# 1. 导出（x86 主机）
tensorrt-edgellm-export-llm --model_dir Qwen/Qwen2.5-VL-7B-Instruct --output_dir onnx_models/qwen2.5-vl-7b_eagle_base --is_eagle_base
tensorrt-edgellm-export-draft --base_model_dir Qwen/Qwen2.5-VL-7B-Instruct --draft_model_dir path/to/draft --output_dir onnx_models/qwen2.5-vl-7b_eagle_draft --use_prompt_tuning
tensorrt-edgellm-export-visual --model_dir Qwen/Qwen2.5-VL-7B-Instruct --output_dir onnx_models/qwen2.5-vl-7b/visual_enc_onnx

# 2. 构建引擎（Thor 设备）
./build/examples/llm/llm_build --onnxDir onnx_models/qwen2.5-vl-7b_eagle_base --engineDir engines/qwen2.5-vl-7b_eagle --vlm --eagleBase
./build/examples/llm/llm_build --onnxDir onnx_models/qwen2.5-vl-7b_eagle_draft --engineDir engines/qwen2.5-vl-7b_eagle --vlm --eagleDraft
./build/examples/multimodal/visual_build --onnxDir onnx_models/qwen2.5-vl-7b/visual_enc_onnx --engineDir visual_engines/qwen2.5-vl-7b

# 3. 运行推理（Thor 设备）
./build/examples/llm/llm_inference --engineDir engines/qwen2.5-vl-7b_eagle --multimodalEngineDir visual_engines/qwen2.5-vl-7b --inputFile input_with_images.json --outputFile output.json --eagle
```

---

## 输入文件格式

### VLM 输入格式（`input_with_images.json`）

多模态（VLM）模型使用包含图像内容的 JSON：

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "max_generate_length": 128,
    "requests": [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "examples/multimodal/pics/woman_and_dog.jpeg"
                        },
                        {
                            "type": "text",
                            "text": "Please describe the image."
                        }
                    ]
                }
            ]
        }
    ]
}
```

### LLM 输入格式（`input.json`）

标准纯文本 LLM 见 `examples/llm/INPUT_FORMAT.md`。

---

## 常用参数

### 构建参数（`llm_build`、`visual_build`）

| 参数 | 说明 | 默认 |
|-----------|-------------|---------|
| `--onnxDir` | 输入 ONNX 目录 | 必填 |
| `--engineDir` | 输出引擎目录 | 必填 |
| `--maxBatchSize` | 最大批大小 | 4 |
| `--maxInputLen` | 最大输入长度 | 128 |
| `--maxKVCacheCapacity` | 最大 KV 缓存容量 | 4096 |
| `--vlm` | VLM 模式 | false |
| `--eagleBase/Draft` | EAGLE 模式 | false |
| `--maxLoraRank` | LoRA 秩（0 表示关闭） | 0 |

### 推理参数（`llm_inference`）

| 参数 | 说明 |
|-----------|-------------|
| `--engineDir` | 引擎目录（必填） |
| `--multimodalEngineDir` | 视觉/草稿引擎（VLM/EAGLE） |
| `--inputFile` | 输入 JSON 路径（必填） |
| `--outputFile` | 输出 JSON 路径（必填） |
| `--eagle` | 启用 EAGLE 模式 |
| `--dumpProfile` | 启用性能分析 |
| `--profileOutputFile` | 性能分析输出路径 |

**说明：** 温度、top_p、top_k 等采样参数写在输入 JSON 中。详见 `examples/llm/INPUT_FORMAT.md`。

---

## 下一步

浏览示例后，你可以：

1. **按需定制**：阅读 [定制指南](07_Customization_Guide.md) 扩展框架
2. **构建自有应用**：以示例为模板开发应用
3. **优化性能**：尝试不同量化、批大小与 CUDA Graph

---

## 更多资料

- [概览](01.1_Overview.md)
- [快速入门](01.2_Quick_Start_Guide.md)
- [支持的模型](02_Supported_Models.md)
- [定制指南](07_Customization_Guide.md)
