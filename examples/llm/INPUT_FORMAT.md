# 输入 JSON 文件格式

本文说明与 LLM 推理工具配合使用的输入 JSON 文件所须遵循的格式。

## 概览

输入 JSON 文件包含配置参数以及待 LLM 处理的请求列表。每条请求是一段由多条带角色与内容的 message 组成的对话。工具支持纯文本与多模态（文本 + 图像）输入，并支持多轮对话。

## 文件结构

JSON 文件顶层须包含如下结构：

```json
{
    "batch_size": <integer>,
    "temperature": <float>,
    "top_p": <float>,
    "top_k": <integer>,
    "max_generate_length": <integer>,
    "apply_chat_template": <boolean>,  // optional (default: true)
    "enable_thinking": <boolean>,  // optional (default: false, Qwen3-specific)
    "available_lora_weights": {  // optional. Only needed for LoRA engines.
        "<lora_name>": "<path_to_safetensors_file>",
        ...
    },
    "requests": [
        {
            "messages": [
                {
                    "role": "<string>",  // "system", "user", or "assistant"
                    "content": "<string>"  // Simple string for text-only messages
                    // OR
                    "content": [  // Array format for multimodal content
                        {
                            "type": "<string>",  // "text", "image", or "video"
                            "text": "<string>",  // for type="text"
                            "image": "<path>",   // for type="image"
                            "video": "<path>"    // for type="video"
                        }
                    ]
                }
            ],
            "lora_name": "<string>",  // optional. Name of LoRA weights from available_lora_weights.
            "save_system_prompt_kv_cache": <boolean>  // optional (default: false). Save system prompt KV cache for reuse.
        }
    ]
}
```


## LoRA（Low-Rank Adaptation）支持

LoRA 通过适配器权重支持微调模型的推理。

### 定义可用的 LoRA 权重

首先在全局 `available_lora_weights` 映射中声明所有可用的 LoRA 适配器：

```json
{
    "available_lora_weights": {
        "french_adapter": "/path/to/french_adapter.safetensors",
        "spanish_adapter": "/path/to/spanish_adapter.safetensors",
        "jailbreak_detector": "/path/to/jailbreak_detector.safetensors"
    }
}
```

### 按对话选择 LoRA

随后在每条请求中用名称引用上述适配器：

```json
{
    "available_lora_weights": {
        "french_adapter": "/path/to/french_adapter.safetensors",
        "spanish_adapter": "/path/to/spanish_adapter.safetensors"
    },
    "requests": [
        {
            "messages": [...],
            "lora_name": "french_adapter"
        },
        {
            "messages": [...],
            "lora_name": "spanish_adapter"
        }
    ]
}
```

### 要求：

- 须使用支持 LoRA 的 TensorRT 构建流程生成 **engine 文件**（即带 LoRA 能力的 plan）
- LoRA 权重须为 `.safetensors` 格式
- 在通过 `lora_name` 引用前，必须先在 `available_lora_weights` 中定义对应适配器
- **重要：** 同一 **batch** 内的所有请求必须使用相同的 LoRA 权重；不同 LoRA 仅能通过不同 batch 切换。

## 全局参数

### 必填参数

- **`requests`**（对象数组）：对话请求列表。每个请求为包含 `messages` 数组及可选逐对话配置的对象。

### 可选参数

- **`batch_size`**（整数，默认 1）：单个 batch 内合并处理的请求条数
- **`temperature`**（浮点，默认 1.0）：控制生成的随机性（0.0 更确定，越大越随机）
- **`top_p`**（浮点，默认 0.8）：Nucleus 采样参数（0.0–1.0）
- **`top_k`**（整数，默认 50）：Top-k 采样参数
- **`max_generate_length`**（整数，默认 256）：最多生成的 token 数
- **`apply_chat_template`**（布尔，默认 true）：是否套用对话模板并插入特殊 token。为 `false` 时，消息将不做角色前后缀与特殊 token 拼接，适用于不需要对话模板的模型
- **`enable_thinking`**（布尔，默认 false）：是否启用思考模式（面向支持该能力的模型，如 Qwen3 系列）。为 `false` 时使用标准生成提示；为 `true` 时在可用时使用带思考的生成提示。对不支持思考模式的模型该字段无效
- **`available_lora_weights`**（对象，默认 {}）：LoRA 名称到权重文件路径的映射；仅在使用支持 LoRA 的 **engine 文件**时需要

## 系统提示（System Prompt）行为

- 若请求中提供了 system 消息，则使用该内容
- 若未提供 system 消息，则使用对话模板中配置的模型默认 system 提示（若有）

## 请求结构

`requests` 数组中每个请求对象可包含以下字段：

### 必填字段

- **`messages`**（数组）：构成一轮或多轮对话的消息列表，用于携带历史上下文。

### 可选字段

- **`lora_name`**（字符串）：本对话使用的 LoRA 名称，对应全局 `available_lora_weights` 中的键。不同对话可选用不同微调适配器。注意：同一 batch 内须使用相同 LoRA 权重。

- **`save_system_prompt_kv_cache`**（布尔，默认 false）：是否保存 system 提示对应的 **KV cache** 以供后续复用。在多条请求重复使用较长 system 提示时可减少重复计算。注意：若 batch 中任一条将该字段设为 `true`，则该 batch 内会对 system 提示做 **KV cache** 缓存行为（与示例程序实现一致）。

### 消息结构

每条 message 对象包含：

#### 必填字段

- **`role`**（字符串）：发送者角色，须为以下之一：
  - `"system"`：系统指令或上下文（可省略，省略时可能使用模板默认）
  - `"user"`：用户输入或问题
  - `"assistant"`：助手历史回复（用于多轮）

- **`content`**（字符串或数组）：消息内容，可为：
  - **字符串形式**（纯文本，较简单）：直接文本
  - **数组形式**（多模态）：由文本、图像、视频等条目组成的列表

#### 内容格式

**字符串形式（纯文本消息）：**

纯文本可直接使用字符串：

```json
"content": "Your text message here"
```

**数组形式（多模态消息）：**

含图像、视频或混合内容时，使用内容项数组。

每项含 `type` 及类型相关字段：

**文本项：**
- **`type`**：`"text"`
- **`text`**（字符串）：文本内容

**图像项：**
- **`type`**：`"image"`
- **`image`**（字符串）：图像文件路径

**视频项：**
- **`type`**：`"video"`
- **`video`**（字符串）：视频文件路径

## 示例

### 纯文本输入（单条请求）

使用字符串形式承载纯文本：

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "max_generate_length": 256,
    "requests": [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Introduce NVIDIA and introduce the CEO of this company."
                }
            ]
        }
    ]
}
```

**说明：** 纯文本也可改用数组形式，例如：

```json
"content": [{"type": "text", "text": "Your message here"}]
```

### 多轮对话

字符串形式便于书写多轮文本对话：

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 0.8,
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
                    "content": "What is the capital of France?"
                },
                {
                    "role": "assistant",
                    "content": "The capital of France is Paris."
                },
                {
                    "role": "user",
                    "content": "What is the population of that city?"
                }
            ]
        }
    ]
}
```

### 多模态输入（文本 + 图像）

多模态请使用数组形式：

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "max_generate_length": 256,
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
                        {"type": "image", "image": "image1.jpeg"},
                        {"type": "image", "image": "image2.jpeg"},
                        {"type": "text", "text": "Compare these two images and identify the similarities."}
                    ]
                }
            ]
        }
    ]
}
```

### 批处理（多条请求）

同一 batch 内可混合字符串形式（纯文本）与数组形式（多模态）：

```json
{
    "batch_size": 2,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "max_generate_length": 256,
    "requests": [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What is machine learning?"
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "diagram.png"},
                        {"type": "text", "text": "Explain this diagram."}
                    ]
                }
            ]
        }
    ]
}
```

### LoRA 输入（按对话指定）

通过名称在不同对话中选用不同 LoRA 适配器：

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "max_generate_length": 256,
    "available_lora_weights": {
        "french_adapter": "/path/to/french_adapter.safetensors",
        "spanish_adapter": "/path/to/spanish_adapter.safetensors"
    },
    "requests": [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Translate this text to French."
                }
            ],
            "lora_name": "french_adapter"
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Translate this text to Spanish."
                }
            ],
            "lora_name": "spanish_adapter"
        }
    ]
}
```

**说明：** 上例在 `batch_size` 为 1 时，每条请求自然落在独立 batch，因而可使用不同 LoRA。若手动将 `batch_size` 设为 2 或更大、把使用不同 LoRA 的请求并入同一 batch，**程序将报错**：`Different LoRA weights within the same batch are not supported`。

### 原始格式输入（不使用对话模板）

若需不做对话模板拼接、不使用模板特殊 token，将全局参数 `apply_chat_template` 设为 `false`：

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "max_generate_length": 256,
    "apply_chat_template": false,
    "requests": [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France?"
                }
            ]
        }
    ]
}
```

效果为原始拼接，不会插入 `<|im_start|>`、`<|im_end|>` 等模板特殊符号（具体符号以所用模板为准）。适用于：

- 训练时未使用对话模板的模型
- 自定义 prompt 工程
- 需要直接控制输入形态的场景

注意：`apply_chat_template` 对同一 batch 内所有请求一致生效。

### System 提示 **KV cache** 优化

在反复使用较长 system 提示时，可将某条请求的 `save_system_prompt_kv_cache` 设为 `true`，以缓存 system 部分的 **KV cache** 供复用：

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "max_generate_length": 256,
    "requests": [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant with extensive knowledge..."
                },
                {
                    "role": "user",
                    "content": "What is the capital of France?"
                }
            ],
            "save_system_prompt_kv_cache": true
        }
    ]
}
```

典型适用场景：

- 多条请求复用同一段较长 system 提示
- 初始化阶段预先缓存系统指令
- system 上下文在多次调用中保持不变

## 处理行为说明

1. **对话模板应用**：当 `apply_chat_template` 为 `true`（默认）时，会按 `processed_chat_template.json` 加载的模板为消息添加角色前后缀与特殊 token。
2. **原始格式模式**：`apply_chat_template` 为 `false` 时，消息不经角色专用 token 或模板化格式直接拼接，适用于无模板模型、自定义 prompt、或简单的图文顺序拼接。
3. **系统提示**：若提供 system 消息则使用之；否则使用模板中的默认 system（若有）。
4. **批处理**：按 `batch_size` 将 `requests` 划分为多个 batch 依次处理。
5. **多轮支持**：单条请求内可含多条 message，以携带完整对话上下文。
6. **内容类型**：文本直接进入格式化后的 prompt；图像/视频占位按模板规则插入。
7. **图像加载**：处理过程中按路径从磁盘加载图像（路径规则见文末「注意事项」）。
8. **LoRA 权重**：初始化时根据 `available_lora_weights` 注册权重；运行期按 batch 的 `lora_name` 切换（须满足同 batch 同 LoRA 的约束）。
9. **System 提示 KV cache**：当 `save_system_prompt_kv_cache` 为 `true` 时，缓存 system 提示对应的 **KV cache** 以加速后续复用。
10. **错误处理**：在以下情况工具/程序会报错或抛异常：
   - JSON 无法解析
   - 消息缺少必填字段 `role` 或 `content`
   - 请求对象缺少必填字段 `messages`
   - `requests` 不是对象数组
   - 出现未知 content `type`
   - 同一 batch 内指定了不同 LoRA
   - `lora_name` 在 `available_lora_weights` 中未定义

## 注意事项

- 图像与视频路径可为相对当前工作目录的相对路径，或绝对路径
- 对话模板会自动添加约定好的特殊 token 与版式
- 对话中间的 assistant 消息用于携带历史回复，实现多轮上下文
- 整体结构与 OpenAI Chat Completions API 高度相似，便于互操作

## 设计原则摘要

输入格式旨在：

- **对齐 OpenAI Chat Completions 结构**，便于生态互操作
- **支持多轮对话**与完整历史上下文
- **支持按对话选择 LoRA**，适配多种微调 **engine 文件**
- **统一承载多模态**（文本、图像、视频）
- **区分全局参数与逐请求对话内容**，职责清晰

---

**英文版说明：** 与本文对应的英文原文见同目录 [INPUT_FORMAT_en.md](INPUT_FORMAT_en.md)。
