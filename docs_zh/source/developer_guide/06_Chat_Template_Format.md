# 对话模板格式指南

本文说明如何为 TensorRT Edge-LLM 模型创建与自定义对话模板。

## 概览

对话模板定义对话消息在送入语言模型前的格式。各模型通常有各自的模板，包含特定符号与结构。模板确保系统提示、用户消息与助手回复按模型训练时的格式组织。

### 实现理念

对话模板实现与 HuggingFace 的 `apply_chat_template` API 对齐：

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # 支持思考模式的模型
)
```

为保持 TensorRT Edge-LLM 轻量、避免依赖 Jinja，采用基于 JSON 的简单格式。实践中 Jinja 的动态性主要在「变量改变模板行为」或「单段逻辑处理多种消息类型」时有用。将消息限制为约定顺序与格式及支持的场景后，可把模板压缩为前缀/后缀对与多模态类型格式列表，效果等价且更简单。

### 核心概念

- **角色**：消息类型（system、user、assistant）
- **前缀/后缀**：包裹每条消息的专用符号
- **内容类型**：多模态内容（文本、图像、视频）的格式
- **生成提示（Generation Prompt）**：提示模型开始回复的词元序列（可含模型专用标记，用于在支持思考能力的模型上关闭思考）
- **生成提示（思考）**：启用思考模式的替代生成提示（可选，仅支持思考的模型提供）
- **默认系统提示**：未提供系统消息时的回退指令

---

## 对话模板文件格式

模板定义在 JSON 文件中。随模型导出时，该文件重命名为 `processed_chat_template.json` 并放入模型引擎目录。

### 文件结构

```json
{
  "model_path": "string (optional)",
  "roles": {
    "system": {
      "prefix": "string",
      "suffix": "string"
    },
    "user": {
      "prefix": "string",
      "suffix": "string"
    },
    "assistant": {
      "prefix": "string",
      "suffix": "string"
    }
  },
  "content_types": {
    "image": {
      "format": "string"
    },
    "video": {
      "format": "string"
    }
  },
  "generation_prompt": "string",
  "generation_prompt_thinking": "string (optional)",
  "default_system_prompt": "string"
}
```

### 字段说明

#### 必填字段

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `roles` | object | 角色名到前缀/后缀格式的映射 |
| `roles.system` | object | 系统消息格式 |
| `roles.user` | object | 用户消息格式 |
| `roles.assistant` | object | 助手消息格式 |
| `roles.<role>.prefix` | string | 内容前的词元（可为空串） |
| `roles.<role>.suffix` | string | 内容后的词元（可为空串） |

#### 可选字段

| 字段 | 类型 | 说明 | 默认 |
|-------|------|-------------|---------|
| `model_path` | string | 模型路径或标识 | "" |
| `content_types` | object | 多模态（图像、视频）格式 | {} |
| `content_types.image.format` | string | 替换图像内容的词元序列 | "" |
| `content_types.video.format` | string | 替换视频内容的词元序列 | "" |
| `generation_prompt` | string | 触发模型生成的词元序列（可含关闭思考的模型专用标记） | "" |
| `generation_prompt_thinking` | string | 触发思考模式生成的词元序列（可选，仅支持思考的模型） | "" |
| `default_system_prompt` | string | 未提供系统消息时的默认指令 | "" |

---

## 思考模式支持

部分模型（如 Qwen3）支持「思考模式」，可在最终答案前可选地生成推理过程，由输入 JSON 的 `enable_thinking` 控制。

**重要：** 思考模式仅对训练具备该能力的模型有效。导出时系统会检查分词器对话模板是否支持思考；若支持，会提取并保存两种生成提示。

### 工作原理

支持思考时，模板可定义两种生成提示：

- **`generation_prompt`**：关闭思考模式的生成提示
  - Qwen3 示例：`<|im_start|>assistant\n<redacted_thinking>\n\n</redacted_thinking>\n\n`
  - 具体机制因模型而异。Qwen3 中空 `<redacted_thinking>` 表示思考已完成，模型应直接作答

- **`generation_prompt_thinking`**：启用思考模式的生成提示
  - Qwen3 示例：`<|im_start|>assistant\n`
  - 无预定义思考标记时，模型可自由生成推理过程

导出时若分词器支持思考，两种生成提示会自动写入模板。若模型不支持思考，仅存在 `generation_prompt`，`enable_thinking` 无效。

**说明：** `enable_thinking` 命名与 HuggingFace 对支持思考模型的实现一致。默认为 `False` 时，可能使用带空 `<redacted_thinking>` 的提示以关闭思考；为 `True` 时使用无思考标记的提示，允许生成推理。

### 输出格式示例

Qwen3 在关闭思考（默认 `enable_thinking=False`）时，格式化提示含空思考标签：

```
<|im_start|>user
Give me a short introduction to large language model.<|redacted_im_end|>
<|im_start|>assistant
<redacted_thinking>

</redacted_thinking>

```

Qwen3 中空 `<redacted_thinking>` 表示不再生成额外推理，直接给出答案。其他模型可能用不同机制控制思考行为。

### 用法

在输入 JSON 中设置 `enable_thinking` 以选择生成提示：

```json
{
  "enable_thinking": true,
  "requests": [
    {
      "messages": [
        {"role": "user", "content": "Solve this complex problem..."}
      ]
    }
  ]
}
```

**行为：**

- **未指定** `enable_thinking`（默认 `false`）：使用 `generation_prompt`（若模型支持思考则关闭思考）
- `enable_thinking` 为 `false`：使用 `generation_prompt`（同上）
- `enable_thinking` 为 `true`：使用 `generation_prompt_thinking`（若模型支持思考则启用思考）
- **模型不支持思考**：`enable_thinking` 无效，始终使用 `generation_prompt`

**说明：** 启用/关闭思考的具体机制因模型而异，在导出时从分词器提取模板时确定。

`enable_thinking` 更多说明见 [INPUT_FORMAT.md](../../../examples/llm/INPUT_FORMAT.md)。

---

## 导出时使用对话模板

默认导出流程会：

1. 尝试从模型分词器检测并提取对话模板。
2. 若已知模型模板不兼容或缺失，自动回退到预置模板（见 [预置模板](#预置模板)）。

### 自定义模板（可选）

可用 `--chat_template` 覆盖默认行为：

```bash
tensorrt-edgellm-export-llm \
    --model_dir /path/to/model \
    --output_dir /path/to/output \
    --chat_template /path/to/my_custom_template.json
```

在以下情况提供自定义模板：

1. **自动提取失败**：无法可靠检测或转换模型对话模板。
2. **模型不在预置列表中**，但分词器模板仍有问题。
3. **需要与默认不同的格式**（例如不同系统提示、角色符号或多模态占位符）。

---

## 预置模板

部分模型的分词器对话模板存在问题（缺失或不兼容格式）。TensorRT Edge-LLM 为这些模型提供预置模板并自动使用。

**位置：** `tensorrt_edgellm/chat_templates/templates/`

| 模型 | 模板文件 | 原因 |
|-------|--------------|--------|
| Phi-4-Multimodal | `phi4mm.json` | 分词器缺少正确的多模态对话模板定义 |

**自动回退：** 导出在「不兼容列表」中的模型（如 Phi-4-Multimodal）时，系统自动使用预置模板。除非你要覆盖，否则无需手动指定。

---

## 示例

### 纯文本模型

```json
{
  "model_path": "/path/to/Qwen2-7B",
  "roles": {
    "system": {
      "prefix": "<|im_start|>system\n",
      "suffix": "<|redacted_im_end|>\n"
    },
    "user": {
      "prefix": "<|im_start|>user\n",
      "suffix": "<|redacted_im_end|>\n"
    },
    "assistant": {
      "prefix": "<|im_start|>assistant\n",
      "suffix": "<|redacted_im_end|>\n"
    }
  },
  "content_types": {},
  "generation_prompt": "<|im_start|>assistant\n",
  "default_system_prompt": "You are a helpful assistant"
}
```

### 含内容类型的多模态模型

对视觉语言模型，`format` 字段指定在格式化提示中替换图像或视频的词元序列：

```json
{
  "model_path": "/path/to/Qwen2-VL-7B",
  "roles": {
    "system": {
      "prefix": "<|im_start|>system\n",
      "suffix": "<|redacted_im_end|>\n"
    },
    "user": {
      "prefix": "<|im_start|>user\n",
      "suffix": "<|redacted_im_end|>\n"
    },
    "assistant": {
      "prefix": "<|im_start|>assistant\n",
      "suffix": "<|redacted_im_end|>\n"
    }
  },
  "content_types": {
    "image": {
      "format": "<|vision_start|><|image_pad|><|vision_end|>"
    },
    "video": {
      "format": "<|vision_start|><|video_pad|><|vision_end|>"
    }
  },
  "generation_prompt": "<|im_start|>assistant\n",
  "default_system_prompt": "You are a helpful assistant."
}
```

### 支持思考模式的模型

支持思考（如 Qwen3）时，模板包含两种生成提示：

```json
{
  "model_path": "/path/to/Qwen3-0.6B",
  "roles": {
    "system": {
      "prefix": "<|im_start|>system\n",
      "suffix": "<|redacted_im_end|>\n"
    },
    "user": {
      "prefix": "<|im_start|>user\n",
      "suffix": "<|redacted_im_end|>\n"
    },
    "assistant": {
      "prefix": "<|im_start|>assistant\n",
      "suffix": "<|redacted_im_end|>\n"
    }
  },
  "content_types": {},
  "generation_prompt": "<|im_start|>assistant\n<redacted_thinking>\n\n</redacted_thinking>\n\n",
  "generation_prompt_thinking": "<|im_start|>assistant\n",
  "default_system_prompt": "You are a helpful assistant"
}
```

`generation_prompt` 可含模型专用关闭思考标记（如 Qwen3 的空 `<redacted_thinking>`）。`generation_prompt_thinking` 通过省略此类标记启用思考（输入 JSON 中 `enable_thinking: true` 时使用）。具体格式在导出时由分词器配置自动确定。

### 多轮对话示例

**模板：**

```json
{
  "roles": {
    "system": {
      "prefix": "<|im_start|>system\n",
      "suffix": "<|redacted_im_end|>\n"
    },
    "user": {
      "prefix": "<|im_start|>user\n",
      "suffix": "<|redacted_im_end|>\n"
    },
    "assistant": {
      "prefix": "<|im_start|>assistant\n",
      "suffix": "<|redacted_im_end|>\n"
    }
  },
  "generation_prompt": "<|im_start|>assistant\n",
  "default_system_prompt": "You are a helpful assistant"
}
```

**输入：**

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a math tutor."
    },
    {
      "role": "user",
      "content": "What is 2+2?"
    },
    {
      "role": "assistant",
      "content": "2+2 equals 4."
    },
    {
      "role": "user",
      "content": "What about 3+3?"
    }
  ]
}
```

**格式化输出：**

```
<|im_start|>system
You are a math tutor.<|redacted_im_end|>
<|im_start|>user
What is 2+2?<|redacted_im_end|>
<|im_start|>assistant
2+2 equals 4.<|redacted_im_end|>
<|im_start|>user
What about 3+3?<|redacted_im_end|>
<|im_start|>assistant

```

---

## 系统提示优先级

系统提示按以下优先级确定：

1. **请求中的显式 system 消息**（最高）
2. **对话模板中的 `default_system_prompt`**（最低）

若两者皆无，则不添加系统提示。

**场景 1：显式 system 消息（最高优先级）**

```json
{
  "requests": [
    {
      "messages": [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 2+2?"}
      ]
    }
  ]
}
```

结果：使用 "You are a math tutor."，不受模板默认值影响。

**场景 2：回退到模板默认**

```json
{
  "requests": [
    {
      "messages": [
        {"role": "user", "content": "Hello!"}
      ]
    }
  ]
}
```

结果：无显式 system 时，使用 `processed_chat_template.json` 中的 `default_system_prompt`（若已定义）。

输入 JSON 格式更多说明见 [INPUT_FORMAT.md](../../../examples/llm/INPUT_FORMAT.md)。
