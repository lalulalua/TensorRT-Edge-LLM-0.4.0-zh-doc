# Tier B：运行时深度注释计划（KV / TensorRT 执行 / 采样 / 同步）

本文档为 **后续可选任务** 的执行大纲：`examples/llm/llm_inference.cpp` 仅负责编排，**Prefill、Decode 循环、KV cache 布局与更新、Top-K/Top-P 采样数学、`IExecutionContext` / `enqueue` 级行为**均在 `edgellmCore` 运行时与相关模块中实现。Tier B 目标是在这些源码上补充与 [`.cursor/.cursorrules.md`](.cursor/.cursorrules.md) 一致的中文注释（**engine 文件**不译成「引擎」；CUDA / TensorRT 保留英文）。

---

## 1. 目标与边界

| 目标 | 说明 |
|------|------|
| KV cache | 线性/分层 buffer 如何与 `executePrefillStep` / `executeVanillaDecodingStep` 配合；是否存在 **circular buffer** 或 **position 取模**；与 `maxKVCacheCapacity` 的关系。 |
| Token 对齐 | `mInputIds` 等张量如何 **pad 或 reshape** 到引擎构建期固定的 max seq；与 `setUpForPrefillExecution`、tokenizer `encode` 输出的关系。 |
| 采样 | `sampleTokens()`（或等价路径）内 **temperature / top_p / top_k** 的截断与归一化顺序；是否在 GPU kernel 或 host 完成。 |
| 异步与同步 | 哪些调用仅 **异步 enqueue 到 `cudaStream_t`**；何处 **`cudaStreamSynchronize`** 或隐式同步；与 CUDA Graph capture 的先后约束。 |
| TensorRT 状态 | `IExecutionContext`（若对外暴露）绑定、**Activation Buffer** 复用、多步执行之间的 tensor 地址是否稳定。 |

---

## 2. 建议优先阅读与注释的文件（按依赖顺序）

1. [`cpp/runtime/llmInferenceRuntime.h`](d:/Dev/Learn/TensorRT-Edge-LLM-0.4.0/cpp/runtime/llmInferenceRuntime.h) — 类职责、`handleRequest` 与成员（runner、KV、tokenizer）一览。
2. [`cpp/runtime/llmInferenceRuntime.cpp`](d:/Dev/Learn/TensorRT-Edge-LLM-0.4.0/cpp/runtime/llmInferenceRuntime.cpp) — `handleRequest` 主路径：`preprocess` / `setUpForPrefillExecution`、`executePrefillStep`、`executeVanillaDecodingStep` 循环、`sampleTokens`、system prompt KV 保存与复用。
3. **LLM Engine Runner**（名称以仓库为准，例如 `*LLMEngineRunner*` / `*EngineRunner*`）— 实际调用 TensorRT 执行与 KV tensor 绑定的实现。
4. **LinearKVCache 与相关 kernel** — `saveKVCacheIntoTensor` / `instantiateKVCacheFromTensor` 等：KV 在显存中的 **layout 与 batch/head/seq 维**。
5. [`cpp/runtime/llmInferenceSpecDecodeRuntime.cpp`](d:/Dev/Learn/TensorRT-Edge-LLM-0.4.0/cpp/runtime/llmInferenceSpecDecodeRuntime.cpp) — Eagle 路径下 draft/base 与 KV 复用边界（若需与标准路径对照）。
6. **Tokenizer 封装**（`cpp/tokenizer/` 或等价路径）— `encode` 与 **特殊 token**、长度检查与运行时 `maxSupportedInputLength` 报错路径。

---

## 3. 注释规范（与 Tier A 对齐）

对每个 **公开或关键内部函数** 使用：

```cpp
/**
 * @desc: 在 LLM 推理流水线中的阶段角色（Prefill / Decode / Sampling / KV 管理）
 * @params: 参数名 — 物理含义 — 与典型 Tensor shape 的对应关系（若适用）
 * @return: 语义与失败时行为（是否已部分修改 GPU 状态）
 * @others: Stream 同步假设、与上一步执行的先后约束、是否可重入
 */
```

对 **enqueue / kernel / 指针算术** 行：在代码上方用中文写清 **「做什么 + 为何」**，避免与 `llm_inference.cpp` 中已写的「边界说明」重复时，可交叉引用 `LLMInferenceRuntime::handleRequest` 内小节标题。

---

## 4. 建议执行步骤（待排期）

1. `rg`/语义搜索定位 `sampleTokens`、`executePrefillStep`、`LinearKVCache`、`enqueue` 定义处。
2. 为 `llmInferenceRuntime::handleRequest` 画 **数据流简图**（Host JSON → formatted prompt → token ids → GPU tensor → prefill → decode 循环 → CPU 文本）。
3. 自内向外注释：先 **KV + runner 执行**，再 **采样**，再 **多模态分支**。
4. 自检：术语与 `.cursorrules` 一致；不臆造不存在的 API；不确定处用 `@others` 标明「需对照 TensorRT 版本行为」。

---

## 6. 执行情况（已落地核心路径）

已在以下文件中补充与 §3 对齐的中文 / Javadoc 风格说明（**未**全覆盖 SpecDecode / tokenizer 全文件）：

- [`cpp/runtime/llmInferenceRuntime.cpp`](../cpp/runtime/llmInferenceRuntime.cpp) — `handleRequest`、`setUpForPrefillExecution`、采样与同步、`captureDecodingCUDAGraph`、`genAndSaveSystemPromptKVCache` 等。
- [`cpp/runtime/llmInferenceRuntime.h`](../cpp/runtime/llmInferenceRuntime.h) — 类级 Tier B 职责摘要。
- [`cpp/runtime/llmEngineRunner.cpp`](../cpp/runtime/llmEngineRunner.cpp) — `bindKVCacheToEngine`、`executePrefillStep`、`executeVanillaDecodingStep` 与 `enqueueV3` 语义。
- [`cpp/runtime/linearKVCache.h`](../cpp/runtime/linearKVCache.h)、[`cpp/runtime/linearKVCache.cpp`](../cpp/runtime/linearKVCache.cpp) — 布局、非环形缓冲、`resetForNewSequences` / `commitSequenceLength`。

可选后续：[`llmInferenceSpecDecodeRuntime.cpp`](../cpp/runtime/llmInferenceSpecDecodeRuntime.cpp)、`cpp/tokenizer/` 对照注释。

---

## 5. 与 Tier A 的关系

- **Tier A**（[`examples/llm/llm_inference.cpp`](d:/Dev/Learn/TensorRT-Edge-LLM-0.4.0/examples/llm/llm_inference.cpp)）：CLI、JSON、`loadEdgellmPluginLib`、`cudaStreamCreate`、`handleRequest` **调用边界**、profile/JSON 写出。
- **Tier B**（本文件）：上述「边界之内」的 **edgellmCore** 实现细节。

完成 Tier B 后，可在 `llm_inference.cpp` 的 `handleRequest` 注释处增加一行 `@see` 式文件指针，指向本计划中列出的首个 runtime 源文件（可选，避免过多链接维护成本时可省略）。
