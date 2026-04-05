# TensorRT 插件指南

本文说明如何在 TensorRT Edge-LLM 中使用 TensorRT 插件，并指导进一步定制。

## 概览

TensorRT 插件是通过用户自定义层实现、扩展 TensorRT 核心库功能的自定义算子。在 TensorRT Edge-LLM 中，插件为关键大型语言模型（LLM）推理算子提供专用实现，这些算子需要超出标准 TensorRT 交付物所能提供的优化。

### 插件架构与能力

TensorRT 插件是实现 `IPluginV2DynamicExt` 接口的用户定义层，具备：

- **能力扩展**：以新的运行时与内核级优化扩展现有 TensorRT 版本。
- **模块化封装**：将复杂计算封装为可复用、可配置组件。

### 当前插件

1. **AttentionPlugin**：实现标准 MHA（多头注意力）与 GQA（分组查询注意力）。
2. **Int4GroupwiseGemmPlugin**：实现 INT4 仅权重的分组 GEMM 与 GEMV。

## AttentionPlugin

**功能说明**：

- 处理旋转位置编码、KVCache 输入输出，以及 MHA/GQA 注意力计算。
- 实现 FP16 精度，覆盖 TensorRT Edge-LLM 支持的全部 SM。
- 支持 prefill（普通与分块）阶段的因果注意力。
- 支持常规解码注意力，以及 EAGLE 推测解码使用的树状解码注意力。
- 支持批内等容量的线性 KVCache。
- Prefill 执行时在批内填充到最大输入序列长度。

**配置参数**：

- `num_q_heads`：Query 头数（整数）
- `num_kv_heads`：KV 头数（整数，支持 MQA/GQA）
- `head_size`：每头维度（整数）
- `enable_tree_attention`：是否启用树注意力（布尔），用于推测解码

**输入张量**：

- `PackedQKV`：Q/K/V 投影打包张量，布局 `[B, S, H, D]`
- `KVCache`：布局 `[B, 2, H, S, D]`，`S` 为每序列 KVCache 容量
- `ContextLengths`：批内各序列长度
- `RopeCosSin`：预计算的 RoPE 余弦/正弦缓存
- `KVCacheStartIndex`：分块 prefill 时 KVCache 起始索引
- `AttentionMask`：（可选）按位描述推测草稿树的注意力模式
- `AttentionPosIds`：（可选）草稿树词元在序列中的位置

**输出张量**：

- `AttentionOutput`：注意力计算结果
- `KVCache`：输出 KVCache（与输入 KVCache 同地址）

**应用领域**：

- 采用标准 MHA/GQA 的 Transformer 自回归语言模型。

### 内核来源

注意力内核编译为 CUDA 二进制。在 `kernelSrcs/` 中提供生成二进制的方法。

**内核库**：

- `fmha_v2`：NVIDIA 开发的上下文阶段高性能注意力内核。扩展说明见原代码仓库。
- `xqa`：NVIDIA 开发的解码阶段高性能内核，实现常规解码与树注意力解码。

### 集成流程

AttentionPlugin 在 TensorRT Edge-LLM 推理流水线中经以下阶段集成：

1. **导出阶段**：ONNX 导出时，通过 Python 导出流水线将注意力算子标注为基于插件的实现。
2. **引擎构建**：TensorRT 引擎构建器通过已注册的插件创建器识别插件算子，并入优化计算图。
3. **运行时执行**：推理时，AttentionPlugin 作为 TensorRT 引擎执行图中的节点运行，显存由 TensorRT 运行时管理。

## Int4GroupwiseGemmPlugin

**功能说明**：

- 实现语义为 A([M, K]) × B([K, N]) 的 GEMM，A 为激活，B 为权重。
- 支持 INT4 仅权重的分组量化 GEMM。
- 分组大小为 128。
- GEMM 与 GEMV 内核均在 FP16 精度下累加。
- 对称量化，不支持零点。

**配置参数**：

- `N`：GEMM 输出特征维
- `K`：GEMM 内积维
- `GroupSize`：每组 INT4 权重对应的缩放因子数量（当前仅支持 128）

**输入张量**：

- `GEMMInput`：GEMM 输入激活
- `Int4Weights`：打包为 INT8 的交错 INT4 权重
- `ScalingFactors`：分组缩放因子

**输出张量**：

- `GEMMOutput`：INT4 分组 GEMM 结果。

### 内核来源

为该插件提供简化内核实现。评估表明，在目标生产平台（主要为 Orin SKU）上，输入序列长度约 2K–3K 词元时，该 INT4 GEMM 内核性能与 CUTLASS 实现相当。草稿树规模为 64–128 词元的推测解码场景下，该 GEMM 内核可能无法提供足够性能。

### 集成流程

Int4GroupwiseGemmPlugin 在流水线中经以下阶段集成：

1. **量化阶段**：模型量化时，线性层转为分组大小 128 的 INT4 仅权重量化格式。
2. **导出阶段**：ONNX 导出时，经 Python 导出流水线将量化矩阵乘标注为 Int4GroupwiseGemmPlugin 实现。
3. **引擎构建**：构建器通过注册的插件创建器识别 Int4GroupwiseGemmPlugin 并并入优化计算图。
4. **运行时执行**：推理时，Int4GroupwiseGemmPlugin 在引擎执行图中执行量化 GEMM/GEMV。

