# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.

# This file has been enhanced with deep Chinese annotations by Cursor AI.

# -----------------------------------------------------------------------------
# 本文件定义 LLM 相关两个示例可执行目标，供聊天、基准测试与 accuracy 评测等场景复用。
# llm_build：编译期从 ONNX 构建 TensorRT engine 文件并序列化到磁盘。
# llm_inference：运行期加载 engine 文件、Tokenizer 与可选多模态资源，执行端到端生成。
# add_cross_build_link_options 由父级 CMake 提供，用于嵌入式/交叉链接时统一追加链接选项（具体 flag 见主工程）。
# -----------------------------------------------------------------------------

# llm_build：单源文件可执行程序，链接 ONNX Parser 与 Edge LLM 构建器，将导出目录编译为 engine 文件目录。
add_executable(llm_build llm_build.cpp)
# PRIVATE 依赖：NV_ONNX_PARSER_LIB 解析图结构；commonLibraryExt 为工程公共扩展；edgellmBuilder 封装 TensorRT 构建与序列化逻辑。
target_link_libraries(llm_build PRIVATE ${NV_ONNX_PARSER_LIB} commonLibraryExt
                                        edgellmBuilder)
# 需要 ONNX Parser 头文件及工程 COMMON_INCLUDE_DIRS（含运行时/构建器公共头）。
target_include_directories(llm_build PRIVATE ${ONNX_PARSER_INCLUDE_DIR}
                                             ${COMMON_INCLUDE_DIRS})
add_cross_build_link_options(llm_build)

# llm_inference：加载已构建的 engine 文件，处理输入 JSON、对话模板与可选图像；依赖运行时核心与分词器。
add_executable(llm_inference llm_inference.cpp)
# edgellmCore：反序列化 engine、ExecutionContext、KV cache / 激活缓存等运行时路径。
# edgellmTokenizer：与模板一致的编解码。
# exampleUtils：示例用显存监控与 profile 格式化（实现位于 examples/utils）。
# commonLibraryExt：日志、张量工具等公共能力。
target_link_libraries(llm_inference PRIVATE edgellmCore edgellmTokenizer
                                            exampleUtils commonLibraryExt)
# 显式加入 utils 目录，使 memoryMonitor.h / profileFormatter.h 等无需安装规则即可被 #include。
target_include_directories(
  llm_inference PRIVATE ${COMMON_INCLUDE_DIRS}
                        ${CMAKE_SOURCE_DIR}/examples/utils)
add_cross_build_link_options(llm_inference)
