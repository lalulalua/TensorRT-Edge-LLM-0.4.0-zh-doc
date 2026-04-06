/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file has been enhanced with deep Chinese annotations by Cursor AI.

// 引入 TensorRT 工具与通用宏定义
#include "common/trtUtils.h"
// 引入显存监控器，用于捕获推理期间的 Device Memory 峰值
#include "memoryMonitor.h"
#include "profileFormatter.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
// 引入标准自回归 LLM 运行时实现
#include "runtime/llmInferenceRuntime.h"
// 引入 Eagle 投机解码运行时实现
#include "runtime/llmInferenceSpecDecodeRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

/**
 * @desc: 定义命令行解析长选项的枚举 ID，用于 getopt_long 返回值与 switch 分发逻辑。
 * @params: 无
 * @return: 无
 * @others: 基础值从 900 开始，严格避免与 ASCII 标准单字符选项（如 'a', 'b'）发生整数值冲突。
 */
enum LLMInferenceOptionId : int
{
    HELP = 900,
    INPUT_FILE = 901,
    ENGINE_DIR = 902,
    MULTIMODAL_ENGINE_DIR = 903,
    OUTPUT_FILE = 904,
    DEBUG = 905,
    DUMP_PROFILE = 906,
    PROFILE_OUTPUT_FILE = 907,
    WARMUP = 908,
    DUMP_OUTPUT = 909,
    EAGLE = 910,
    EAGLE_DRAFT_TOP_K = 911,
    EAGLE_DRAFT_STEP = 912,
    EAGLE_VERIFY_TREE_SIZE = 913,
    BATCH_SIZE = 914,
    MAX_GENERATE_LENGTH = 915
};

/**
 * @desc: Eagle 投机解码的核心超参配置聚合体，决定 draft model 生成 Token 树的拓扑结构。
 * @params: 无（结构体数据容器）
 * @return: 无
 * @others: 决定了底层显存中 Draft KV Cache 和 Tree 验证阶段的张量形状 (Tensor Shape)。
 */
struct EagleArgs
{
    // 是否启用 Eagle 投机解码流
    bool enabled{false};
    // Draft 阶段每步选择的候选 Token 数量，控制树的分支因子 (Branching factor)
    int32_t draftTopK{10};
    // Draft 预测的步数，决定树的深度
    int32_t draftStep{6};
    // Base 模型验证的节点总数，最大理论值: 1 + topK + (step-1) * topK^2
    int32_t verifyTreeSize{60};
};

/**
 * @desc: 承载运行时所有环境与执行策略配置的聚合体，由命令行参数直接映射。
 * @params: 无（结构体数据容器）
 * @return: 无
 * @others: 这里的 batchSize 和 maxGenerateLength 若被设置，将覆盖输入 JSON 文件中的同名全局设定。
 */
struct LLMInferenceArgs
{
    bool help{false};
    // 文本大模型对应的 TensorRT engine 文件所在目录
    std::string engineDir;
    // 多模态视觉 Encoder 对应的 TensorRT engine 目录（纯文本推理时为空）
    std::string multimodalEngineDir{""};
    // 输入的 JSON 请求清单文件路径
    std::string inputFile;
    // 推理结果回写的 JSON 文件路径
    std::string outputFile{""};
    // 性能 Profiling 结果导出的路径
    std::string profileOutputFile{""};
    // 是否开启 TRT 插件层和内存分配层的冗余日志 (kVERBOSE)
    bool debug{false};
    // 是否将 Profiling 结果 Dump 到控制台标准输出
    bool dumpProfile{false};
    // 在正式记录性能前，空跑请求的次数，用于完成 CUDA Graph 捕获和 JIT 预热
    int32_t warmup{0};
    // 是否在终端直接输出生成的 Token 文本
    bool dumpOutput{false};
    // 强制覆盖的 Batch Size 大小（决定输入张量 [Batch, SeqLen] 中的 Batch 维）
    int32_t batchSize{-1};
    // 强制覆盖的最大生成长度（决定 KV Cache 显存预分配的最大深度）
    int64_t maxGenerateLength{-1};
    // Eagle 投机解码参数组
    EagleArgs eagleArgs;
};

/**
 * @desc: 向终端标准错误 (stderr) 输出工具的命令行使用帮助说明。
 * @params: programName - 可执行程序的名称 (通常传入 argv[0])
 * @return: 无
 * @others: 仅在参数解析失败或用户显式输入 --help 时触发调用，不涉及设备侧操作。
 */
void printUsage(char const* programName)
{
    // 输出包含所有长选项的示例命令行格式
    std::cerr << "Usage: " << programName
              << " [--help] [--engineDir=<path to engine directory>] [--multimodalEngineDir=<path to multimodal engine "
                 "directory>] [--inputFile=<path to input file>] [--outputFile=<path to output file>] "
                 "[--dumpProfile] [--profileOutputFile=<path to profile output file>] [--warmup=<number>] [--debug] "
                 "[--dumpOutput] [--batchSize=<number>] [--maxGenerateLength=<number>] [--eagle] "
                 "[--eagleDraftTopK=<number>] [--eagleDraftStep=<number>] "
                 "[--eagleVerifyTreeSize=<number>]"
              << std::endl;
    // 逐个解释各参数的作用域及其对推理管线的影响
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                    Display this help message" << std::endl;
    std::cerr << "  --inputFile               Path to input JSON file with requests" << std::endl;
    std::cerr << "  --engineDir               Path to engine directory" << std::endl;
    std::cerr << "  --multimodalEngineDir     Path to multimodal engine directory (optional)" << std::endl;
    std::cerr << "  --outputFile              Path to output JSON file (optional)" << std::endl;
    std::cerr << "  --dumpProfile             Dump profiling summary to console" << std::endl;
    std::cerr << "  --profileOutputFile       Path to profile JSON output file (optional)" << std::endl;
    std::cerr << "  --warmup                  Number of warmup runs using the first request (default: 0)" << std::endl;
    std::cerr << "  --debug                   Enable debug logging" << std::endl;
    std::cerr << "  --dumpOutput              Dump inference output to console" << std::endl;
    std::cerr << "  --batchSize               Override batch size from input file" << std::endl;
    std::cerr << "  --maxGenerateLength       Override max generate length from input file" << std::endl;
    std::cerr << "                            NOTE: For sampling parameters (temperature, top_p, top_k)," << std::endl;
    std::cerr << "                            please specify them in the input JSON file instead of CLI" << std::endl;
    std::cerr << "  --eagle                   Enable Eagle speculative decoding mode" << std::endl;
    std::cerr << "  --eagleDraftTopK          Number of tokens selected per drafting step (default: 10)" << std::endl;
    std::cerr << "                            Controls branching factor at each draft tree level" << std::endl;
    std::cerr << "  --eagleDraftStep          Number of drafting steps to perform (default: 6)" << std::endl;
    std::cerr << "                            Each step extends the draft tree by one more level" << std::endl;
    std::cerr << "  --eagleVerifyTreeSize     Number of tokens for base model verification (default: 60)" << std::endl;
    std::cerr << "                            Total draft tree size: 1 + topK + (step-1) * topK^2" << std::endl;
}

/**
 * @desc: 执行命令行参数捕获与校验，填充配置聚合体，并根据 Debug 开关设定 TensorRT 全局日志登记。
 * @params: args - 传入引用的配置结构体，将被解析结果填充
 * @params: argc - 命令行参数数量
 * @params: argv - 命令行参数字符数组指针
 * @return: bool - 解析是否成功。若为 false，进程应直接终止。
 * @others: 仅做纯字符串级逻辑校验，不进行 engine 文件的反序列化检查。
 */
bool parseLLMInferenceArgs(LLMInferenceArgs& args, int argc, char* argv[])
{
    // 定义 C 风格的 option 结构体数组，建立长选项字符串到枚举 ID 的映射关系
    static struct option inferenceOptions[] = {{"help", no_argument, 0, LLMInferenceOptionId::HELP},
        {"inputFile", required_argument, 0, LLMInferenceOptionId::INPUT_FILE},
        {"engineDir", required_argument, 0, LLMInferenceOptionId::ENGINE_DIR},
        {"multimodalEngineDir", required_argument, 0, LLMInferenceOptionId::MULTIMODAL_ENGINE_DIR},
        {"outputFile", required_argument, 0, LLMInferenceOptionId::OUTPUT_FILE},
        {"debug", no_argument, 0, LLMInferenceOptionId::DEBUG},
        {"dumpProfile", no_argument, 0, LLMInferenceOptionId::DUMP_PROFILE},
        {"profileOutputFile", required_argument, 0, LLMInferenceOptionId::PROFILE_OUTPUT_FILE},
        {"warmup", required_argument, 0, LLMInferenceOptionId::WARMUP},
        {"dumpOutput", no_argument, 0, LLMInferenceOptionId::DUMP_OUTPUT},
        {"eagle", no_argument, 0, LLMInferenceOptionId::EAGLE},
        {"eagleDraftTopK", required_argument, 0, LLMInferenceOptionId::EAGLE_DRAFT_TOP_K},
        {"eagleDraftStep", required_argument, 0, LLMInferenceOptionId::EAGLE_DRAFT_STEP},
        {"eagleVerifyTreeSize", required_argument, 0, LLMInferenceOptionId::EAGLE_VERIFY_TREE_SIZE},
        {"batchSize", required_argument, 0, LLMInferenceOptionId::BATCH_SIZE},
        {"maxGenerateLength", required_argument, 0, LLMInferenceOptionId::MAX_GENERATE_LENGTH}, {0, 0, 0, 0}};

    int opt;
    // 使用 getopt_long 遍历参数向量，"" 表示不接收任何短选项
    while ((opt = getopt_long(argc, argv, "", inferenceOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case LLMInferenceOptionId::HELP: args.help = true; return true;
        // 捕获指针指向的值，隐式转换为 std::string 存入 args
        case LLMInferenceOptionId::INPUT_FILE: args.inputFile = optarg; break;
        case LLMInferenceOptionId::ENGINE_DIR: args.engineDir = optarg; break;
        case LLMInferenceOptionId::MULTIMODAL_ENGINE_DIR: args.multimodalEngineDir = optarg; break;
        case LLMInferenceOptionId::OUTPUT_FILE: args.outputFile = optarg; break;
        case LLMInferenceOptionId::DEBUG: args.debug = true; break;
        case LLMInferenceOptionId::DUMP_PROFILE: args.dumpProfile = true; break;
        case LLMInferenceOptionId::PROFILE_OUTPUT_FILE: args.profileOutputFile = optarg; break;
        case LLMInferenceOptionId::WARMUP:
            try
            {
                // 将 C 字符串强转为 int，校验非负性，决定后续是否执行预热轮次
                args.warmup = std::stoi(optarg);
                if (args.warmup < 0)
                {
                    LOG_ERROR("Invalid warmup value: %s (must be non-negative)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid warmup value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::DUMP_OUTPUT: args.dumpOutput = true; break;
        case LLMInferenceOptionId::EAGLE: args.eagleArgs.enabled = true; break;
        case LLMInferenceOptionId::EAGLE_DRAFT_TOP_K:
            try
            {
                args.eagleArgs.draftTopK = std::stoi(optarg);
                if (args.eagleArgs.draftTopK <= 0)
                {
                    LOG_ERROR("Invalid eagleDraftTopK value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid eagleDraftTopK value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::EAGLE_DRAFT_STEP:
            try
            {
                args.eagleArgs.draftStep = std::stoi(optarg);
                if (args.eagleArgs.draftStep <= 0)
                {
                    LOG_ERROR("Invalid eagleDraftStep value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid eagleDraftStep value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::EAGLE_VERIFY_TREE_SIZE:
            try
            {
                args.eagleArgs.verifyTreeSize = std::stoi(optarg);
                if (args.eagleArgs.verifyTreeSize <= 0)
                {
                    LOG_ERROR("Invalid eagleVerifyTreeSize value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid eagleVerifyTreeSize value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::BATCH_SIZE:
            try
            {
                // 控制 TensorRT 推理时的 Batch 维度 [B, S] 中的 B，影响 Activation Buffer 显存预分配
                args.batchSize = std::stoi(optarg);
                if (args.batchSize <= 0)
                {
                    LOG_ERROR("Invalid batchSize value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid batchSize value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::MAX_GENERATE_LENGTH:
            try
            {
                // 控制 KV Cache 池的最大容量，超长将导致 OutOfMemory
                args.maxGenerateLength = std::stoll(optarg);
                if (args.maxGenerateLength <= 0)
                {
                    LOG_ERROR("Invalid maxGenerateLength value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid maxGenerateLength value: %s", optarg);
                return false;
            }
            break;
        default: return false;
        }
    }

    // --- 以下为必需参数完整性硬性校验 ---
    LOG_INFO("args.inputFile: %s", args.inputFile.c_str());
    if (args.inputFile.empty())
    {
        LOG_ERROR("ERROR: --inputFile is required");
        return false;
    }
    LOG_INFO("args.engineDir: %s", args.engineDir.c_str());
    if (args.engineDir.empty())
    {
        LOG_ERROR("ERROR: --engineDir is required");
        return false;
    }
    if (!args.multimodalEngineDir.empty())
    {
        LOG_INFO("args.multimodalEngineDir: %s", args.multimodalEngineDir.c_str());
    }

    if (args.outputFile.empty())
    {
        LOG_ERROR("ERROR: --outputFile is required");
        return false;
    }
    LOG_INFO("args.outputFile: %s", args.outputFile.c_str());

    if (args.dumpOutput)
    {
        LOG_INFO("args.dumpOutput: enabled");
    }

    if (!args.profileOutputFile.empty())
    {
        LOG_INFO("args.profileOutputFile: %s", args.profileOutputFile.c_str());
    }

    if (args.dumpProfile)
    {
        LOG_INFO("Profile dumping to console is enabled");
    }

    if (args.warmup > 0)
    {
        LOG_INFO("Warmup runs: %d", args.warmup);
    }

    if (args.eagleArgs.enabled)
    {
        LOG_INFO("Eagle mode enabled");
        LOG_INFO("Eagle draft topK: %d", args.eagleArgs.draftTopK);
        LOG_INFO("Eagle draft step: %d", args.eagleArgs.draftStep);
        LOG_INFO("Eagle verify tree size: %d", args.eagleArgs.verifyTreeSize);
    }

    // 根据 --debug 参数动态调节 TensorRT ILogger 报告器的过滤层级
    // 开启后将在 stdout 打印极详尽的 Kernel 绑定、Shape 推导和显存偏移量信息
    if (args.debug)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    }

    return true;
}

/**
 * @desc: 解析输入 JSON，提取系统配置与会话，通过滑窗机制组装出按 Batch 划分的 Request 张量描述容器。
 * @params: inputFilePath - 规范的输入 JSON 绝对或相对路径
 * @params: batchSizeOverride - CLI传入的 BatchSize，用于覆盖 JSON 中的设定
 * @params: maxGenerateLengthOverride - CLI传入的序列长度，用于覆盖 JSON 设定
 * @return: pair - 返回全局 LoRA 名字-路径映射表，以及组装好的结构化 Batch Request 列表
 * @others: 这里的 temperature、topP、topK 等采样超参将直接挂载到请求体，在 Logits 采样算子层执行数学分布截断。
 */
std::pair<std::unordered_map<std::string, std::string>, std::vector<rt::LLMGenerationRequest>> parseInputFile(
    std::filesystem::path const& inputFilePath, int32_t batchSizeOverride = -1, int64_t maxGenerateLengthOverride = -1)
{
    std::vector<rt::LLMGenerationRequest> batchedRequests;

    Json inputData;
    // 实例化文件输入流读取 JSON 数据到内存
    std::ifstream inputFileStream(inputFilePath);
    if (!inputFileStream.is_open())
    {
        LOG_ERROR("Failed to open input file: %s", inputFilePath.string().c_str());
        throw std::runtime_error("Failed to open input file: " + inputFilePath.string());
    }
    try
    {
        // 调用 nlohmann/json 将数据反序列化为树状对象
        inputData = Json::parse(inputFileStream);
        inputFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse input file with error: %s", e.what());
        throw std::runtime_error("Failed to parse input file: " + inputFilePath.string());
    }

    // 解析 Batch Size：优先读取 Override，若无则读 JSON 根节点，控制推理 Input 张量的 [B] 维度
    int batchSize = (batchSizeOverride != -1) ? batchSizeOverride : inputData.value("batch_size", 1);
    if (batchSize <= 0)
    {
        LOG_ERROR("Invalid batch_size value: %d (must be positive)", batchSize);
        throw std::runtime_error("Invalid batch_size value (must be positive)");
    }

    // 解析采样逻辑标量，这将在 Post-processing 时参与 Softmax/Top-K 概率截断运算
    float temperature = inputData.value("temperature", 1.0f);
    float topP = inputData.value("top_p", 0.8f);
    int64_t topK = inputData.value("top_k", 50);
    // 控制张量的 SeqLen 维度阈值，超出将触发 Early Stopping
    int64_t maxGenerateLength
        = (maxGenerateLengthOverride != -1) ? maxGenerateLengthOverride : inputData.value("max_generate_length", 256);
    if (maxGenerateLength <= 0)
    {
        LOG_ERROR(
            "Invalid max_generate_length value: %lld (must be positive)", static_cast<long long>(maxGenerateLength));
        throw std::runtime_error("Invalid max_generate_length value (must be positive)");
    }

    // 控制是否在 Tokenizer 阶段应用 Chat 模板（注入 <|im_start|> 等特殊符）
    bool applyChatTemplate = inputData.value("apply_chat_template", true);
    bool addGenerationPrompt = inputData.value("add_generation_prompt", true);
    bool enableThinking = inputData.value("enable_thinking", false);

    std::unordered_map<std::string, std::string> loraWeightsMap;
    // 扫描 JSON 中的可用 LoRA 权重池，存入字典以供 Runtime 初始化时按需加载
    if (inputData.contains("available_lora_weights") && inputData["available_lora_weights"].is_object())
    {
        auto const& availableLoraWeights = inputData["available_lora_weights"];
        for (auto const& [loraName, loraPath] : availableLoraWeights.items())
        {
            if (!loraPath.is_string())
            {
                LOG_ERROR("LoRA weight path for '%s' must be a string", loraName.c_str());
                throw std::runtime_error("LoRA weight path for '" + loraName + "' must be a string");
            }
            if (loraWeightsMap.find(loraName) != loraWeightsMap.end())
            {
                LOG_ERROR("Lora weights with name %s already exists", loraName.c_str());
                throw std::runtime_error("Lora weights with name " + loraName + " already exists");
            }
            loraWeightsMap[loraName] = loraPath.get<std::string>();
            LOG_INFO("Registered LoRA weights '%s' -> '%s'", loraName.c_str(), loraWeightsMap[loraName].c_str());
        }
    }

    // 提取核心 Request 列表
    if (inputData.contains("requests") && inputData["requests"].is_array())
    {
        auto& requestsArray = inputData["requests"];
        size_t numRequests = requestsArray.size();

        // 核心滑窗机制：以 batchSize 为步长聚合子请求。每个 batchedRequests 元素对应一次 TensorRT enqueueV3 提交
        for (size_t startIdx = 0; startIdx < numRequests; startIdx += batchSize)
        {
            rt::LLMGenerationRequest batchRequest;
            // 绑定采样参数到当前批次上下文
            batchRequest.temperature = temperature;
            batchRequest.topP = topP;
            batchRequest.topK = topK;
            batchRequest.maxGenerateLength = maxGenerateLength;
            batchRequest.applyChatTemplate = applyChatTemplate;
            batchRequest.addGenerationPrompt = addGenerationPrompt;
            batchRequest.enableThinking = enableThinking;

            std::string batchLoraWeightsName = "";
            bool firstInBatch = true;

            // 切片：计算当前 Batch 的实际末尾索引（防止越界）
            size_t endIdx = std::min(startIdx + batchSize, numRequests);
            // 遍历并组装当前 Batch 内的每一个子 Request
            for (size_t requestIdx = startIdx; requestIdx < endIdx; ++requestIdx)
            {
                auto const& requestItem = requestsArray[requestIdx];

                if (!requestItem.is_object())
                {
                    LOG_ERROR("Each request must be an object with 'messages' key");
                    throw std::runtime_error("Each request must be an object with 'messages' key");
                }

                // 决定是否持久化该批次的 System Prompt 产生的 KV Cache 张量，便于后续复用减少显存写入开销
                bool saveSystemPromptKVCache = requestItem.value("save_system_prompt_kv_cache", false);
                if (saveSystemPromptKVCache)
                {
                    batchRequest.saveSystemPromptKVCache = true;
                }

                if (!requestItem.contains("messages") || !requestItem["messages"].is_array())
                {
                    LOG_ERROR("Each request object must contain a 'messages' array");
                    throw std::runtime_error("Each request object must contain a 'messages' array");
                }

                auto const& messagesArray = requestItem["messages"];

                // LoRA 一致性校验：单次 GPU Launch 无法为同 batch 的不同请求加载不同 LoRA 权重矩阵
                std::string requestLoraName = "";
                if (requestItem.contains("lora_name") && !requestItem["lora_name"].is_null())
                {
                    requestLoraName = requestItem["lora_name"].get<std::string>();

                    if (!requestLoraName.empty() && loraWeightsMap.find(requestLoraName) == loraWeightsMap.end())
                    {
                        LOG_ERROR("LoRA name '%s' not found in available_lora_weights", requestLoraName.c_str());
                        throw std::runtime_error(
                            "LoRA name '" + requestLoraName + "' not found in available_lora_weights");
                    }
                }

                // 强制对齐同 Batch 内的 LoRA Name
                if (firstInBatch)
                {
                    batchLoraWeightsName = requestLoraName;
                    firstInBatch = false;
                }
                else
                {
                    if (requestLoraName != batchLoraWeightsName)
                    {
                        LOG_ERROR(
                            "All requests within the same batch must use the same LoRA weights. Batch has %d requests.",
                            static_cast<int>(endIdx - startIdx));
                        throw std::runtime_error("Different LoRA weights within the same batch are not supported");
                    }
                }

                std::vector<rt::Message> chatMessages;
                std::vector<rt::imageUtils::ImageData> imageBuffers;

                // 提取多轮对话的 message 细节
                for (auto const& messageJson : messagesArray)
                {
                    if (!messageJson.contains("role") || !messageJson.contains("content"))
                    {
                        LOG_ERROR("Each message must have 'role' and 'content' fields");
                        throw std::runtime_error("Each message must have 'role' and 'content' fields");
                    }

                    rt::Message chatMsg;
                    chatMsg.role = messageJson["role"].get<std::string>();

                    auto const& contentJson = messageJson["content"];

                    // 如果是纯文本内容
                    if (contentJson.is_string())
                    {
                        rt::Message::MessageContent msgContent;
                        msgContent.type = "text";
                        msgContent.content = contentJson.get<std::string>();
                        chatMsg.contents.push_back(msgContent);
                    }
                    // 如果是多模态混合内容数组
                    else if (contentJson.is_array())
                    {
                        for (auto const& contentItemJson : contentJson)
                        {
                            if (!contentItemJson.contains("type"))
                            {
                                LOG_ERROR("Each content item must have a 'type' field");
                                throw std::runtime_error("Each content item must have a 'type' field");
                            }

                            rt::Message::MessageContent msgContent;
                            msgContent.type = contentItemJson["type"].get<std::string>();

                            if (msgContent.type == "text")
                            {
                                msgContent.content = contentItemJson["text"].get<std::string>();
                            }
                            else if (msgContent.type == "image")
                            {
                                msgContent.content = contentItemJson["image"].get<std::string>();
                                // 多模态处理点：将图像文件读入 Host (CPU) 内存缓冲区，并作为 ImageData 留存。
                                // 该 Buffer 后续将在 Multimodal Runtime 阶段触发 cudaMemcpy 拷贝入 Device 显存。
                                auto image = rt::imageUtils::loadImageFromFile(msgContent.content);
                                if (image.buffer != nullptr)
                                {
                                    imageBuffers.push_back(std::move(image));
                                }
                            }
                            else
                            {
                                LOG_ERROR("Content type must be 'text', 'image', but got: %s", msgContent.type.c_str());
                                throw std::runtime_error(format::fmtstr(
                                    "Content type must be 'text', 'image', but got: %s", msgContent.type.c_str()));
                            }

                            chatMsg.contents.push_back(msgContent);
                        }
                    }
                    else
                    {
                        LOG_ERROR("Message content must be a string or an array");
                        throw std::runtime_error("Message content must be a string or an array");
                    }

                    chatMessages.push_back(chatMsg);
                }

                // 组装当前 request，将所有提取出的 messages 与 host memory 中的 imageBuffers 转移给 request 对象
                rt::LLMGenerationRequest::Request request;
                request.messages = std::move(chatMessages);
                request.imageBuffers = std::move(imageBuffers);
                batchRequest.requests.push_back(std::move(request));
            }

            if (!batchLoraWeightsName.empty())
            {
                batchRequest.loraWeightsName = batchLoraWeightsName;
            }

            // 完成一个 Batch 的拼接，放入全局提交队列
            batchedRequests.push_back(std::move(batchRequest));
        }
    }
    else
    {
        LOG_ERROR("'requests' array not found in input file");
        throw std::runtime_error("'requests' array not found in input file");
    }

    return std::make_pair(std::move(loraWeightsMap), std::move(batchedRequests));
}

/**
 * @desc: 主程序入口。负责驱动整个推理状态机：解析参数 -> 加载 TRT 插件 -> 反序列化 Engine -> 
 * 构建并绑定 IExecutionContext 的显存映射 -> CUDA Graph 预热 -> Batch 提交主循环 -> 收尾 Dump。
 * @params: argc - 参数个数
 * @params: argv - 参数数组指针
 * @return: int - 进程退出状态码 (EXIT_SUCCESS / EXIT_FAILURE)
 * @others: 这里的 cudaStream_t 定义了整个推理过程的异步执行管线。内存同步点隐式包含在 Runtime 的 handleRequest 中。
 */
int main(int argc, char* argv[])
{
    LLMInferenceArgs args;
    // 1. 初始化命令行环境
    if (!parseLLMInferenceArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }
    bool profilerEnabled = args.dumpProfile;
    
    // 2. 实例化显存探针对象。若为 dGPU 架构，start() 会产生一个旁路线程不断调用 nvml / cudaMemGetInfo
    MemoryMonitor memoryMonitor;
    if (profilerEnabled)
    {
        memoryMonitor.start();
    }

    // 3. 关键动作：装载 Custom Plugin (如 FlashAttention, LayerNorm 等算子融合节点)。
    // 必须在 deserializeCudaEngine 前注册到 TensorRT Plugin Registry，否则 Engine 解析器无法识别网络结构。
    // 返回的 pluginHandles 必须保持存活直到推理任务完全结束。
    auto pluginHandles = loadEdgellmPluginLib();
    
    std::unordered_map<std::string, std::string> loraWeightsMap;
    std::vector<rt::LLMGenerationRequest> batchedRequests;
    try
    {
        // 4. 读取 JSON 数据，提取出 Batch 请求队列
        std::tie(loraWeightsMap, batchedRequests)
            = parseInputFile(args.inputFile, args.batchSize, args.maxGenerateLength);
        LOG_INFO("Successfully parsed %zu LoRA weights from input file.", loraWeightsMap.size());
        LOG_INFO("Successfully parsed %zu batches of requests from input file.", batchedRequests.size());
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to parse input file: %s", e.what());
        return EXIT_FAILURE;
    }

    if (batchedRequests.empty())
    {
        LOG_ERROR("No valid requests found in input file.");
        return EXIT_FAILURE;
    }

    std::unique_ptr<rt::LLMInferenceRuntime> llmInferenceRuntime{nullptr};
    std::unique_ptr<rt::LLMInferenceSpecDecodeRuntime> eagleInferenceRuntime{nullptr};
    
    // 5. 创建 CUDA 执行流。所有涉及 Tensor 计算、KV Cache 显存滑动、Logits 通信的任务均派发给此异步流
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 6. Runtime 实例化与 Engine 装载
    if (args.eagleArgs.enabled)
    {
        // Eagle 投机解码分支
        if (!loraWeightsMap.empty())
        {
            LOG_WARNING("Eagle mode does not support LoRA weights. Ignoring LoRA weights.");
        }

        // 定义 Eagle 树状生成策略
        rt::EagleDraftingConfig draftingConfig{
            args.eagleArgs.draftTopK, args.eagleArgs.draftStep, args.eagleArgs.verifyTreeSize};
        try
        {
            // 在构造期，Runtime 将自动读取 engineDir，调用 TRT IRuntime 反序列化出 ICudaEngine。
            // 随后建立对应的 IExecutionContext，并分配 Activation Buffer 显存。
            eagleInferenceRuntime = std::make_unique<rt::LLMInferenceSpecDecodeRuntime>(
                args.engineDir, args.multimodalEngineDir, draftingConfig, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize LLMInferenceSpecDecodeRuntime: %s", e.what());
            return EXIT_FAILURE;
        }

        // CUDA Graph 捕获：记录 Eagle 框架下 Draft 和 Base 阶段的显存读写与算子调用依赖图
        // 此举通过去除 Host 端 CPU 发射 Kernel 的 Overhead，极大增强高频 Decode 操作的吞吐量
        bool const draftProposalCaptureStatus = eagleInferenceRuntime->captureDraftProposalCudaGraph(stream);
        if (!draftProposalCaptureStatus)
        {
            LOG_WARNING(
                "Failed to capture CUDA graph for draft proposal usage, proceeding with normal engine execution.");
        }

        bool const draftAcceptCaptureStatus = eagleInferenceRuntime->captureDraftAcceptDecodeTokenCudaGraph(stream);
        if (!draftAcceptCaptureStatus)
        {
            LOG_WARNING(
                "Failed to capture CUDA graph for draft accept decode token usage, proceeding with normal engine "
                "execution.");
        }

        bool const baseCaptureStatus = eagleInferenceRuntime->captureBaseVerificationCudaGraph(stream);
        if (!baseCaptureStatus)
        {
            LOG_WARNING(
                "Failed to capture CUDA graph for base model verification usage, proceeding with normal engine "
                "execution.");
        }
    }
    else
    {
        // 标准自回归生成分支
        try
        {
            // 创建基础运行上下文，绑定 LoRA 字典与指定的 CUDA Stream
            llmInferenceRuntime = std::make_unique<rt::LLMInferenceRuntime>(
                args.engineDir, args.multimodalEngineDir, loraWeightsMap, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize LLMInferenceRuntime: %s", e.what());
            return EXIT_FAILURE;
        }
        
        // 捕获标准自回归流程的 CUDA Graph，固化 Decode 循环中的显存指针地址映射
        if (!llmInferenceRuntime->captureDecodingCUDAGraph(stream))
        {
            LOG_WARNING("Failed to capture CUDA graph for decoding usage, proceeding with normal engine execution.");
        }
    }

    // 7. Warmup 预热动作
    if (args.warmup > 0)
    {
        // 临时关闭 Profiling 计时器，防止脏数据影响最终 Benchmark 指标
        setProfilingEnabled(false);
        LOG_INFO("Starting warmup with %d runs using the first request...", args.warmup);
        auto& firstRequest = batchedRequests[0];

        // 迭代预热：迫使 CUDA Runtime 执行 JIT 编译并填充部分设备的 Cache 热区
        for (int32_t warmupRun = 0; warmupRun < args.warmup; ++warmupRun)
        {
            rt::LLMGenerationResponse warmupResponse;
            bool requestStatus = false;
            // 真实触发 enqueueV3 推理。KV Cache 内部指针将在 Context -> Generate 中循环递增
            if (args.eagleArgs.enabled)
            {
                requestStatus = eagleInferenceRuntime->handleRequest(firstRequest, warmupResponse, stream);
            }
            else
            {
                requestStatus = llmInferenceRuntime->handleRequest(firstRequest, warmupResponse, stream);
            }

            if (!requestStatus)
            {
                LOG_ERROR("Warmup run %d/%d failed", warmupRun + 1, args.warmup);
                return EXIT_FAILURE;
            }
        }
        LOG_INFO("Warmup of %d runs completed. Starting actual benchmark runs...", args.warmup);
    }

    if (profilerEnabled)
    {
        // 正式开启内置算子耗时埋点
        setProfilingEnabled(true);
    }

    nlohmann::json outputData;
    outputData["input_file"] = args.inputFile;
    outputData["responses"] = nlohmann::json::array();

    bool hasFailedRequest = false;
    std::string errorMessage = "TensorRT Edge LLM cannot handle this request. Fails.";
    size_t failedCount = 0;

    // 8. 核心执行循环：遍历已切片打包完毕的 batches
    LOG_INFO("Processing %zu batched requests...", batchedRequests.size());
    for (size_t requestIdx = 0; requestIdx < batchedRequests.size(); ++requestIdx)
    {
        auto& request = batchedRequests[requestIdx];
        rt::LLMGenerationResponse response;

        size_t progressInterval = std::max(size_t(1), std::min(batchedRequests.size() / 10, size_t(100)));
        if ((requestIdx + 1) % progressInterval == 0 || requestIdx == 0 || requestIdx == batchedRequests.size() - 1)
        {
            LOG_INFO("Progress: %zu/%zu (%f%%)", requestIdx + 1, batchedRequests.size(),
                100.0 * (requestIdx + 1) / batchedRequests.size());
        }

        bool requestStatus = false;
        
        // --- Runtime Execution Pipeline ---
        // 此处的 handleRequest 会执行完整的流水线：
        // 1. Tokenizer 编码，产生 Input IDs 的张量 [B, S]
        // 2. Prefill (Context Phase) 发射: IExecutionContext::enqueueV3 异步下发
        // 3. 进入 Incremental Decode (Generate Phase) 循环: 每次步进产出 logits
        // 4. Sampler: CPU/GPU 同步，对概率分布施加 Top-K / Top-P 过滤操作
        // 5. KV 缓冲池游标滑动更新。
        if (args.eagleArgs.enabled)
        {
            requestStatus = eagleInferenceRuntime->handleRequest(request, response, stream);
        }
        else
        {
            requestStatus = llmInferenceRuntime->handleRequest(request, response, stream);
        }

        // 推理返回，若用户要求，将在控制台打印 Host 端解后的生成字符串
        if (requestStatus)
        {
            if (args.dumpOutput)
            {
                for (size_t batchIdx = 0; batchIdx < response.outputTexts.size(); ++batchIdx)
                {
                    LOG_INFO("Response for request %zu batch %zu: %s", requestIdx, batchIdx,
                        response.outputTexts[batchIdx].c_str());
                }
            }
        }
        else
        {
            hasFailedRequest = true;
            failedCount++;
            LOG_ERROR("*** FAILED *** Request %zu failed to process!", requestIdx);
        }

        // 9. 构建 JSON Output
        for (size_t batchIdx = 0; batchIdx < request.requests.size(); ++batchIdx)
        {
            nlohmann::json responseJson;
            // 清理非法 UTF-8 字符，防止 JSON dump 序列化时崩溃抛异常
            std::string outputText = requestStatus ? response.outputTexts[batchIdx] : errorMessage;
            responseJson["output_text"] = sanitizeUtf8ForJson(outputText);
            responseJson["request_idx"] = requestIdx;
            responseJson["batch_idx"] = batchIdx;
            
            // 组装回显的消息树
            nlohmann::json messagesJson = nlohmann::json::array();
            for (auto const& msg : request.requests[batchIdx].messages)
            {
                nlohmann::json msgJson;
                msgJson["role"] = msg.role;
                msgJson["content"] = nlohmann::json::array();
                for (auto const& content : msg.contents)
                {
                    nlohmann::json contentJson;
                    contentJson["type"] = content.type;
                    if (content.type == "text")
                    {
                        contentJson["text"] = content.content;
                    }
                    else if (content.type == "image")
                    {
                        contentJson["image"] = content.content;
                    }
                    else if (content.type == "video")
                    {
                        contentJson["video"] = content.content;
                    }
                    msgJson["content"].push_back(contentJson);
                }
                messagesJson.push_back(msgJson);
            }
            responseJson["messages"] = messagesJson;
            // 记录应用 Chat 模板后产生的原始 Prompt 张量映射的对应文本，用于 Debug Prompt Token 拼接错误
            responseJson["formatted_system_prompt"] = request.formattedRequests[batchIdx].formattedSystemPrompt;
            responseJson["formatted_complete_request"] = request.formattedRequests[batchIdx].formattedCompleteRequest;
            outputData["responses"].push_back(responseJson);
        }
    }

    LOG_INFO("Processing complete: %zu/%zu batched requests successful", batchedRequests.size() - failedCount,
        batchedRequests.size());
    if (failedCount > 0)
    {
        LOG_ERROR("*** %zu BATCHED REQUESTS FAILED ***", failedCount);
    }

    // 10. Profiler 结束并搜集数据
    if (profilerEnabled)
    {
        setProfilingEnabled(false);
        // join() 旁路统计线程，锁定峰值显存读数
        memoryMonitor.stop();
    }

    // 打印性能报告至终端 stdout
    if (args.dumpProfile)
    {
        std::ostringstream profileOutput;
        profileOutput << std::endl;
        profileOutput << "=== Performance Summary ===" << std::endl;
        if (args.eagleArgs.enabled)
        {
            auto prefillMetrics = eagleInferenceRuntime->getPrefillMetrics();
            auto eagleGenerationMetrics = eagleInferenceRuntime->getEagleGenerationMetrics();
            auto multimodalMetrics = eagleInferenceRuntime->getMultimodalMetrics();
            outputPrefillProfile(profileOutput, prefillMetrics);
            outputEagleGenerationProfile(profileOutput, eagleGenerationMetrics);
            outputMultimodalProfile(profileOutput, multimodalMetrics);
            outputMemoryProfile(profileOutput, memoryMonitor);
        }
        else
        {
            auto multimodalMetrics = llmInferenceRuntime->getMultimodalMetrics();
            outputPrefillProfile(profileOutput, llmInferenceRuntime->getPrefillMetrics());
            outputGenerationProfile(profileOutput, llmInferenceRuntime->getGenerationMetrics());
            outputMultimodalProfile(profileOutput, multimodalMetrics);
            outputMemoryProfile(profileOutput, memoryMonitor);
        }
        profileOutput << "=====================================" << std::endl;
        LOG_INFO("%s", profileOutput.str().c_str());
    }

    // 将 Profiling 数据结构序列化写入外挂磁盘文件
    if (!args.profileOutputFile.empty())
    {
        try
        {
            nlohmann::json profileJson;

            if (args.eagleArgs.enabled)
            {
                auto prefillMetrics = eagleInferenceRuntime->getPrefillMetrics();
                auto eagleGenerationMetrics = eagleInferenceRuntime->getEagleGenerationMetrics();
                auto multimodalMetrics = eagleInferenceRuntime->getMultimodalMetrics();

                addJsonPrefillSummary(profileJson, prefillMetrics);
                addJsonEagleGenerationSummary(profileJson, eagleGenerationMetrics);
                addJsonMultimodalSummary(profileJson, multimodalMetrics);
                addJsonTimingStages(profileJson);
                addJsonMemorySummary(profileJson, memoryMonitor);
            }
            else
            {
                auto multimodalMetrics = llmInferenceRuntime->getMultimodalMetrics();
                addJsonPrefillSummary(profileJson, llmInferenceRuntime->getPrefillMetrics());
                addJsonGenerationSummary(profileJson, llmInferenceRuntime->getGenerationMetrics());
                addJsonMultimodalSummary(profileJson, multimodalMetrics);
                addJsonTimingStages(profileJson);
                addJsonMemorySummary(profileJson, memoryMonitor);
            }

            std::ofstream profileFile(args.profileOutputFile);
            if (profileFile.is_open())
            {
                profileFile << profileJson.dump(2);
                profileFile.close();
                LOG_INFO("Profile data exported to: %s", args.profileOutputFile.c_str());
            }
            else
            {
                LOG_ERROR("Failed to open profile output file: %s", args.profileOutputFile.c_str());
                return EXIT_FAILURE;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to write profile output file: %s", e.what());
            return EXIT_FAILURE;
        }
    }

    // 将最终生成的 response array 序列化为磁盘 JSON 文件供评测脚本（如 ROUGE/BLEU 评估）调用
    try
    {
        std::ofstream outputFile(args.outputFile);
        if (outputFile.is_open())
        {
            outputFile << outputData.dump(4);
            outputFile.close();
            LOG_INFO("All responses exported to: %s", args.outputFile.c_str());
        }
        else
        {
            LOG_ERROR("Failed to open output file: %s", args.outputFile.c_str());
            return EXIT_FAILURE;
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to write output file: %s", e.what());
        return EXIT_FAILURE;
    }

    // 清理资源流并安全退出。析构执行顺序: stream -> Runtime (含 engine) -> pluginHandles
    return hasFailedRequest ? EXIT_FAILURE : EXIT_SUCCESS;
}
}