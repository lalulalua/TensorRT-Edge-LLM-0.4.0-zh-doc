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

#include "common/trtUtils.h"
#include "memoryMonitor.h"
#include "profileFormatter.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/llmInferenceRuntime.h"
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
 * @desc: llm_inference 示例进程：将磁盘 JSON 请求转为 `rt::LLMGenerationRequest`，加载 TensorRT 插件与 engine 文件，经 `cudaStream_t` 调用 `handleRequest`，并把文本结果与可选 profile 写回 JSON。不包含 Prefill/Decode 内核、KV 张量布局或 Top-K/Top-P 数学实现（见 `cpp/runtime/llmInferenceRuntime.cpp` 与 `.cursor/TIER_B_LLM_RUNTIME_ANNOTATION.md`）。
 * @params: 无（翻译单元级说明）
 * @return: 无
 * @others: 所有设备侧张量 shape 由 engine 构建期 profile 与运行时 runner 决定；本文件仅传递标量超参（temperature/top_p/top_k/max_generate_length）与路径字符串。
 */

/**
 * @desc: 与 `getopt_long` 的 `val` 字段对应的 CLI 选项枚举，用于 `switch` 分发。
 * @params: 无
 * @return: 无
 * @others: 取值自 900 起，避免与 ASCII 单字符短选项返回值冲突；与 `inferenceOptions[]` 顺序一致。
 */
// Enum for command line option IDs (using traditional enum for C library compatibility)
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
 * @desc: Eagle 投机解码 CLI 超参容器；传入 `LLMInferenceSpecDecodeRuntime` 的 `EagleDraftingConfig`，影响 draft 树形与 base 验证 batch 宽度。
 * @params: 无（聚合类型）
 * @return: 无
 * @others: `draftTopK`/`draftStep` 决定候选树规模；`verifyTreeSize` 须不大于理论节点数 `1 + draftTopK + (draftStep-1)*draftTopK^2`。与标准路径互斥，且当前实现忽略 LoRA 权重表。
 */
struct EagleArgs
{
    bool enabled{false};
    int32_t draftTopK{10};
    int32_t draftStep{6};
    int32_t verifyTreeSize{60};
};

/**
 * @desc: 从 `argv` 解析得到的运行期开关与路径；决定加载哪套 engine 目录、是否打 profile、是否走 Eagle，以及对输入 JSON 中 `batch_size` / `max_generate_length` 的覆盖。
 * @params: 无（聚合类型）
 * @return: 无
 * @others: `engineDir` — 文本 LLM 的 engine 文件目录；`multimodalEngineDir` — 视觉 encoder 的 engine 目录（可空）。`batchSize`/`maxGenerateLength` 为 -1 表示不覆盖 JSON。`temperature`/`top_p`/`top_k` **不能**从 CLI 改，仅 JSON（见 `parseInputFile`）。
 */
struct LLMInferenceArgs
{
    bool help{false};
    std::string engineDir;
    std::string multimodalEngineDir{""};
    std::string inputFile;
    std::string outputFile{""};
    std::string profileOutputFile{""};
    bool debug{false};
    bool dumpProfile{false};
    int32_t warmup{0};
    bool dumpOutput{false};
    int32_t batchSize{-1};
    int64_t maxGenerateLength{-1};
    EagleArgs eagleArgs;
};

/**
 * @desc: 向 stderr 输出用法与选项列表（英文），与 `parseLLMInferenceArgs` 接受的 long option 名一致。
 * @params: programName — 可执行文件展示名，通常为 `argv[0]`
 * @return: 无
 * @others: 不参与 I/O 重定向以外的逻辑；调用方在解析失败或 `--help` 时触发。
 */
void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] [--engineDir=<path to engine directory>] [--multimodalEngineDir=<path to multimodal engine "
                 "directory>] [--inputFile=<path to input file>] [--outputFile=<path to output file>] "
                 "[--dumpProfile] [--profileOutputFile=<path to profile output file>] [--warmup=<number>] [--debug] "
                 "[--dumpOutput] [--batchSize=<number>] [--maxGenerateLength=<number>] [--eagle] "
                 "[--eagleDraftTopK=<number>] [--eagleDraftStep=<number>] "
                 "[--eagleVerifyTreeSize=<number>]"
              << std::endl;
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
 * @desc: 使用 `getopt_long` 解析长选项，填充 `LLMInferenceArgs`；校验 `--inputFile`/`--engineDir`/`--outputFile` 非空；按 `--debug` 设置全局 TensorRT `gLogger` 级别。
 * @params: args — 输出，聚合路径与开关；argc — 参数个数；argv — 参数向量（会被 getopt 重排，仅应在解析阶段使用）
 * @return: true 表示可继续运行（含 `--help`）；false 表示非法选项或数值
 * @others: 不打开 JSON 文件；与 `parseInputFile` 正交。`optarg` 指向当前选项的 C 字符串参数（由 getopt 管理生命周期，应立即拷贝到 `std::string` 的选项此处已直接赋值指针，符合 getopt 在单次解析内的用法）。
 */
bool parseLLMInferenceArgs(LLMInferenceArgs& args, int argc, char* argv[])
{
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
    // 循环消费 argv；返回 -1 表示选项结束。空字符串 "" 作为 shortopts 表示只接受长选项。
    while ((opt = getopt_long(argc, argv, "", inferenceOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case LLMInferenceOptionId::HELP: args.help = true; return true;
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

    // 以下必填项与 JSON 解析无关：确保后续能打开输入文件、加载 engine、写出结果路径。
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

    // TensorRT 插件与 engine 加载阶段的日志粒度；VERBOSE 便于排查 plan 绑定与插件注册问题。
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
 * @desc: 将符合 `INPUT_FORMAT.md` 的 JSON 转为 `rt::LLMGenerationRequest` 向量：读取全局采样与模板标志，按 `batch_size`（可被 CLI 覆盖）将 `requests` 滑窗分组，并校验同 batch 内 LoRA 名一致。
 * @params: inputFilePath — 输入 JSON 路径；batchSizeOverride — 若 >=1 则覆盖 JSON 的 `batch_size`，否则用 JSON；maxGenerateLengthOverride — 若 >=1 则覆盖 JSON 的 `max_generate_length`
 * @return: `first` 为 LoRA 逻辑名到 `.safetensors` 路径映射；`second` 为每个合成 batch 的请求对象列表；异常时抛出 `std::runtime_error`
 * @others: `temperature`/`top_p`/`top_k` 写入每个 `LLMGenerationRequest`，**实际采样在运行时** `sampleTokens` 路径执行，本函数不做概率计算。图像经 `loadImageFromFile` 留在 **Host** 侧 `ImageData`，GPU 上传在 `handleRequest` 内。张量 shape（如 `[B,T]`）由 engine 与 runner 决定，此处仅构造对话结构与标量超参。
 */
std::pair<std::unordered_map<std::string, std::string>, std::vector<rt::LLMGenerationRequest>> parseInputFile(
    std::filesystem::path const& inputFilePath, int32_t batchSizeOverride = -1, int64_t maxGenerateLengthOverride = -1)
{
    std::vector<rt::LLMGenerationRequest> batchedRequests;

    Json inputData;
    // 以文本方式读入整文件；超大 JSON 时峰值内存约等于文件大小量级。
    std::ifstream inputFileStream(inputFilePath);
    if (!inputFileStream.is_open())
    {
        LOG_ERROR("Failed to open input file: %s", inputFilePath.string().c_str());
        throw std::runtime_error("Failed to open input file: " + inputFilePath.string());
    }
    try
    {
        // 整文件解析为 nlohmann::json；大文件时注意内存占用（评测 JSON 可能较大）。
        inputData = Json::parse(inputFileStream);
        inputFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse input file with error: %s", e.what());
        throw std::runtime_error("Failed to parse input file: " + inputFilePath.string());
    }

    // 全局超参：所有合成 batch 共享；写入 `LLMGenerationRequest` 后由运行时统一读取。
    int batchSize = (batchSizeOverride != -1) ? batchSizeOverride : inputData.value("batch_size", 1);
    if (batchSize <= 0)
    {
        LOG_ERROR("Invalid batch_size value: %d (must be positive)", batchSize);
        throw std::runtime_error("Invalid batch_size value (must be positive)");
    }

    // 采样标量：无 Tensor shape；在 runtime 内应用于 logits 分布（实现见 Tier B 文档所列源文件）。
    float temperature = inputData.value("temperature", 1.0f);
    float topP = inputData.value("top_p", 0.8f);
    int64_t topK = inputData.value("top_k", 50);
    int64_t maxGenerateLength
        = (maxGenerateLengthOverride != -1) ? maxGenerateLengthOverride : inputData.value("max_generate_length", 256);
    if (maxGenerateLength <= 0)
    {
        LOG_ERROR(
            "Invalid max_generate_length value: %lld (must be positive)", static_cast<long long>(maxGenerateLength));
        throw std::runtime_error("Invalid max_generate_length value (must be positive)");
    }

    bool applyChatTemplate = inputData.value("apply_chat_template", true);
    bool addGenerationPrompt = inputData.value("add_generation_prompt", true);
    bool enableThinking = inputData.value("enable_thinking", false);

    std::unordered_map<std::string, std::string> loraWeightsMap;
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

    // requests 数组按 batchSize 滑窗聚合为多个 LLMGenerationRequest；同一 batch 内 lora_name 必须一致（权重名映射在返回的 map 中）。
    if (inputData.contains("requests") && inputData["requests"].is_array())
    {
        auto& requestsArray = inputData["requests"];
        size_t numRequests = requestsArray.size();

        for (size_t startIdx = 0; startIdx < numRequests; startIdx += batchSize)
        {
            rt::LLMGenerationRequest batchRequest;
            batchRequest.temperature = temperature;
            batchRequest.topP = topP;
            batchRequest.topK = topK;
            batchRequest.maxGenerateLength = maxGenerateLength;
            batchRequest.applyChatTemplate = applyChatTemplate;
            batchRequest.addGenerationPrompt = addGenerationPrompt;
            batchRequest.enableThinking = enableThinking;

            // 同一 batch 内所有子请求的 lora_name 须相同，否则运行时无法单次 launch 切换多套权重。
            std::string batchLoraWeightsName = "";
            bool firstInBatch = true;

            size_t endIdx = std::min(startIdx + batchSize, numRequests);
            for (size_t requestIdx = startIdx; requestIdx < endIdx; ++requestIdx)
            {
                auto const& requestItem = requestsArray[requestIdx];

                if (!requestItem.is_object())
                {
                    LOG_ERROR("Each request must be an object with 'messages' key");
                    throw std::runtime_error("Each request must be an object with 'messages' key");
                }

                // 任一条目置 true 则整 batch 标记 saveSystemPromptKVCache（示例程序限制）；适合初始化阶段单独发 batch 缓存长 system。
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

                    if (contentJson.is_string())
                    {
                        rt::Message::MessageContent msgContent;
                        msgContent.type = "text";
                        msgContent.content = contentJson.get<std::string>();
                        chatMsg.contents.push_back(msgContent);
                    }
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
                                // 图像解码在 CPU 侧完成，buffer 挂在 request.imageBuffers，后续由 runtime 负责拷到 GPU 并走视觉 engine。
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

                rt::LLMGenerationRequest::Request request;
                request.messages = std::move(chatMessages);
                request.imageBuffers = std::move(imageBuffers);
                batchRequest.requests.push_back(std::move(request));
            }

            if (!batchLoraWeightsName.empty())
            {
                batchRequest.loraWeightsName = batchLoraWeightsName;
            }

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
 * @desc: 进程入口：CLI → 可选 MemoryMonitor → `loadEdgellmPluginLib` → JSON → 构造 `LLMInferenceRuntime` 或 `LLMInferenceSpecDecodeRuntime` → CUDA Graph 捕获 → warmup → 逐 batch `handleRequest` → 控制台/profile JSON/响应 JSON 写出。
 * @params: argc — 参数个数；argv — 参数向量
 * @return: `EXIT_SUCCESS` 或 `EXIT_FAILURE`（任一 batch 的 `handleRequest` 失败则失败）
 * @others: `pluginHandles` 析构前须保持有效，以便插件在 engine 反序列化及整段推理期间已注册。`cudaStream_t stream` 贯穿 `handleRequest` 与 Graph capture，**同步点均在 runtime 实现内部**（本文件不显式 `cudaStreamSynchronize`）。Prefill/Decode、KV 更新、logits 采样见 Tier B 所列 `llmInferenceRuntime.cpp` 等。
 */
int main(int argc, char* argv[])
{
    // 创建命令行参数承载对象，用于保存解析结果。
    LLMInferenceArgs args;
    // 解析 CLI 参数；失败时返回 false。
    if (!parseLLMInferenceArgs(args, argc, argv))
    {
        // 参数非法时打印用法帮助。
        printUsage(argv[0]);
        // 参数错误按失败退出。
        return EXIT_FAILURE;
    }
    // 处理 `--help` 分支。
    if (args.help)
    {
        // 显示帮助信息。
        printUsage(argv[0]);
        // help 不算错误，成功退出。
        return EXIT_SUCCESS;
    }
    // 记录是否开启 profile 输出。
    bool profilerEnabled = args.dumpProfile;
    // 创建内存监视器实例（可选启用）。
    MemoryMonitor memoryMonitor;
    // dGPU 时启动异步 `cudaMemGetInfo` 轮询；iGPU 路径不启线程，详见 `memoryMonitor.cpp`。
    // 若开启 profile，则启动内存监控线程。
    if (profilerEnabled)
    {
        // 开始周期性采样显存/内存指标。
        memoryMonitor.start();
    }

    // 必须在创建 Runtime / 反序列化 engine 之前完成；返回的句柄容器需存活至进程退出前。
    // 动态加载 TensorRT Edge LLM 插件库并保留句柄生命周期。
    auto pluginHandles = loadEdgellmPluginLib();
    // 保存输入 JSON 中解析出的 LoRA 名称到路径映射。
    std::unordered_map<std::string, std::string> loraWeightsMap;
    // 保存批处理后的请求列表。
    std::vector<rt::LLMGenerationRequest> batchedRequests;
    // 通过异常保护输入文件解析过程。
    try
    {
        // `formattedRequests` 等字段在 `handleRequest` 内由运行时填充；此处仅构造原始对话与超参。
        // 解析输入文件并生成 LoRA 映射与 batched 请求。
        std::tie(loraWeightsMap, batchedRequests)
            = parseInputFile(args.inputFile, args.batchSize, args.maxGenerateLength);
        // 打印成功解析的 LoRA 权重数。
        LOG_INFO("Successfully parsed %zu LoRA weights from input file.", loraWeightsMap.size());
        // 打印成功解析的 batch 请求数。
        LOG_INFO("Successfully parsed %zu batches of requests from input file.", batchedRequests.size());
    }
    // 捕获解析阶段的标准异常。
    catch (std::exception const& e)
    {
        // 记录输入解析失败原因。
        LOG_ERROR("Failed to parse input file: %s", e.what());
        // 输入解析失败直接退出。
        return EXIT_FAILURE;
    }

    // 校验是否至少有一个有效请求。
    if (batchedRequests.empty())
    {
        // 没有请求则报错。
        LOG_ERROR("No valid requests found in input file.");
        // 无工作可执行，按失败退出。
        return EXIT_FAILURE;
    }

    // 运行时构造时会从 engineDir 读取序列化 plan 并建立推理状态；multimodalEngineDir 非空则额外加载视觉 engine。
    // 所有 GPU 提交默认经 stream，便于与可选 CUDA Graph 捕获范围对齐。
    // 经典 LLM 运行时实例（非 Eagle）。
    std::unique_ptr<rt::LLMInferenceRuntime> llmInferenceRuntime{nullptr};
    // Eagle SpecDecode 运行时实例。
    std::unique_ptr<rt::LLMInferenceSpecDecodeRuntime> eagleInferenceRuntime{nullptr};
    // 声明 CUDA stream 句柄。
    cudaStream_t stream;
    // 非阻塞流：后续 kernel 与 TensorRT enqueue 默认异步；同步由 runtime 在读写 logits/结果前触发。
    // 创建 CUDA stream 用于整个推理生命周期。
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 根据参数选择 Eagle 还是标准解码路径。
    if (args.eagleArgs.enabled)
    {
        // Eagle 路径当前不支持随请求切换 LoRA 权重映射。
        // 若输入包含 LoRA，给出忽略警告。
        if (!loraWeightsMap.empty())
        {
            // 记录 Eagle 下不支持 LoRA 的提示。
            LOG_WARNING("Eagle mode does not support LoRA weights. Ignoring LoRA weights.");
        }

        // 组装 Eagle 草稿/验证阶段配置。
        rt::EagleDraftingConfig draftingConfig{
            args.eagleArgs.draftTopK, args.eagleArgs.draftStep, args.eagleArgs.verifyTreeSize};
        // 通过异常保护 Eagle runtime 构造。
        try
        {
            // 构造期加载 base/draft 相关 engine 文件与资源；可能触发设备内存分配与 TensorRT runtime 初始化。
            // 创建 Eagle 推理运行时对象。
            eagleInferenceRuntime = std::make_unique<rt::LLMInferenceSpecDecodeRuntime>(
                args.engineDir, args.multimodalEngineDir, draftingConfig, stream);
        }
        // 捕获 Eagle runtime 初始化异常。
        catch (std::exception const& e)
        {
            // 打印初始化失败信息。
            LOG_ERROR("Failed to initialize LLMInferenceSpecDecodeRuntime: %s", e.what());
            // 初始化失败直接退出。
            return EXIT_FAILURE;
        }

        // Graph 捕获阶段会执行一次「模板」推理序列以记录依赖；失败则仍走动态 launch（见 runtime 实现）。
        // 捕获 Eagle draft proposal 阶段 CUDA Graph。
        bool const draftProposalCaptureStatus = eagleInferenceRuntime->captureDraftProposalCudaGraph(stream);
        // 若捕获失败，降级普通执行并告警。
        if (!draftProposalCaptureStatus)
        {
            // 记录 proposal 图捕获失败告警。
            LOG_WARNING(
                "Failed to capture CUDA graph for draft proposal usage, proceeding with normal engine execution.");
        }

        // 捕获 Eagle draft accept decode token 阶段 CUDA Graph。
        bool const draftAcceptCaptureStatus = eagleInferenceRuntime->captureDraftAcceptDecodeTokenCudaGraph(stream);
        // 若捕获失败，降级普通执行并告警。
        if (!draftAcceptCaptureStatus)
        {
            // 记录 accept 图捕获失败告警。
            LOG_WARNING(
                "Failed to capture CUDA graph for draft accept decode token usage, proceeding with normal engine "
                "execution.");
        }

        // 捕获 Eagle base verification 阶段 CUDA Graph。
        bool const baseCaptureStatus = eagleInferenceRuntime->captureBaseVerificationCudaGraph(stream);
        // 若捕获失败，降级普通执行并告警。
        if (!baseCaptureStatus)
        {
            // 记录 base verification 图捕获失败告警。
            LOG_WARNING(
                "Failed to capture CUDA graph for base model verification usage, proceeding with normal engine "
                "execution.");
        }
    }
    // 标准（非 Eagle）自回归路径。
    else
    {
        // 标准自回归解码：构造时传入 loraWeightsMap，运行期可按请求名加载对应权重（与 Eagle 路径互斥）。
        // 通过异常保护标准 runtime 构造。
        try
        {
            // 传入 LoRA 映射供非 Eagle 路径在 `handleRequest` 中按名加载权重；构造期读 engine 目录。
            // 创建标准 LLM 推理运行时对象。
            llmInferenceRuntime = std::make_unique<rt::LLMInferenceRuntime>(
                args.engineDir, args.multimodalEngineDir, loraWeightsMap, stream);
        }
        // 捕获标准 runtime 初始化异常。
        catch (std::exception const& e)
        {
            // 打印初始化失败信息。
            LOG_ERROR("Failed to initialize LLMInferenceRuntime: %s", e.what());
            // 初始化失败直接退出。
            return EXIT_FAILURE;
        }
        // 解码阶段同样可捕获 CUDA Graph；首次捕获会执行一遍“模板”推理以记录依赖。
        // 尝试捕获标准 decode CUDA Graph。
        if (!llmInferenceRuntime->captureDecodingCUDAGraph(stream))
        {
            // 捕获失败仅告警，继续动态执行。
            LOG_WARNING("Failed to capture CUDA graph for decoding usage, proceeding with normal engine execution.");
        }
    }

    // Warmup：用首个 batch 反复跑通以完成 JIT/缓存/graph 稳定化；期间关闭 profiling 计时，避免污染正式 benchmark。
    // 判断是否需要预热。
    if (args.warmup > 0)
    {
        // 关闭 gTimer 等统计，避免把预热算进正式 profile。
        // 暂时关闭 profile 统计。
        setProfilingEnabled(false);
        // 打印预热开始日志。
        LOG_INFO("Starting warmup with %d runs using the first request...", args.warmup);
        // 取第一个 batch 作为预热请求。
        auto& firstRequest = batchedRequests[0];

        // 按指定次数循环预热。
        for (int32_t warmupRun = 0; warmupRun < args.warmup; ++warmupRun)
        {
            // 每次预热都创建独立响应对象。
            rt::LLMGenerationResponse warmupResponse;
            // 保存本次预热调用状态。
            bool requestStatus = false;
            // 与正式推理相同走 handleRequest，预热 CUDA Graph / 缓存，不计入 profile。
            // Eagle 路径调用 Eagle runtime。
            if (args.eagleArgs.enabled)
            {
                // 执行 Eagle 预热请求。
                requestStatus = eagleInferenceRuntime->handleRequest(firstRequest, warmupResponse, stream);
            }
            // 非 Eagle 路径调用标准 runtime。
            else
            {
                // 执行标准路径预热请求。
                requestStatus = llmInferenceRuntime->handleRequest(firstRequest, warmupResponse, stream);
            }

            // 检查预热是否执行成功。
            if (!requestStatus)
            {
                // 打印预热失败的进度信息。
                LOG_ERROR("Warmup run %d/%d failed", warmupRun + 1, args.warmup);
                // 预热失败按失败退出。
                return EXIT_FAILURE;
            }
        }
        // 打印预热完成日志。
        LOG_INFO("Warmup of %d runs completed. Starting actual benchmark runs...", args.warmup);
    }

    // 若用户要求 profile，开启统计开关。
    if (profilerEnabled)
    {
        // 启用 profiling。
        setProfilingEnabled(true);
    }

    // 准备总输出 JSON 根对象。
    nlohmann::json outputData;
    // 记录输入文件路径，便于追溯。
    outputData["input_file"] = args.inputFile;
    // 创建 responses 数组容器。
    outputData["responses"] = nlohmann::json::array();

    // 标记是否出现过失败请求。
    bool hasFailedRequest = false;
    // 失败时写入默认错误文本。
    std::string errorMessage = "TensorRT Edge LLM cannot handle this request. Fails.";
    // 统计失败 batch 数量。
    size_t failedCount = 0;

    // 各 batch 顺序执行；吞吐优化需在 runtime 或批大小上调整，而非本循环并行化。
    // 打印批处理总量。
    LOG_INFO("Processing %zu batched requests...", batchedRequests.size());
    // 顺序遍历每个 batched request。
    for (size_t requestIdx = 0; requestIdx < batchedRequests.size(); ++requestIdx)
    {
        // 取当前请求引用。
        auto& request = batchedRequests[requestIdx];
        // 为当前请求准备响应对象。
        rt::LLMGenerationResponse response;

        // 控制日志频率：大批量评测时避免每 batch 一条 LOG。
        // 计算日志打印间隔（约 10% 粒度，最大 100）。
        size_t progressInterval = std::max(size_t(1), std::min(batchedRequests.size() / 10, size_t(100)));
        // 在首个、末个或命中间隔时打印进度。
        if ((requestIdx + 1) % progressInterval == 0 || requestIdx == 0 || requestIdx == batchedRequests.size() - 1)
        {
            // 打印当前处理进度百分比。
            LOG_INFO("Progress: %zu/%zu (%f%%)", requestIdx + 1, batchedRequests.size(),
                100.0 * (requestIdx + 1) / batchedRequests.size());
        }

        // 保存本请求执行状态。
        bool requestStatus = false;
        // handleRequest：内部串起 prefill、解码步进、KV cache 更新与 logits 采样；GPU 工作经 stream 提交，同步点在实现内。
        // Eagle 分支调用 Eagle runtime。
        if (args.eagleArgs.enabled)
        {
            // 执行 Eagle 请求。
            requestStatus = eagleInferenceRuntime->handleRequest(request, response, stream);
        }
        // 标准分支调用 LLMInferenceRuntime。
        else
        {
            // 执行标准请求。
            requestStatus = llmInferenceRuntime->handleRequest(request, response, stream);
        }

        // 请求成功分支。
        if (requestStatus)
        {
            // 若开启输出，则打印每个 batch 子请求文本。
            if (args.dumpOutput)
            {
                // 遍历当前响应中的所有输出文本。
                for (size_t batchIdx = 0; batchIdx < response.outputTexts.size(); ++batchIdx)
                {
                    // 打印单条响应文本。
                    LOG_INFO("Response for request %zu batch %zu: %s", requestIdx, batchIdx,
                        response.outputTexts[batchIdx].c_str());
                }
            }
        }
        // 请求失败分支。
        else
        {
            // 记录出现失败。
            hasFailedRequest = true;
            // 失败计数加一。
            failedCount++;
            // 打印失败请求编号。
            LOG_ERROR("*** FAILED *** Request %zu failed to process!", requestIdx);
        }

        // 写出评测用 JSON：output_text 经 sanitizeUtf8ForJson，避免非法 UTF-8 字节序列导致 dump() 抛异常。
        // 遍历当前 request 内每个子样本，构造输出 JSON。
        for (size_t batchIdx = 0; batchIdx < request.requests.size(); ++batchIdx)
        {
            // 创建单条响应 JSON 对象。
            nlohmann::json responseJson;
            // 成功时取模型输出，失败时写固定错误文本。
            std::string outputText = requestStatus ? response.outputTexts[batchIdx] : errorMessage;
            // 写入清洗后的文本，确保可序列化。
            responseJson["output_text"] = sanitizeUtf8ForJson(outputText);
            // 写入外层请求索引。
            responseJson["request_idx"] = requestIdx;
            // 写入 batch 内索引。
            responseJson["batch_idx"] = batchIdx;
            // 创建 messages 数组用于回填输入消息。
            nlohmann::json messagesJson = nlohmann::json::array();
            // 遍历该子请求的对话消息。
            for (auto const& msg : request.requests[batchIdx].messages)
            {
                // 创建单条 message JSON。
                nlohmann::json msgJson;
                // 写入角色字段（system/user/assistant）。
                msgJson["role"] = msg.role;
                // 初始化 content 数组。
                msgJson["content"] = nlohmann::json::array();
                // 遍历一条 message 中的多段内容。
                for (auto const& content : msg.contents)
                {
                    // 创建单段 content JSON。
                    nlohmann::json contentJson;
                    // 写入 content 类型。
                    contentJson["type"] = content.type;
                    // 文本类型写入 text 字段。
                    if (content.type == "text")
                    {
                        // 保存文本内容。
                        contentJson["text"] = content.content;
                    }
                    // 图片类型写入 image 字段。
                    else if (content.type == "image")
                    {
                        // 保存图片路径/标识。
                        contentJson["image"] = content.content;
                    }
                    // 视频类型写入 video 字段。
                    else if (content.type == "video")
                    {
                        // 保存视频路径/标识。
                        contentJson["video"] = content.content;
                    }
                    // 追加当前 content 到 message 内容数组。
                    msgJson["content"].push_back(contentJson);
                }
                // 追加当前 message 到 messages 数组。
                messagesJson.push_back(msgJson);
            }
            // 写入回填后的完整 messages。
            responseJson["messages"] = messagesJson;
            // `formattedRequests` 由本次 `handleRequest` 填充，供复现模板应用后的完整 prompt。
            // 写入格式化后的 system prompt。
            responseJson["formatted_system_prompt"] = request.formattedRequests[batchIdx].formattedSystemPrompt;
            // 写入格式化后的完整请求文本。
            responseJson["formatted_complete_request"] = request.formattedRequests[batchIdx].formattedCompleteRequest;
            // 将当前响应对象追加到总 responses。
            outputData["responses"].push_back(responseJson);
        }
    }

    // 打印整体处理成功统计。
    LOG_INFO("Processing complete: %zu/%zu batched requests successful", batchedRequests.size() - failedCount,
        batchedRequests.size());
    // 若存在失败 batch，额外打印错误统计。
    if (failedCount > 0)
    {
        // 输出失败总数。
        LOG_ERROR("*** %zu BATCHED REQUESTS FAILED ***", failedCount);
    }

    // 收尾 profile 与内存监控。
    if (profilerEnabled)
    {
        // 关闭 profiling 开关，停止继续累计。
        setProfilingEnabled(false);
        // stop() 会 join 监控线程，保证随后读取的峰值显存/CPU 为整段推理区间统计。
        // 停止内存监控并等待线程退出。
        memoryMonitor.stop();
    }

    // 若开启 profile 控制台输出，则拼装并打印性能摘要。
    if (args.dumpProfile)
    {
        // 创建 profile 字符串缓冲。
        std::ostringstream profileOutput;
        // 输出首个空行以提升可读性。
        profileOutput << std::endl;
        // 输出标题行。
        profileOutput << "=== Performance Summary ===" << std::endl;
        // Eagle 模式下输出 Eagle 对应指标。
        if (args.eagleArgs.enabled)
        {
            // 获取 prefill 指标。
            auto prefillMetrics = eagleInferenceRuntime->getPrefillMetrics();
            // 获取 Eagle generation 指标。
            auto eagleGenerationMetrics = eagleInferenceRuntime->getEagleGenerationMetrics();
            // 获取多模态指标。
            auto multimodalMetrics = eagleInferenceRuntime->getMultimodalMetrics();
            // 写入 prefill 摘要。
            outputPrefillProfile(profileOutput, prefillMetrics);
            // 写入 Eagle generation 摘要。
            outputEagleGenerationProfile(profileOutput, eagleGenerationMetrics);
            // 写入多模态摘要。
            outputMultimodalProfile(profileOutput, multimodalMetrics);
            // 写入内存摘要。
            outputMemoryProfile(profileOutput, memoryMonitor);
        }
        // 非 Eagle 模式输出标准 generation 指标。
        else
        {
            // 获取多模态指标。
            auto multimodalMetrics = llmInferenceRuntime->getMultimodalMetrics();
            // 写入 prefill 摘要。
            outputPrefillProfile(profileOutput, llmInferenceRuntime->getPrefillMetrics());
            // 写入标准 generation 摘要。
            outputGenerationProfile(profileOutput, llmInferenceRuntime->getGenerationMetrics());
            // 写入多模态摘要。
            outputMultimodalProfile(profileOutput, multimodalMetrics);
            // 写入内存摘要。
            outputMemoryProfile(profileOutput, memoryMonitor);
        }
        // 输出分隔线。
        profileOutput << "=====================================" << std::endl;
        // 打印汇总结果。
        LOG_INFO("%s", profileOutput.str().c_str());
    }

    // 若指定 profile JSON 输出路径，则写文件。
    if (!args.profileOutputFile.empty())
    {
        // 通过异常保护 profile 文件写出。
        try
        {
            // 创建 profile JSON 容器。
            nlohmann::json profileJson;

            // Eagle 模式填充 Eagle 指标字段。
            if (args.eagleArgs.enabled)
            {
                // 获取 prefill 指标。
                auto prefillMetrics = eagleInferenceRuntime->getPrefillMetrics();
                // 获取 Eagle generation 指标。
                auto eagleGenerationMetrics = eagleInferenceRuntime->getEagleGenerationMetrics();
                // 获取多模态指标。
                auto multimodalMetrics = eagleInferenceRuntime->getMultimodalMetrics();

                // 写入 prefill 汇总到 JSON。
                addJsonPrefillSummary(profileJson, prefillMetrics);
                // 写入 Eagle generation 汇总到 JSON。
                addJsonEagleGenerationSummary(profileJson, eagleGenerationMetrics);
                // 写入多模态汇总到 JSON。
                addJsonMultimodalSummary(profileJson, multimodalMetrics);
                // 写入阶段计时信息到 JSON。
                addJsonTimingStages(profileJson);
                // 写入内存统计到 JSON。
                addJsonMemorySummary(profileJson, memoryMonitor);
            }
            // 非 Eagle 模式填充标准 generation 字段。
            else
            {
                // 获取多模态指标。
                auto multimodalMetrics = llmInferenceRuntime->getMultimodalMetrics();
                // 写入 prefill 汇总到 JSON。
                addJsonPrefillSummary(profileJson, llmInferenceRuntime->getPrefillMetrics());
                // 写入标准 generation 汇总到 JSON。
                addJsonGenerationSummary(profileJson, llmInferenceRuntime->getGenerationMetrics());
                // 写入多模态汇总到 JSON。
                addJsonMultimodalSummary(profileJson, multimodalMetrics);
                // 写入阶段计时信息到 JSON。
                addJsonTimingStages(profileJson);
                // 写入内存统计到 JSON。
                addJsonMemorySummary(profileJson, memoryMonitor);
            }

            // 打开 profile 输出文件。
            std::ofstream profileFile(args.profileOutputFile);
            // 检查文件是否成功打开。
            if (profileFile.is_open())
            {
                // 以两空格缩进写入 profile JSON。
                profileFile << profileJson.dump(2);
                // 主动关闭文件句柄。
                profileFile.close();
                // 输出写入成功日志。
                LOG_INFO("Profile data exported to: %s", args.profileOutputFile.c_str());
            }
            // 打开失败分支。
            else
            {
                // 记录 profile 文件打开失败。
                LOG_ERROR("Failed to open profile output file: %s", args.profileOutputFile.c_str());
                // 无法写 profile，按失败退出。
                return EXIT_FAILURE;
            }
        }
        // 捕获 profile 写出过程异常。
        catch (std::exception const& e)
        {
            // 记录 profile 写出异常原因。
            LOG_ERROR("Failed to write profile output file: %s", e.what());
            // 写出失败按失败退出。
            return EXIT_FAILURE;
        }
    }

    // 写出最终 responses JSON 文件。
    try
    {
        // 打开响应输出文件。
        std::ofstream outputFile(args.outputFile);
        // 检查输出文件是否成功打开。
        if (outputFile.is_open())
        {
            // 以四空格缩进写出响应 JSON。
            outputFile << outputData.dump(4);
            // 主动关闭输出文件。
            outputFile.close();
            // 打印写出成功日志。
            LOG_INFO("All responses exported to: %s", args.outputFile.c_str());
        }
        // 打开失败分支。
        else
        {
            // 记录输出文件打开失败。
            LOG_ERROR("Failed to open output file: %s", args.outputFile.c_str());
            // 无法写输出文件，按失败退出。
            return EXIT_FAILURE;
        }
    }
    // 捕获响应写出过程异常。
    catch (std::exception const& e)
    {
        // 记录响应写出异常原因。
        LOG_ERROR("Failed to write output file: %s", e.what());
        // 写出失败按失败退出。
        return EXIT_FAILURE;
    }

    // 依据是否有失败请求返回最终进程退出码。
    return hasFailedRequest ? EXIT_FAILURE : EXIT_SUCCESS;
}