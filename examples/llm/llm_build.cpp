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

#include "builder/builder.h"
#include "common/cudaUtils.h"
#include "common/fileUtils.h"
#include "common/logger.h"

#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>

using namespace trt_edgellm;

// 本文件实现 llm_build 可执行程序：从 ONNX 导出目录读取 config.json 与图结构，
// 调用 edgellmBuilder 将网络编译为 TensorRT plan，并序列化写入 --engineDir 指定的目录（即“engine 文件”产物目录）。
// maxInputLen / maxKVCacheCapacity / maxBatchSize 等参数在构建期固定进 engine，直接决定运行时可用的上下文上限与显存占用规模。

// Enum for command line option IDs (using traditional enum for C library compatibility)
enum LLMBuildOptionId : int
{
    HELP = 701,
    ONNX_DIR = 702,
    ENGINE_DIR = 703,
    MAX_INPUT_LEN = 704,
    MAX_KV_CACHE_CAPACITY = 705,
    DEBUG = 706,
    MAX_BATCH_SIZE = 707,
    MAX_LORA_RANK = 708,
    EAGLE_DRAFT = 709,
    EAGLE_BASE = 710,
    MAX_VERIFY_TREE_SIZE = 711,
    MAX_DRAFT_TREE_SIZE = 712,
    VLM = 713,
    MIN_IMAGE_TOKENS = 714,
    MAX_IMAGE_TOKENS = 715
};

struct LLMBuildArgs
{
    bool help{false};
    std::string onnxDir;
    std::string engineDir;
    int64_t maxInputLen{1024};
    int64_t maxKVCacheCapacity{4096};
    bool debug{false};
    int64_t maxBatchSize{4};
    int64_t maxLoraRank{0}; // 0 表示构建的 engine 不包含动态 LoRA 槽位
    bool eagleDraft{false};
    bool eagleBase{false};
    int64_t maxVerifyTreeSize{60};
    int64_t maxDraftTreeSize{60};
    bool isVlm{false};
    int64_t minImageTokens{4};
    int64_t maxImageTokens{1024};
};

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] --onnxDir <dir> --engineDir <dir> [--maxInputLen <int>] "
                 "[--maxKVCacheCapacity <int>] [--maxBatchSize <int>] [--debug] [--maxLoraRank <int>]"
                 "[--eagleDraft] [--eagleBase] [--maxVerifyTreeSize <int>] "
                 "[--maxDraftTreeSize <int>] [--vlm] [--minImageTokens <int>] [--maxImageTokens <int>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                    Display this help message" << std::endl;
    std::cerr << "  --onnxDir                 Provide the input ONNX directory path. Required. " << std::endl;
    std::cerr << "  --engineDir               Provide the output TensorRT engine directory path. Required. "
              << std::endl;
    std::cerr << "  --maxInputLen             Provide the maximum input length for the model. Default = 128"
              << std::endl;
    std::cerr << "  --maxKVCacheCapacity      Provide the maximum KV cache capacity (sequence length). "
                 "Default = 4096"
              << std::endl;
    std::cerr << "  --maxBatchSize            Provide the maximum batch_size for builder. Default = 4" << std::endl;
    std::cerr << "  --debug                   Use debug mode, which outputs more logs." << std::endl;
    std::cerr << "  --maxLoraRank             Maximum LoRA rank for dynamic LoRA adaptation. Default = 0 (no LoRA)"
              << std::endl;
    std::cerr << "  --eagleDraft              Enable Eagle draft mode" << std::endl;
    std::cerr << "  --eagleBase               Enable Eagle base mode" << std::endl;
    std::cerr << "  --maxVerifyTreeSize       Maximum input_ids tokens passed into Eagle base model for tree "
                 "verification. Default = 60"
              << std::endl;
    std::cerr << "  --maxDraftTreeSize        Maximum input_ids tokens passed into Eagle draft model for draft "
                 "generation. Default = 60"
              << std::endl;
    std::cerr << "  --vlm                     Enable VLM mode" << std::endl;
    std::cerr << "  --minImageTokens          Minimum image tokens for VLM. Default = 4" << std::endl;
    std::cerr << "  --maxImageTokens          Maximum image tokens for VLM. Default = 1024" << std::endl << std::endl;
}

bool parseLLMBuildArgs(LLMBuildArgs& args, int argc, char* argv[])
{
    static struct option buildOptions[] = {{"help", no_argument, 0, LLMBuildOptionId::HELP},
        {"onnxDir", required_argument, 0, LLMBuildOptionId::ONNX_DIR},
        {"engineDir", required_argument, 0, LLMBuildOptionId::ENGINE_DIR},
        {"maxInputLen", required_argument, 0, LLMBuildOptionId::MAX_INPUT_LEN},
        {"maxKVCacheCapacity", required_argument, 0, LLMBuildOptionId::MAX_KV_CACHE_CAPACITY},
        {"debug", no_argument, 0, LLMBuildOptionId::DEBUG},
        {"maxBatchSize", required_argument, 0, LLMBuildOptionId::MAX_BATCH_SIZE},
        {"maxLoraRank", required_argument, 0, LLMBuildOptionId::MAX_LORA_RANK},
        {"eagleDraft", no_argument, 0, LLMBuildOptionId::EAGLE_DRAFT},
        {"eagleBase", no_argument, 0, LLMBuildOptionId::EAGLE_BASE},
        {"maxVerifyTreeSize", required_argument, 0, LLMBuildOptionId::MAX_VERIFY_TREE_SIZE},
        {"maxDraftTreeSize", required_argument, 0, LLMBuildOptionId::MAX_DRAFT_TREE_SIZE},
        {"vlm", no_argument, 0, LLMBuildOptionId::VLM},
        {"minImageTokens", required_argument, 0, LLMBuildOptionId::MIN_IMAGE_TOKENS},
        {"maxImageTokens", required_argument, 0, LLMBuildOptionId::MAX_IMAGE_TOKENS}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", buildOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case LLMBuildOptionId::HELP: args.help = true; return true;
        case LLMBuildOptionId::ONNX_DIR:
            if (optarg)
            {
                args.onnxDir = optarg;
            }
            else
            {
                LOG_ERROR("--onnxDir requires option argument.");
                return false;
            }
            break;
        case LLMBuildOptionId::ENGINE_DIR:
            if (optarg)
            {
                args.engineDir = optarg;
            }
            else
            {
                LOG_ERROR("--engineDir requires option argument.");
                return false;
            }
            break;
        case LLMBuildOptionId::MAX_INPUT_LEN:
            if (optarg)
            {
                args.maxInputLen = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::MAX_KV_CACHE_CAPACITY:
            if (optarg)
            {
                args.maxKVCacheCapacity = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::DEBUG: args.debug = true; break;
        case LLMBuildOptionId::MAX_BATCH_SIZE:
            if (optarg)
            {
                args.maxBatchSize = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::MAX_LORA_RANK:
            if (optarg)
            {
                args.maxLoraRank = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::EAGLE_DRAFT: args.eagleDraft = true; break;
        case LLMBuildOptionId::EAGLE_BASE: args.eagleBase = true; break;
        case LLMBuildOptionId::MAX_VERIFY_TREE_SIZE:
            if (optarg)
            {
                args.maxVerifyTreeSize = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::MAX_DRAFT_TREE_SIZE:
            if (optarg)
            {
                args.maxDraftTreeSize = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::VLM: args.isVlm = true; break;
        case LLMBuildOptionId::MIN_IMAGE_TOKENS:
            if (optarg)
            {
                args.minImageTokens = std::stoi(optarg);
            }
            break;
        case LLMBuildOptionId::MAX_IMAGE_TOKENS:
            if (optarg)
            {
                args.maxImageTokens = std::stoi(optarg);
            }
            break;
        default: LOG_ERROR("Invalid Argument %c is %s.", opt, optarg); return false;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    LLMBuildArgs args;
    if ((argc < 2) || (!parseLLMBuildArgs(args, argc, argv)))
    {
        LOG_ERROR("Unable to parse builder args.");
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    if (args.debug)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    }

    // 导出流水线会在 ONNX 旁写入 config.json；构建器据此恢复模型元数据（如词表、架构与 VLM 开关），缺失则无法对齐网络。
    std::string configPath = args.onnxDir + "/config.json";
    std::ifstream configFile(configPath);
    if (!configFile.good())
    {
        LOG_ERROR("config.json not found in onnx directory: %s", args.onnxDir.c_str());
        return EXIT_FAILURE;
    }
    configFile.close();

    // 将 CLI 超参灌入 LLMBuilderConfig；后续 LLMBuilder 内部会据此设置 TensorRT profile、插件与 KV 相关上限。
    builder::LLMBuilderConfig config;
    config.maxInputLen = args.maxInputLen;
    config.maxKVCacheCapacity = args.maxKVCacheCapacity;
    config.maxBatchSize = args.maxBatchSize;
    config.maxLoraRank = args.maxLoraRank;
    config.eagleDraft = args.eagleDraft;
    config.eagleBase = args.eagleBase;
    config.maxVerifyTreeSize = args.maxVerifyTreeSize;
    config.maxDraftTreeSize = args.maxDraftTreeSize;
    config.isVlm = args.isVlm;
    config.minImageTokens = args.minImageTokens;
    config.maxImageTokens = args.maxImageTokens;

    // ONNX 目录 -> 解析与优化 -> 序列化 engine 文件到 engineDir；失败时返回 false（常见原因：ONNX 与 TRT 版本不兼容、显存不足）。
    builder::LLMBuilder llmBuilder(args.onnxDir, args.engineDir, config);
    if (!llmBuilder.build())
    {
        LOG_ERROR("Failed to build LLM engine.");
        return EXIT_FAILURE;
    }

    LOG_INFO("LLM engine built successfully.");
    return EXIT_SUCCESS;
}
