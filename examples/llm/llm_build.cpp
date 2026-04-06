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

/**
 * @desc: 与 getopt_long 配合使用的 CLI 选项数值 ID；使用整型枚举以便与 C 库 longopts 表兼容。
 * @params: 无
 * @return: 无
 * @others: 数值需避免与 getopt 返回的字符冲突，故从 701 起编号。
 */
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

/**
 * @desc: 解析后的 llm_build 命令行参数，映射到 LLMBuilderConfig 并影响 engine 文件中的 TensorRT profile 与 KV 上限。
 * @params: 无（聚合类型）
 * @return: 无
 * @others: maxLoraRank 为 0 表示不启用动态 LoRA；VLM 相关 token 范围须与 visual_build / 文本侧构建保持一致。
 */
struct LLMBuildArgs
{
    bool help{false};
    std::string onnxDir;
    std::string engineDir;
    int64_t maxInputLen{1024};
    int64_t maxKVCacheCapacity{4096};
    bool debug{false};
    int64_t maxBatchSize{4};
    int64_t maxLoraRank{0};
    bool eagleDraft{false};
    bool eagleBase{false};
    int64_t maxVerifyTreeSize{60};
    int64_t maxDraftTreeSize{60};
    bool isVlm{false};
    int64_t minImageTokens{4};
    int64_t maxImageTokens{1024};
};

/**
 * @desc: 向 stderr 打印用法与选项说明（英文），供非法参数或 --help 时使用。
 * @params: programName 可执行文件名（通常为 argv[0]）
 * @return: 无
 * @others: 仅输出文本，不参与业务逻辑。
 */
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

/**
 * @desc: 使用 getopt_long 解析命令行，填充 LLMBuildArgs；遇 HELP 则置 help 并返回 true。
 * @params: args 输出参数；argc/argv 标准主函数形参
 * @return: 解析是否成功（非法选项或缺参时 false）
 * @others: 不校验 onnxDir/engineDir 非空，由 main 与后续文件检查承担。
 */
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
    // 逐项消费 argv；optarg 指向当前选项的参数串（若有）。
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

/**
 * @desc: llm_build 入口：解析 CLI、校验 ONNX 目录下 config.json、组装 LLMBuilderConfig 并调用 LLMBuilder 生成 engine 文件。
 * @params: argc 参数个数；argv 参数向量
 * @return: EXIT_SUCCESS 成功或仅打印 help；EXIT_FAILURE 解析失败、缺文件或 build() 失败
 * @others: TensorRT 日志级别由 --debug 控制；真正构图与序列化均在 edgellmBuilder 内完成。
 */
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

    // 导出流水线在 ONNX 目录写入 config.json，供构建器恢复元数据（词表、架构、VLM 开关等）。
    std::string configPath = args.onnxDir + "/config.json";
    std::ifstream configFile(configPath);
    if (!configFile.good())
    {
        LOG_ERROR("config.json not found in onnx directory: %s", args.onnxDir.c_str());
        return EXIT_FAILURE;
    }
    configFile.close();

    // 将 CLI 映射为 LLMBuilderConfig；内部据此设置 TensorRT profile、插件与 KV cache 容量等构建期常量。
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

    // ONNX 目录经解析与 TensorRT 优化后，序列化写入 engineDir；失败常见于版本不兼容或显存不足。
    builder::LLMBuilder llmBuilder(args.onnxDir, args.engineDir, config);
    if (!llmBuilder.build())
    {
        LOG_ERROR("Failed to build LLM engine.");
        return EXIT_FAILURE;
    }

    LOG_INFO("LLM engine built successfully.");
    return EXIT_SUCCESS;
}
