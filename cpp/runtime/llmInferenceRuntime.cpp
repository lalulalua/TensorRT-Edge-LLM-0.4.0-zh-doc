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

// 中文运行时注释（Tier B）：本文件实现标准（非 Eagle）路径下单次请求的 prefill → 自回归 decode → 采样闭环；
// TensorRT enqueueV3、KV 线性缓冲与设备侧序列长度提交详见 LLMEngineRunner / LinearKVCache。

#include "llmInferenceRuntime.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/hashUtils.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "multimodal/multimodalRunner.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "sampler/sampling.h"
#include <fstream>
#include <functional>
#include <string>

using namespace nvinfer1;

namespace trt_edgellm
{

namespace
{

// Left a utility function here in case we want to move to a better hashing method.
size_t hashSystemPromptWithLoraWeights(std::string const& systemPrompt, std::string const& loraWeightsName)
{
    size_t hashValue = 0;
    hash_utils::hashCombine(hashValue, systemPrompt);
    hash_utils::hashCombine(hashValue, loraWeightsName);
    return hashValue;
}

} // namespace
namespace rt
{
LLMInferenceRuntime::LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
    std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream)
{
    std::filesystem::path const enginePath = std::filesystem::path(engineDir) / "llm.engine";
    std::filesystem::path const configPath = std::filesystem::path(engineDir) / "config.json";

    try
    {
        mLLMEngineRunner = std::make_unique<LLMEngineRunner>(enginePath, configPath, loraWeightsMap, stream);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to initialize LLMEngineRunner: %s", e.what());
        throw std::runtime_error("Failed to initialize LLMEngineRunner: " + std::string(e.what()));
    }
    LOG_INFO("LLMEngineRunner successfully loaded and initialized llm engine.");

    mEngineConfig = mLLMEngineRunner->getEngineConfig();

    // 按「最费工作区」的采样配置预留 GPU 字节缓冲；实际请求可用更小 top_p/top_k，但不会超过此上界。
    // Use TopP sampling parameter to reserve max possible workspace size for sampling.
    int32_t const defaultTopK{0};
    float const defaultTopP{0.9F};
    trt_edgellm::SamplingParams samplingParams(
        mEngineConfig.maxSupportedBatchSize, mEngineConfig.outputVocabSize, 1.0f, defaultTopK, defaultTopP);
    int64_t maxSamplingWorkspaceSize = static_cast<int64_t>(trt_edgellm::getTopKtopPSamplingWorkspaceSize(
        mEngineConfig.maxSupportedBatchSize, mEngineConfig.outputVocabSize, samplingParams));

    try
    {
        // mSamplingWorkspace：topK/topP 等采样 kernel 的临时显存（按字节计，类型 kINT8 仅作占位）。
        mSamplingWorkspace = rt::Tensor({maxSamplingWorkspaceSize}, rt::DeviceType::kGPU, DataType::kINT8,
            "LLMInferenceRuntime::mSamplingWorkspace");
        // mInputIds：Prefill 阶段 [B, T_pad]，Decode 阶段 reshape 为 [B, 1]；T 上限为构建 engine 时的 maxSupportedInputLength。
        mInputIds = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxSupportedInputLength},
            rt::DeviceType::kGPU, DataType::kINT32, "LLMInferenceRuntime::mInputIds");
        mHostPackedInputIds = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxSupportedInputLength},
            rt::DeviceType::kCPU, DataType::kINT32, "LLMInferenceRuntime::mHostPackedInputIds");
        // mOutputLogits：每步最后一 token 的 logits，典型 shape [B, vocabSize]（或 reduced vocab 后再映射）。
        mOutputLogits = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.vocabSize}, rt::DeviceType::kGPU,
            DataType::kFLOAT, "LLMInferenceRuntime::mOutputLogits");
        // mSelectedIndices：GPU 上采样得到的当前步 token 下标，shape [B, 1]，同时作为 decode 步的 inputIds。
        mSelectedIndices = rt::Tensor({mEngineConfig.maxSupportedBatchSize, 1}, rt::DeviceType::kGPU, DataType::kINT32,
            "LLMInferenceRuntime::mSelectedIndices");
        mHostSelectedTokenIds = rt::Tensor({mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU,
            DataType::kINT32, "LLMInferenceRuntime::mHostSelectedTokenIds");
        mHostContextLengths = rt::Tensor({mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU, DataType::kINT32,
            "LLMInferenceRuntime::mHostContextLengths");
        mHostReuseKVCacheLengths = rt::Tensor({mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU,
            DataType::kINT32, "LLMInferenceRuntime::mHostReuseKVCacheLengths");
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to allocate workspace and activation tensors for LLM Inference Runtime: %s", e.what());
        throw std::runtime_error(
            "Failed to allocate workspace and activation tensors for LLM Inference Runtime: " + std::string(e.what()));
    }

    // Setup tokenizer
    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    LOG_INFO("Start loading tokenizer from model directory: %s", engineDir.c_str());
    if (!mTokenizer->loadFromHF(engineDir))
    {
        LOG_ERROR("Failed to load tokenizer from model directory: %s", engineDir.c_str());
        throw std::runtime_error("Failed to load tokenizer from model directory: " + engineDir);
    }

    // Optional: Load vocabulary mapping table if reduced vocabulary is used
    if (mEngineConfig.reducedVocabSize > 0)
    {
        LOG_INFO("Loading vocabulary mapping table for reduced vocab size: %d -> %d", mEngineConfig.reducedVocabSize,
            mEngineConfig.vocabSize);
        std::filesystem::path const vocabMapPath = std::filesystem::path(engineDir) / binding_names::kVocabMapFileName;

        std::vector<rt::Tensor> vocabMapTensors;
        if (!safetensors::loadSafetensors(vocabMapPath, vocabMapTensors, stream))
        {
            LOG_ERROR(
                "Failed to load %s from model directory: %s", binding_names::kVocabMapFileName, engineDir.c_str());
            throw std::runtime_error("Failed to load " + std::string(binding_names::kVocabMapFileName)
                + " from model directory: " + engineDir);
        }

        // Check we have exactly one tensor and use it
        check::check(vocabMapTensors.size() == 1,
            std::string(binding_names::kVocabMapFileName) + " should contain exactly one tensor");
        check::check(vocabMapTensors[0].getShape().getNumDims() == 1, "vocab_map tensor should be 1D");
        check::check(vocabMapTensors[0].getShape()[0] == mEngineConfig.reducedVocabSize,
            "vocab_map tensor length should match reduced vocab size");
        mVocabMappingTable = std::move(vocabMapTensors[0]);
        LOG_INFO("Vocabulary mapping table successfully loaded.");
    }

    // Optional: Setup multimodal engine runner
    if (!multimodalEngineDir.empty())
    {
        try
        {
            mMultimodalRunner = MultimodalRunner::create(
                multimodalEngineDir, mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxKVCacheCapacity, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize MultimodalRunner: %s", e.what());
            throw std::runtime_error("Failed to initialize MultimodalRunner: " + std::string(e.what()));
        }
        LOG_INFO("MultimodalRunner successfully loaded and initialized multimodal engine.");
    }
}

bool LLMInferenceRuntime::examineRequest(LLMGenerationRequest const& request)
{
    int32_t const activeBatchSize = static_cast<int32_t>(request.requests.size());

    if (activeBatchSize == 0)
    {
        LOG_ERROR("LLMInferenceRuntime(): The request is empty with no requests supplied.");
        return false;
    }

    if (activeBatchSize > mEngineConfig.maxSupportedBatchSize)
    {
        LOG_ERROR("LLMInferenceRuntime(): The batched request size (%d) exceeds the max supported batch size (%d).",
            activeBatchSize, mEngineConfig.maxSupportedBatchSize);
        return false;
    }

    for (auto const& request : request.requests)
    {
        if (request.messages.empty())
        {
            LOG_ERROR(
                "There is an empty request in the batch. 'messages' must be provided. "
                "Skip this batch of requests. Please check the input data contents.");
            return false;
        }
    }

    return true;
}

/**
 * @desc: Prefill 前准备：处理 system prompt KV 前缀复用、将变长 token 序列 pack/pad 到本 batch 统一长度，重置 LinearKVCache 设备状态并把 inputIds/contextLengths 拷到 GPU，可选切换 LoRA。
 * @params: batchedInputIds — 每条样本一行 token id（已含模板后完整 prompt）；systemPrompts — 与 batch 对齐的 system 文本用于查哈希缓存；loraWeightsName — 当前 batch 逻辑名；stream — 异步 memcpy/kernel 所在 CUDA stream
 * @return: false 表示超长输入、LoRA 切换失败等；成功时 mInputIds/mHostContextLengths 已与 runner 约定一致
 * @others: 非环形 KV：显存块固定为 maxSequenceLength（见 LinearKVCache），「写到哪里」由引擎插件结合 KVCacheLengths 决定；本函数不调用 enqueueV3。
 */
bool LLMInferenceRuntime::setUpForPrefillExecution(std::vector<std::vector<int32_t>> const& batchedInputIds,
    std::vector<std::string> const& systemPrompts, std::string const& loraWeightsName, cudaStream_t stream)
{
    std::vector<std::vector<int32_t>> processedInputIds;
    std::vector<int32_t> processedIdsLengths;
    int32_t const activeBatchSize = static_cast<int32_t>(batchedInputIds.size());

    rt::LinearKVCache& linearKVCache = mLLMEngineRunner->getLinearKVCache();
    rt::Tensor kvCacheBuffer = linearKVCache.getKVCacheBuffer();

    // 记录每条序列已从「缓存的 system KV」复用的 token 长度；后续 resetForNewSequences 据此初始化设备侧 KV 长度张量。
    mHostReuseKVCacheLengths.reshape({activeBatchSize});
    int32_t* reuseKVCacheLengthsData = mHostReuseKVCacheLengths.dataPointer<int32_t>();

    // 若命中 system prompt 缓存：把保存的 KV 块拷回全局 KV 大缓冲的 batch 行 i，并从 inputIds 前部去掉已缓存长度的 token，仅对剩余段做 prefill。
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        auto promptHash = hashSystemPromptWithLoraWeights(systemPrompts[i], loraWeightsName);
        if (mSystemPromptKVCache.find(promptHash) != mSystemPromptKVCache.end())
        {
            auto& precachedKVCache = mSystemPromptKVCache[promptHash];
            auto const& kvCacheContent = precachedKVCache.kvCacheContent;
            kernel::instantiateKVCacheFromTensor(kvCacheBuffer, kvCacheContent, i, stream);
            int32_t reuseLength = static_cast<int32_t>(kvCacheContent.getShape()[3]);
            processedInputIds.emplace_back(batchedInputIds[i].begin() + reuseLength, batchedInputIds[i].end());
            processedIdsLengths.emplace_back(static_cast<int32_t>(batchedInputIds[i].size() - reuseLength));
            reuseKVCacheLengthsData[i] = reuseLength;
            // If the system prompt is not well designed, the boundary of the inputIDs could be mis-aligned.
            check::check(
                reuseLength < batchedInputIds[i].size(), "The reuse length shall not exceed the input length.");
            bool const matchIds = std::equal(precachedKVCache.tokenizedPrompt.begin(),
                precachedKVCache.tokenizedPrompt.end(), batchedInputIds[i].begin());
            if (!matchIds)
            {
                LOG_WARNING(
                    "LLMInferenceRuntime(): Though system prompt strings are matched, token_ids are not perfectly "
                    "aligned. "
                    "This may generate incorrect result, please check your system prompt design.");
            }
        }
        else
        {
            processedInputIds.emplace_back(batchedInputIds[i]);
            processedIdsLengths.emplace_back(static_cast<int32_t>(batchedInputIds[i].size()));
            reuseKVCacheLengthsData[i] = 0;
        }
    }

    // 本 batch 内按最长有效 token 数对齐；每条样本右侧 pad 到同一 T_pad（且不小于引擎 minSupportedInputLength）。
    int32_t const maxInputLength = *std::max_element(processedIdsLengths.begin(), processedIdsLengths.end());
    if (maxInputLength > mEngineConfig.maxSupportedInputLength)
    {
        LOG_ERROR(
            "LLMInferenceRuntime(): The max input length (%d) exceeds the max supported input length (%d) of the LLM "
            "Engine.",
            maxInputLength, mEngineConfig.maxSupportedInputLength);
        return false;
    }

    int32_t const packedInputLength = std::max(maxInputLength, mEngineConfig.minSupportedInputLength);

    // Host 侧先整表填 pad_id，再逐行 copy 真实 token；随后异步 H2D 到 mInputIds，避免在 GPU 上逐元素填 pad。
    mHostPackedInputIds.reshape({activeBatchSize, packedInputLength});
    int32_t* packedInputIdsData = mHostPackedInputIds.dataPointer<int32_t>();
    std::fill(packedInputIdsData, packedInputIdsData + activeBatchSize * packedInputLength, mTokenizer->getPadId());

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        // TODO: 可改为去除 padding 的变长 kernel 以省算力；当前实现为简单右对齐 pad。
        std::copy(processedInputIds[i].begin(), processedInputIds[i].end(), packedInputIdsData + i * packedInputLength);
    }

    // 把复用长度拷到 GPU KV 长度张量并标记 mKVCacheAllEmpty；不清空整块 KV 显存，由插件按长度解释有效区。
    linearKVCache.resetForNewSequences(mHostReuseKVCacheLengths, stream);
    mInputIds.reshape({activeBatchSize, packedInputLength});
    mHostContextLengths.reshape({activeBatchSize});
    mOutputLogits.reshape({activeBatchSize, mEngineConfig.outputVocabSize});

    CUDA_CHECK(cudaMemcpyAsync(mInputIds.rawPointer(), mHostPackedInputIds.rawPointer(),
        activeBatchSize * packedInputLength * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    // contextLengths 为每条序列「本步实际 token 数」（非 pad 后总长时可小于 packedInputLength），供引擎写 KV 有效长度。
    memcpy(mHostContextLengths.dataPointer<int32_t>(), processedIdsLengths.data(), activeBatchSize * sizeof(int32_t));

    if (mEngineConfig.maxSupportedLoraRank > 0 && !mLLMEngineRunner->switchLoraWeights(loraWeightsName, stream))
    {
        LOG_ERROR("Failed to switch LoRA weights to %s", loraWeightsName.c_str());
        return false;
    }

    return true;
}

/**
 * @desc: 端到端处理一个 LLMGenerationRequest：模板 → token 化 →（可选）多模态视觉特征 → setUpForPrefill → executePrefillStep → 每步 GPU 采样 + executeVanillaDecodingStep 直至 EOS 或达到 max_generate_length。
 * @params: request — 含 messages、采样温度/top_p/top_k、maxGenerateLength 等；response — 输出 token 与 decode 文本；stream — 与示例进程共用的 CUDA stream
 * @return: 是否全程成功；失败时 response 可能部分未填
 * @others: Prefill 后 `mInputIds` reshape 为 [B,1] 进入 decode；采样在 `topKtopPSamplingFromLogits`（GPU）后 **cudaStreamSynchronize** 再读 Host 更新 EOS 状态。KV 指针「滑动」由引擎内部按 commit 后的长度字段写入线性缓冲，非本文件维护下标。
 */
bool LLMInferenceRuntime::handleRequest(
    LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream)
{
    std::vector<std::vector<int32_t>> batchedInputIds;
    std::vector<std::string> batchSystemPrompts;
    std::string loraWeightsName = request.loraWeightsName;

    if (!examineRequest(request))
    {
        LOG_ERROR("LLMInferenceRuntime(): Input request examination failed. This request cannot be handled.");
        return false;
    }

    int32_t const activeBatchSize = static_cast<int32_t>(request.requests.size());

    // Apply chat template, extract system prompts, and optionally save KVCache
    request.formattedRequests.resize(activeBatchSize);
    batchSystemPrompts.reserve(activeBatchSize);

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        // Apply chat template
        mTokenizer->applyChatTemplate(request.requests[i], request.formattedRequests[i], request.applyChatTemplate,
            request.addGenerationPrompt, request.enableThinking);

        // Extract system prompt
        batchSystemPrompts.emplace_back(request.formattedRequests[i].formattedSystemPrompt);

        // Save KVCache if requested
        if (request.saveSystemPromptKVCache)
        {
            if (mMultimodalRunner)
            {
                mMultimodalRunner->preprocessSystemPrompt(
                    batchSystemPrompts[i], mTokenizer.get(), mLLMEngineRunner->getRopeCosSinCacheTensor(), stream);
            }
            bool const saveCacheStatus = genAndSaveSystemPromptKVCache(batchSystemPrompts[i], loraWeightsName, stream);
            if (!saveCacheStatus)
            {
                LOG_WARNING(
                    "Failed to save system prompt KVCache. Continue to handle the request without saving the system "
                    "prompt KVCache.");
            }
        }
    }

    // Preprocess user prompts and encode them.
    if (!mMultimodalRunner)
    {
        batchedInputIds.reserve(activeBatchSize);
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            batchedInputIds.emplace_back(
                mTokenizer->encode(request.formattedRequests[i].formattedCompleteRequest, true));
        }
    }
    else
    {
        if (!mMultimodalRunner->preprocess(
                request, batchedInputIds, mTokenizer.get(), mLLMEngineRunner->getRopeCosSinCacheTensor(), stream))
        {
            LOG_ERROR(
                "LLMInferenceRuntime(): Multimodal input request processing failed. This request cannot be handled.");
            return false;
        }

        if (!mMultimodalRunner->infer(stream))
        {
            LOG_ERROR("LLMInferenceRuntime(): Multimodal inference failed. This request cannot be handled.");
            return false;
        }
    }

    if (!setUpForPrefillExecution(batchedInputIds, batchSystemPrompts, loraWeightsName, stream))
    {
        LOG_ERROR("LLMInferenceRuntime(): Prefill execution setup failed. This request cannot be handled.");
        return false;
    }

    // Record context information for performance tracking
    auto tokenCount = calculateTokenCounts(batchedInputIds, batchSystemPrompts, loraWeightsName);

    // maxKVCacheCapacity 来自构建 engine 时的序列容量；input+生成超过则截断生成步数，避免 KV 线性缓冲越界。
    int32_t const maxInputIdsLength = mInputIds.getShape()[1];
    int32_t maxGenerationLength = request.maxGenerateLength;
    if (maxInputIdsLength + maxGenerationLength > mEngineConfig.maxKVCacheCapacity)
    {
        maxGenerationLength = mEngineConfig.maxKVCacheCapacity - maxInputIdsLength;
        LOG_WARNING(
            "The requested input length (%d) + max generation length (%d) = %d exceeds the max KV "
            "cache capacity (%d). Reduce the generation length to %d to avoid the truncation of the generated tokens.",
            maxInputIdsLength, maxGenerationLength, maxInputIdsLength + maxGenerationLength,
            mEngineConfig.maxKVCacheCapacity, maxGenerationLength);
    }

    int32_t unFinishedBatchNum = activeBatchSize;
    int32_t generationIter{0};
    std::vector<std::vector<int32_t>> outputIds(activeBatchSize);
    std::vector<bool> finishedStates(activeBatchSize, false);
    mSelectedIndices.reshape({activeBatchSize, 1});
    mHostSelectedTokenIds.reshape({activeBatchSize});
    int32_t* hostSelectedTokenIdsData = mHostSelectedTokenIds.dataPointer<int32_t>();

    SamplingParams params(
        activeBatchSize, mEngineConfig.outputVocabSize, request.temperature, request.topK, request.topP);
    // 闭包：在 GPU 上对 mOutputLogits 做温度缩放 + topK/topP 截断与采样（实现见 sampler 模块），再 D2H 取 token id 并更新 EOS。
    auto sampleTokens = [&]() {
        trt_edgellm::topKtopPSamplingFromLogits(mOutputLogits, mSelectedIndices, params, mSamplingWorkspace, stream);
        if (mEngineConfig.reducedVocabSize > 0)
        {
            trt_edgellm::mapReducedVocabToFullVocab(mSelectedIndices, mVocabMappingTable, stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mSelectedIndices.rawPointer(),
            activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        // 必须同步：后续在 CPU 上读 hostSelectedTokenIds 并判断 EOS；prefill 后首 token 与每 decode 步各调用一次。
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int32_t i = 0; i < activeBatchSize; ++i)
        {
            if (!finishedStates[i])
            {
                outputIds[i].push_back(hostSelectedTokenIdsData[i]);
                finishedStates[i] = hostSelectedTokenIdsData[i] == mTokenizer->getEosId();
                if (finishedStates[i])
                {
                    unFinishedBatchNum--;
                }
            }
        }
        ++generationIter;
    };

    // Use empty tensor for when no multimodal runner is available.
    // All other data input used by prefill step is already set up in setUpForPrefillExecution().
    rt::OptionalInputTensor multimodalEmbeddings
        = mMultimodalRunner ? std::optional{std::ref(mMultimodalRunner->getOutputEmbedding())} : std::nullopt;
    rt::OptionalInputTensors extraVisualFeatures
        = mMultimodalRunner ? mMultimodalRunner->getExtraVisualFeatures() : rt::OptionalInputTensors{};

    rt::OptionalOutputTensor outputHiddenStates{std::nullopt};
    {
        TIME_STAGE(metrics::StageNames::kLLM_PREFILL, stream);

        // 一次 enqueueV3：处理整段 prompt token，向 KV 缓冲写入 prefill 阶段 K/V；commit 后设备侧序列长度更新。
        bool prefillStatus = mLLMEngineRunner->executePrefillStep(mInputIds, mHostContextLengths, multimodalEmbeddings,
            extraVisualFeatures, mOutputLogits, outputHiddenStates, stream);
        if (!prefillStatus)
        {
            LOG_ERROR(
                "LLMInferenceRuntime(): Failed to execute prefill step. Cannot generate the KVCache for this prompt.");
            return false;
        }
        sampleTokens();
    }

    // Record prefill metrics
    mPrefillMetrics.recordRun(tokenCount.totalReusedTokens, tokenCount.totalComputedTokens);

    // 将 mInputIds 收束为 [B,1] 形态以符合引擎 decode profile；实际每步传入 executeVanillaDecodingStep 的是上一步采样得到的 mSelectedIndices（同为 [B,1]）。
    mInputIds.reshape({activeBatchSize, 1});

    {
        TIME_STAGE(metrics::StageNames::kLLM_GENERATION, stream);

        while (unFinishedBatchNum > 0 && generationIter < maxGenerationLength)
        {
            // enqueueV3（或已捕获的 CUDA Graph）一步：在已有 KV 上追加 1 token，更新 logits；内部 commit 使 KV 长度 +1。
            bool decodingStatus = mLLMEngineRunner->executeVanillaDecodingStep(mSelectedIndices, mOutputLogits, stream);
            if (!decodingStatus)
            {
                LOG_ERROR("LLMInferenceRuntime(): Failed to execute decoding step.");
                return false;
            }

            sampleTokens();
        }
    }

    // Record generation and sampling metrics
    int32_t totalGeneratedTokens = 0;
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        totalGeneratedTokens += static_cast<int32_t>(outputIds[i].size() - 1);
    }

    if (totalGeneratedTokens > 0)
    {
        mGenerationMetrics.recordRun(totalGeneratedTokens);
    }

    // Clean the response field and fill the generated outputIds and decoded texts.
    response.outputIds.clear();
    response.outputTexts.clear();
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        response.outputIds.emplace_back(outputIds[i]);
        response.outputTexts.emplace_back(mTokenizer->decode(outputIds[i], true));
    }

    return true;
}

/**
 * @desc: 为各 batch 规模（及每套 LoRA）预捕获 decode 步 CUDA Graph，使重复执行时走 cudaGraphLaunch 而非逐次 setTensor+enqueueV3。
 * @params: stream — 捕获与 launch 使用的流
 * @return: 是否全部捕获成功（部分失败仍返回 false，但运行时可能回退动态路径）
 * @others: 捕获顺序与形状须与实际 decode 一致；见 LLMEngineRunner::executeVanillaDecodingStep 内 graph 分支。
 */
bool LLMInferenceRuntime::captureDecodingCUDAGraph(cudaStream_t stream)
{
    int32_t const maxSupportedBatchSize = mEngineConfig.maxSupportedBatchSize;
    int32_t const minSupportedBatchSize = 1;

    bool captureStatus{true};
    // Capture the CUDA graph for all available batch sizes.
    for (int32_t batchSize = minSupportedBatchSize; batchSize <= maxSupportedBatchSize; ++batchSize)
    {
        mSelectedIndices.reshape({batchSize, 1});
        mOutputLogits.reshape({batchSize, mEngineConfig.outputVocabSize});
        captureStatus &= mLLMEngineRunner->captureVanillaDecodingCudaGraph(
            mSelectedIndices, mOutputLogits, mEmptyLoraWeightsName, stream);
        if (mEngineConfig.maxSupportedLoraRank > 0)
        {
            for (auto const& loraWeightsName : mLLMEngineRunner->getAvailableLoraWeights())
            {
                captureStatus &= mLLMEngineRunner->captureVanillaDecodingCudaGraph(
                    mSelectedIndices, mOutputLogits, loraWeightsName, stream);
            }
        }
    }

    if (captureStatus)
    {
        LOG_INFO(
            "LLMInferenceRuntime(): Successfully captured the decoding CUDA graph for all execution batch sizes and "
            "LoRA weights.");
    }
    else
    {
        LOG_WARNING(
            "LLMInferenceRuntime(): Failed to capture the decoding CUDA graph for some of execution batch sizes and "
            "LoRA weights.");
    }
    return captureStatus;
}

LLMInferenceRuntime::TokenCountInfo LLMInferenceRuntime::calculateTokenCounts(
    std::vector<std::vector<int32_t>> const& batchedInputIds, std::vector<std::string> const& systemPrompts,
    std::string const& loraWeightsName) const
{
    TokenCountInfo tokenCount;
    int32_t const activeBatchSize = static_cast<int32_t>(batchedInputIds.size());

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        int32_t contextLength = static_cast<int32_t>(batchedInputIds[i].size());
        // Calculate reused length from system prompt cache
        auto promptHash = hashSystemPromptWithLoraWeights(systemPrompts[i], loraWeightsName);
        if (mSystemPromptKVCache.find(promptHash) != mSystemPromptKVCache.end())
        {
            int32_t reusedLength = static_cast<int32_t>(mSystemPromptKVCache.at(promptHash).tokenizedPrompt.size());
            tokenCount.totalReusedTokens += reusedLength;
            tokenCount.totalComputedTokens += (contextLength - reusedLength);
        }
        else
        {
            tokenCount.totalComputedTokens += contextLength;
        }
    }

    return tokenCount;
}

bool LLMInferenceRuntime::genAndSaveSystemPromptKVCache(
    std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream)
{
    if (prompt.empty())
    {
        LOG_DEBUG("LLMInferenceRuntime(): The prompt is empty. Skip saving system prompt KVCache.");
        return true;
    }

    // hash the prompt if check if the prompt cache already exists.
    size_t const promptHash = hashSystemPromptWithLoraWeights(prompt, loraWeightsName);
    if (mSystemPromptKVCache.find(promptHash) != mSystemPromptKVCache.end())
    {
        LOG_DEBUG(
            "LLMInferenceRuntime(): The system prompt KVCache already exists for the prompt: {%s}", prompt.c_str());
        return true;
    }

    auto tokenizedPrompt = mTokenizer->encode(prompt, true);
    int32_t const promptIdsLength = static_cast<int32_t>(tokenizedPrompt.size());
    int32_t const activeBatchSize = 1;

    if (promptIdsLength > mEngineConfig.maxSupportedInputLength)
    {
        LOG_ERROR(
            "LLMInferenceRuntime(): The prompt length (%d) exceeds the max supported input length (%d) of the LLM "
            "Engine.",
            promptIdsLength, mEngineConfig.maxSupportedInputLength);
        return false;
    }

    std::vector<std::vector<int32_t>> batchedInputIds(activeBatchSize, tokenizedPrompt);
    std::vector<std::string> batchedSystemPrompts(activeBatchSize, prompt);
    if (!setUpForPrefillExecution(batchedInputIds, batchedSystemPrompts, loraWeightsName, stream))
    {
        LOG_ERROR(
            "LLMInferenceRuntime(): Prefill execution setup failed. Cannot generate the KVCache for this prompt.");
        return false;
    }

    // Execute prefill step to initialize the KVCache data.
    rt::OptionalInputTensor multimodalEmbeddings
        = mMultimodalRunner ? std::optional{std::ref(mMultimodalRunner->getOutputEmbedding())} : std::nullopt;
    rt::OptionalInputTensors extraVisualFeatures
        = mMultimodalRunner ? mMultimodalRunner->getExtraVisualFeatures() : rt::OptionalInputTensors{};

    rt::OptionalOutputTensor outputHiddenStates{std::nullopt};
    bool prefillStatus = mLLMEngineRunner->executePrefillStep(mInputIds, mHostContextLengths, multimodalEmbeddings,
        extraVisualFeatures, mOutputLogits, outputHiddenStates, stream);
    if (!prefillStatus)
    {
        LOG_ERROR("LLMInferenceRuntime(): Failed to execute prefill step.");
        return false;
    }

    // 从全局线性 KV 大缓冲中切出 batch 0、长度 promptIdsLength 的块拷入独立 Tensor，供下次 instantiateKVCacheFromTensor 恢复。
    auto& linearKVCache = mLLMEngineRunner->getLinearKVCache();
    auto cacheConfig = linearKVCache.getConfig();
    auto kvCacheBuffer = linearKVCache.getKVCacheBuffer();
    rt::Coords savedKVCacheShape{
        cacheConfig.numDecoderLayers, 2, cacheConfig.numKVHeads, promptIdsLength, cacheConfig.headDim};

    SystemPromptKVCache savedKVCache;
    savedKVCache.systemPrompt = prompt;
    savedKVCache.tokenizedPrompt = tokenizedPrompt;
    savedKVCache.kvCacheContent = rt::Tensor(savedKVCacheShape, rt::DeviceType::kGPU, rt::LinearKVCache::KVCacheTypeTRT,
        "LLMInferenceRuntime::savedKVCache.kvCacheContent");

    // We only process one sequence at a time.
    constexpr int32_t CACHE_BATCH_IDX{0};
    kernel::saveKVCacheIntoTensor(savedKVCache.kvCacheContent, kvCacheBuffer, CACHE_BATCH_IDX, stream);
    mSystemPromptKVCache.insert({promptHash, std::move(savedKVCache)});

    CUDA_CHECK(cudaStreamSynchronize(stream));
    LOG_DEBUG("LLMInferenceRuntime(): The KVCache is saved for the prompt: {%s}", prompt.c_str());

    return true;
}

} // namespace rt
} // namespace trt_edgellm