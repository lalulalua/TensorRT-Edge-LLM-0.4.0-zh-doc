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

#include "memoryMonitor.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/tensor.h"
#include <chrono>
#include <cuda_runtime.h>
#include <exception>
#include <sys/resource.h>
#include <thread>

using namespace trt_edgellm;

namespace
{
//! 查询当前设备全局空闲/总显存；用于 dGPU 下估算本进程相对基线的增量占用。
std::pair<size_t, size_t> getGpuMemoryInfo()
{
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    return {freeMem, totalMem};
}

//! cudaDevAttrIntegrated：Jetson 等平台上为 1，桌面独显一般为 0。
bool detectIntegratedGPU()
{
    int device{-1};
    CUDA_CHECK(cudaGetDevice(&device));
    int integrated = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device));
    return integrated == 1;
}
} // namespace

void MemoryMonitor::start()
{
    if (mTask.valid())
    {
        mActive = false;
        mTask.get();
    }

    mIsIGPU = detectIntegratedGPU();
    mPeakGpuMemory = 0;

    if (mIsIGPU)
    {
        // 统一内存：不在此启动异步轮询；后续 getPeakUnifiedMemory() 走 RSS，与 outputMemoryProfile 的 iGPU 分支一致。
        LOG_INFO("Memory Monitor Started - iGPU detected, monitoring unified memory through RSS");
    }
    else
    {
        // 记录 monitor 开始瞬间的 GPU 空闲量作为基线，后续用 (baselineFree - currentFree) 更新 mPeakGpuMemory。
        auto [freeMem, totalMem] = getGpuMemoryInfo();
        mBaselineGpuFreeMemory = freeMem;

        // Output memory info in MB
        double gpuFreeMemMB = trt_edgellm::rt::utils::toMB(freeMem);
        double gpuTotalMemMB = trt_edgellm::rt::utils::toMB(totalMem);
        LOG_INFO(
            "Memory Monitor Started - dGPU detected, GPU Free: %.2f MB / %.2f MB, monitoring both GPU and CPU memory",
            gpuFreeMemMB, gpuTotalMemMB);

        mActive = true;
        // 异步轮询避免阻塞主线程；周期 50ms，在 profile 结束 stop() 时 join。
        mTask = std::async(std::launch::async, [this]() { monitor(); });
    }
}

void MemoryMonitor::stop()
{
    if (mTask.valid())
    {
        mActive = false;
        mTask.get();
    }
}

size_t MemoryMonitor::getPeakGpuMemory() const
{
    return mPeakGpuMemory;
}

size_t MemoryMonitor::getPeakCpuMemory() const
{
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        // ru_maxrss is in kilobytes on Linux, convert to bytes
        return static_cast<size_t>(usage.ru_maxrss) * 1024;
    }
    return 0;
}

size_t MemoryMonitor::getPeakUnifiedMemory() const
{
    // For iGPU, unified memory(CPU + GPU) is tracked through the same measurement as CPU memory on dGPU.
    return getPeakCpuMemory();
}

void MemoryMonitor::monitor()
{
    while (mActive.load())
    {
        auto [currentFreeMem, totalMem] = getGpuMemoryInfo();
        // 空闲显存变少即认为本进程（及同设备上其他进程）占用了更多；示例场景下通常以本进程为主。
        // 若其他进程释放显存导致 currentFree > baseline，则本次样本记 0，避免 size_t 下溢。
        size_t gpuMemoryUsed = (currentFreeMem < mBaselineGpuFreeMemory) ? mBaselineGpuFreeMemory - currentFreeMem : 0;
        if (mPeakGpuMemory < gpuMemoryUsed)
        {
            mPeakGpuMemory = gpuMemoryUsed;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}
