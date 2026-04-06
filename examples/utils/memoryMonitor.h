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

#pragma once

#include <atomic>
#include <future>
#include <thread>

//! 示例程序用显存/内存观测器：在 start() 时区分集成 GPU（iGPU）与独立 GPU（dGPU），统计策略不同。
//! - dGPU：后台线程周期性 cudaMemGetInfo，用“起始空闲显存 − 当前空闲显存”近似进程峰值 GPU 占用（若其他进程释放显存则可能低估，代码已用下溢保护）。
//! - iGPU：统一内存架构下不启动该轮询线程；峰值统一内存通过 getrusage(RUSAGE_SELF) 的 RSS 在查询时反映（与 dGPU 的 CPU 峰值统计方式一致）。
class MemoryMonitor
{
public:
    MemoryMonitor()
        : mActive(false)
        , mPeakGpuMemory(0)
        , mBaselineGpuFreeMemory(0)
        , mIsIGPU(false)
    {
    }

    void start();
    void stop();

    //! Get peak GPU memory in bytes (returns 0 for iGPU)
    size_t getPeakGpuMemory() const;

    //! Get peak CPU memory (RSS) in bytes
    size_t getPeakCpuMemory() const;

    //! Get peak unified memory in bytes (for iGPU systems)
    size_t getPeakUnifiedMemory() const;

    //! Check if device is integrated GPU
    bool isIntegratedGPU() const
    {
        return mIsIGPU;
    }

private:
    void monitor();

    std::atomic_bool mActive{false};
    std::future<void> mTask;
    std::atomic<size_t> mPeakGpuMemory{0};
    size_t mBaselineGpuFreeMemory{0};
    bool mIsIGPU{false};
};
