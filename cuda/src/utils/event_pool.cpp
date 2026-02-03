/*
 * (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Modifications (c) 2026 CERN for the benefit of the ACTS project
 *
 * This file is a modified version of `cuda_event_pool.cu` provided
 * by NVIDIA CORPORATION & AFFILIATES. Modifications were made to
 * function and variable names. Use of mutual exclusion locks for
 * thread safety was also removed.
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
 * VecMem project, part of the ACTS project (R&D line)
 */

#include "vecmem/utils/cuda/event_pool.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <vector>

#include "cuda_error_handling.hpp"
#include "cuda_pooled_event.hpp"
#include "vecmem/utils/abstract_event.hpp"
#include "vecmem/utils/debug.hpp"

namespace vecmem::cuda {
struct event_pool::impl {
    explicit impl(std::size_t _n) : m_pool(_n) {
        for (cudaEvent_t& e : m_pool) {
            VECMEM_CUDA_ERROR_CHECK(cudaEventCreate(&e));
        }
    }

    ~impl() {
        for (cudaEvent_t e : m_pool) {
            VECMEM_CUDA_ERROR_CHECK(cudaEventDestroy(e));
        }
    }

    std::vector<cudaEvent_t> m_pool;
    std::size_t m_used_events{0};
};

event_pool::event_pool(std::size_t _n) : m_impl{std::make_unique<impl>(_n)} {}

event_pool::event_pool() : event_pool::event_pool(1) {}

event_pool::~event_pool() = default;

event_pool::event_type event_pool::create_event() const {
    assert(m_impl->m_used_events <= m_impl->m_pool.size());

    if (m_impl->m_pool.size() <= m_impl->m_used_events) {
        cudaEvent_t e;
        VECMEM_CUDA_ERROR_CHECK(cudaEventCreate(&e));
        m_impl->m_pool.push_back(e);
    }

    cudaEvent_t e = m_impl->m_pool[m_impl->m_used_events];
    ++(m_impl->m_used_events);
    return std::make_unique<cuda_pooled_event>(e, *this);
}

void event_pool::free_event(void* ep) const {
    auto e = static_cast<cudaEvent_t>(ep);

    if (e == nullptr) {
        return;
    }

    assert(m_impl->m_used_events > 0);
    if (m_impl->m_used_events <= 0) {
        return;
    }

    auto it = std::find(m_impl->m_pool.begin(), m_impl->m_pool.end(), e);

    if (it == m_impl->m_pool.end()) {
        throw std::logic_error("Event to be freed was not found in event pool");
    }

    --m_impl->m_used_events;

    std::swap(*it,
              m_impl->m_pool[m_impl->m_used_events]);  // swap it out of the
                                                       // used events set
}
}  // namespace vecmem::cuda
