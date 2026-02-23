/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 *
 * This file incorporates work covered by the following copyright and
 * permission notice:
 *
 *   Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

// Local include(s).
#include "event_pool.hpp"

#include "hip_error_handling.hpp"

// System include(s).
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace vecmem {
namespace hip {
namespace details {

event_pool::event_pool(std::size_t size) : m_pool(size), m_used_events(0) {

    // Create the (initial) events in the pool.
    for (hipEvent_t& e : m_pool) {
        VECMEM_HIP_ERROR_CHECK(
            hipEventCreateWithFlags(&e, hipEventDisableTiming));
    }
}

event_pool::~event_pool() {

    // Destroy all events in the pool.
    for (hipEvent_t& e : m_pool) {
        VECMEM_HIP_ERROR_CHECK(hipEventDestroy(e));
    }
}

hipEvent_t event_pool::create() {

    // A sanity check.
    assert(m_used_events <= m_pool.size());

    // Create a new event if we don't have any available in the pool.
    if (m_pool.size() <= m_used_events) {
        hipEvent_t e;
        VECMEM_HIP_ERROR_CHECK(
            hipEventCreateWithFlags(&e, hipEventDisableTiming));
        m_pool.push_back(e);
    }

    // Return an (unused) event from the pool.
    return m_pool[m_used_events++];
}

void event_pool::free(hipEvent_t event) {

    // Some sanity checks.
    assert(event != nullptr);
    assert(m_used_events > 0u);

    // Find this event in the pool.
    auto it = std::find(m_pool.begin(), m_pool.end(), event);
    if (it == m_pool.end()) {
        throw std::logic_error(
            "HIP event to be freed was not found in event pool");
    }

    // Put this event back at the end of the pool. So that the element pointed
    // at by m_used_events would always be available for use.
    std::swap(*it, m_pool[--m_used_events]);
}

}  // namespace details
}  // namespace hip
}  // namespace vecmem
