/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "event_pool.hpp"

#include "cuda_error_handling.hpp"

// System include(s).
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace vecmem {
namespace cuda {
namespace details {

event_pool::event_pool(std::size_t size) : m_pool(size), m_used_events(0) {

    // Create the (initial) events in the pool.
    for (cudaEvent_t& e : m_pool) {
        VECMEM_CUDA_ERROR_CHECK(cudaEventCreate(&e));
    }
}

event_pool::~event_pool() {

    // Destroy all events in the pool.
    for (cudaEvent_t& e : m_pool) {
        VECMEM_CUDA_ERROR_CHECK(cudaEventDestroy(e));
    }
}

cudaEvent_t event_pool::create() {

    // A sanity check.
    assert(m_used_events <= m_pool.size());

    // Create a new event if we don't have any available in the pool.
    if (m_pool.size() <= m_used_events) {
        cudaEvent_t e;
        VECMEM_CUDA_ERROR_CHECK(cudaEventCreate(&e));
        m_pool.push_back(e);
    }

    // Return an (unused) event from the pool.
    return m_pool[m_used_events++];
}

void event_pool::free(cudaEvent_t event) {

    // Some sanity checks.
    assert(event != nullptr);
    assert(m_used_events > 0u);

    // Find this event in the pool.
    auto it = std::find(m_pool.begin(), m_pool.end(), event);
    if (it == m_pool.end()) {
        throw std::logic_error(
            "CUDA event to be freed was not found in event pool");
    }

    // Put this event back at the end of the pool. So that the element pointed
    // at by m_used_events would always be available for use.
    std::swap(*it, m_pool[--m_used_events]);
}

}  // namespace details
}  // namespace cuda
}  // namespace vecmem
