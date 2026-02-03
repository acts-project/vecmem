/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "cuda_pooled_event.hpp"

#include <cuda_runtime_api.h>

#include <cassert>

#include "cuda_error_handling.hpp"
#include "vecmem/utils/debug.hpp"

namespace vecmem::cuda {

cuda_pooled_event::cuda_pooled_event(cudaEvent_t event,
                                     const vecmem::cuda::event_pool& pool)
    : m_event(event), m_pool(&pool) {
    assert(m_event != nullptr);
}

/// Move constructor
cuda_pooled_event::cuda_pooled_event(cuda_pooled_event&& parent) noexcept
    : m_event(parent.m_event), m_pool(parent.m_pool) {
    parent.m_event = nullptr;
}

/// Destructor
cuda_pooled_event::~cuda_pooled_event() {
    // Check if the user forgot to wait on this asynchronous event.
    if (m_event != nullptr) {
        // If so, wait implicitly now.
        VECMEM_DEBUG_MSG(1, "Asynchronous CUDA event was not waited on!");
        cuda_pooled_event::wait();
#ifdef VECMEM_FAIL_ON_ASYNC_ERRORS
        // If the user wants to fail on asynchronous errors, do so now.
        std::terminate();
#endif  // VECMEM_FAIL_ON_ASYNC_ERRORS
    }
}

/// Move assignment
cuda_pooled_event& cuda_pooled_event::operator=(
    cuda_pooled_event&& rhs) noexcept {
    if (this != &rhs) {
        m_event = rhs.m_event;
        rhs.m_event = nullptr;
        m_pool = rhs.m_pool;
    }
    return *this;
}

/// Synchronize on the underlying CUDA event
void cuda_pooled_event::wait() {
    if (m_event == nullptr) {
        return;
    }
    VECMEM_CUDA_ERROR_CHECK(cudaEventSynchronize(m_event));
    cuda_pooled_event::ignore();
}

/// Ignore the underlying CUDA event
void cuda_pooled_event::ignore() {
    assert(m_pool != nullptr);
    if (m_event == nullptr) {
        return;
    }
    m_pool->free_event(static_cast<void*>(m_event));
    m_event = nullptr;
}

cudaEvent_t cuda_pooled_event::get_event() {
    return m_event;
}
}  // namespace vecmem::cuda
