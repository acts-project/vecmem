/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

#include <cuda_runtime_api.h>

#include "vecmem/utils/cuda/event_pool.hpp"

namespace vecmem::cuda {

/// CUDA specific implementation of the abstract event interface
struct cuda_pooled_event : public vecmem::abstract_event {

    /// Constructor with the created event.
    explicit cuda_pooled_event(cudaEvent_t event,
                               const vecmem::cuda::event_pool& pool);

    /// Copy constructor
    cuda_pooled_event(const cuda_pooled_event&) = delete;

    /// Move constructor
    cuda_pooled_event(cuda_pooled_event&& parent) noexcept;

    /// Destructor
    ~cuda_pooled_event() override;

    /// Copy assignment
    cuda_pooled_event& operator=(const cuda_pooled_event&) = delete;

    /// Move assignment
    cuda_pooled_event& operator=(cuda_pooled_event&& rhs) noexcept;

    /// Synchronize on the underlying CUDA event
    void wait() override;

    /// Ignore the underlying CUDA event
    void ignore() override;

    cudaEvent_t get_event();

    /// The CUDA event wrapped by this struct
    cudaEvent_t m_event;
    const vecmem::cuda::event_pool* m_pool;
};

}  // namespace vecmem::cuda
