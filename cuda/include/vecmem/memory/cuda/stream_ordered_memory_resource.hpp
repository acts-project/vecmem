/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/utils/cuda/stream_wrapper.hpp"
#include "vecmem/vecmem_cuda_export.hpp"

namespace vecmem::cuda {

/**
 * @brief Memory resource that wraps stream-ordered CUDA device allocation.
 *
 * This is an allocator-type memory resource that allocates CUDA device
 * memory asynchronously, on a specific CUDA stream.
 */
class stream_ordered_memory_resource final : public memory_resource {

public:
    /// Constructor, with the stream to use for the (de-)allocations
    VECMEM_CUDA_EXPORT
    explicit stream_ordered_memory_resource(const stream_wrapper& stream);
    /// Destructor
    VECMEM_CUDA_EXPORT
    ~stream_ordered_memory_resource() override;

private:
    /// @name Function(s) implementing @c vecmem::memory_resource
    /// @{

    /// Allocate stream-ordered device memory
    VECMEM_CUDA_EXPORT
    void* do_allocate(std::size_t, std::size_t) override;
    /// De-allocate a previously allocated stream-ordered memory block
    VECMEM_CUDA_EXPORT
    void do_deallocate(void* p, std::size_t, std::size_t) override;
    /// Compares @c *this for equality with @c other
    VECMEM_CUDA_EXPORT
    bool do_is_equal(const memory_resource& other) const noexcept override;

    /// @}

    /// CUDA stream to use for the (de-)allocations
    stream_wrapper m_stream;

};  // class stream_ordered_memory_resource

}  // namespace vecmem::cuda
