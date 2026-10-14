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
#include "vecmem/utils/hip/stream_wrapper.hpp"
#include "vecmem/vecmem_hip_export.hpp"

namespace vecmem::hip {

/**
 * @brief Memory resource that wraps stream-ordered HIP device allocation.
 *
 * This is an allocator-type memory resource that allocates HIP device
 * memory asynchronously, on a specific HIP stream.
 */
class stream_ordered_memory_resource final : public memory_resource {

public:
    /// Constructor, with the stream to use for the (de-)allocations
    VECMEM_HIP_EXPORT
    explicit stream_ordered_memory_resource(const stream_wrapper& stream);
    /// Destructor
    VECMEM_HIP_EXPORT
    ~stream_ordered_memory_resource() noexcept override;

private:
    /// @name Function(s) implementing @c vecmem::memory_resource
    /// @{

    /// Allocate stream-ordered device memory
    VECMEM_HIP_EXPORT
    void* do_allocate(std::size_t, std::size_t) override;
    /// De-allocate a previously allocated stream-ordered memory block
    VECMEM_HIP_EXPORT
    void do_deallocate(void* p, std::size_t, std::size_t) override;
    /// Compares @c *this for equality with @c other
    VECMEM_HIP_EXPORT
    bool do_is_equal(const memory_resource& other) const noexcept override;

    /// @}

    /// HIP stream to use for the (de-)allocations
    stream_wrapper m_stream;

};  // class stream_ordered_memory_resource

}  // namespace vecmem::hip
