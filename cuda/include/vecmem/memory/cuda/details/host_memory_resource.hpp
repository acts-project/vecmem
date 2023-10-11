/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/details/memory_resource_base.hpp"
#include "vecmem/vecmem_cuda_export.hpp"

namespace vecmem::cuda::details {

/**
 * @brief Memory resource that wraps page-locked CUDA host allocation.
 *
 * This is an allocator-type memory resource that allocates CUDA host
 * memory, which is page-locked by default to allow faster transfer to the
 * CUDA devices.
 */
class VECMEM_CUDA_EXPORT host_memory_resource
    : public vecmem::details::memory_resource_base {

protected:
    /// @name Function(s) implementing @c vecmem::details::memory_resource_base
    /// @{

    /// Allocate page-locked host memory
    virtual void* mr_allocate(std::size_t, std::size_t) override;
    /// De-allocate a previously allocated page-locked memory block
    virtual void mr_deallocate(void* p, std::size_t, std::size_t) override;
    /// Compares @c *this for equality with @c other
    virtual bool mr_is_equal(
        const memory_resource& other) const noexcept override;

    /// @}

};  // class host_memory_resource

}  // namespace vecmem::cuda::details
