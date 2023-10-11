/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/details/memory_resource_base.hpp"
#include "vecmem/vecmem_hip_export.hpp"

namespace vecmem::hip::details {

/// Memory resource for HIP shared host/device memory
class VECMEM_HIP_EXPORT host_memory_resource
    : public vecmem::details::memory_resource_base {

protected:
    /// @name Function(s) implementing @c vecmem::details::memory_resource_base
    /// @{

    /// Function performing the memory allocation
    void* mr_allocate(std::size_t nbytes, std::size_t alignment) override final;

    /// Function performing the memory de-allocation
    void mr_deallocate(void* ptr, std::size_t nbytes,
                       std::size_t alignment) override final;

    /// Function comparing two memory resource instances
    bool mr_is_equal(
        const memory_resource& other) const noexcept override final;

    /// @}

};  // class host_memory_resource

}  // namespace vecmem::hip::details
