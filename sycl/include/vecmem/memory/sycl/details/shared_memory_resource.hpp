/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/sycl/details/memory_resource_base.hpp"
#include "vecmem/vecmem_sycl_export.hpp"

namespace vecmem::sycl::details {

/// Memory resource shared between the host and a specific SYCL device
class VECMEM_SYCL_EXPORT shared_memory_resource
    : public details::memory_resource_base {

public:
    // Inherit the base class's constructor(s).
    using details::memory_resource_base::memory_resource_base;

protected:
    /// @name Function(s) implementing @c vecmem::details::memory_resource_base
    /// @{

    /// Function performing the memory allocation
    void* mr_allocate(std::size_t nbytes, std::size_t alignment) override final;

    /// @}

};  // class shared_memory_resource

}  // namespace vecmem::sycl::details
