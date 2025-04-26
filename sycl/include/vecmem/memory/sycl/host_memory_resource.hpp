/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/sycl/details/memory_resource_base.hpp"
#include "vecmem/vecmem_sycl_export.hpp"

namespace vecmem::sycl {

/// Host memory resource, connected to a specific SYCL device
class host_memory_resource final : public details::memory_resource_base {

public:
    /// Inherit the base class's constructor(s)
    using details::memory_resource_base::memory_resource_base;

private:
    /// @name Function(s) implementing @c vecmem::memory_resource
    /// @{

    /// Function performing the memory allocation
    VECMEM_SYCL_EXPORT
    void* do_allocate(std::size_t nbytes, std::size_t alignment) override;

    /// @}

};  // class host_memory_resource

}  // namespace vecmem::sycl
