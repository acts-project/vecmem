/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/utils/sycl/details/queue_holder.hpp"
#include "vecmem/vecmem_sycl_export.hpp"

/// @brief Namespace for types that should not be used directly by clients
namespace vecmem::sycl::details {

/// SYCL memory resource base class
///
/// This class is used as base by all of the oneAPI/SYCL memory resource
/// classes. It holds functionality that those classes all need.
///
class memory_resource_base : public memory_resource, public queue_holder {

public:
    /// Inherit the constructor(s) from @c vecmem::sycl::details::queue_holder
    using queue_holder::queue_holder;

private:
    /// @name Function(s) implemented from @c vecmem::memory_resource
    /// @{

    /// Function performing the memory de-allocation
    VECMEM_SYCL_EXPORT
    void do_deallocate(void* ptr, std::size_t nbytes,
                       std::size_t alignment) final;

    /// Function comparing two memory resource instances
    VECMEM_SYCL_EXPORT
    bool do_is_equal(const memory_resource& other) const noexcept final;

    /// @}

};  // memory_resource_base

}  // namespace vecmem::sycl::details
