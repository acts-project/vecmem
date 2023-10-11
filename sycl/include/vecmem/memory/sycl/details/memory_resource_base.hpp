/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/details/memory_resource_base.hpp"
#include "vecmem/utils/sycl/queue_wrapper.hpp"
#include "vecmem/vecmem_sycl_export.hpp"

/// @brief Namespace for types that should not be used directly by clients
namespace vecmem::sycl::details {

/// SYCL memory resource base class
///
/// This class is used as base by all of the oneAPI/SYCL memory resource
/// classes. It holds functionality that those classes all need.
///
class VECMEM_SYCL_EXPORT memory_resource_base
    : public vecmem::details::memory_resource_base {

public:
    /// Constructor on top of a user-provided queue
    memory_resource_base(const queue_wrapper& queue = {});

protected:
    /// @name Function(s) implemented from @c vecmem::memory_resource
    /// @{

    /// Function performing the memory de-allocation
    void mr_deallocate(void* ptr, std::size_t nbytes,
                       std::size_t alignment) override final;

    /// Function comparing two memory resource instances
    bool mr_is_equal(
        const memory_resource& other) const noexcept override final;

    /// @}

    /// The queue that the allocations are made for/on
    queue_wrapper m_queue;

};  // memory_resource_base

}  // namespace vecmem::sycl::details
