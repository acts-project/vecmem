/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/details/memory_resource_base.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/memory/pool_options.hpp"
#include "vecmem/vecmem_core_export.hpp"

// System include(s).
#include <memory>

namespace vecmem {

// Forward declaration(s).
namespace details {
struct pool_memory_resource_impl;
}

/// Memory resource pooling allocations of various sizes
///
/// This is a "downstream" memory resource allowing for pooling and caching
/// allocations from an upstream resource, using memory allocated from it for
/// both the blocks provided to the user and for internal bookkeeping of the
/// cached memory.
///
/// The code is a copy of @c thrust::mr::unsynchronized_pool_resource, giving
/// it a standard @c std::pmr::memory_resource interface.
///
class pool_memory_resource final : public details::memory_resource_base {

public:
    /// Create a pool memory resource with the given options
    ///
    /// @param upstream The upstream memory resource to use for allocations
    /// @param opts The options to use for the pool memory resource
    ///
    VECMEM_CORE_EXPORT
    pool_memory_resource(memory_resource& upstream,
                         const pool_options& opts = {});
    /// Move constructor
    VECMEM_CORE_EXPORT
    pool_memory_resource(pool_memory_resource&& parent);
    /// Disallow copying the memory resource
    pool_memory_resource(const pool_memory_resource&) = delete;

    /// Destructor, freeing all allocations
    VECMEM_CORE_EXPORT
    ~pool_memory_resource();

    /// Move assignment operator
    VECMEM_CORE_EXPORT
    pool_memory_resource& operator=(pool_memory_resource&& rhs);
    /// Disallow copying the memory resource
    pool_memory_resource& operator=(const pool_memory_resource&) = delete;

private:
    /// @name Function(s) implementing @c vecmem::memory_resource
    /// @{

    /// Allocate a blob of memory
    VECMEM_CORE_EXPORT
    virtual void* do_allocate(std::size_t, std::size_t) override final;
    /// De-allocate a previously allocated memory blob
    VECMEM_CORE_EXPORT
    virtual void do_deallocate(void* p, std::size_t,
                               std::size_t) override final;

    /// @}

    /// Object implementing the memory resource's logic
    std::unique_ptr<details::pool_memory_resource_impl> m_impl;

};  // class pool_memory_resource

}  // namespace vecmem
