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
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/vecmem_core_export.hpp"

namespace vecmem::details {

// Forward declaration(s).
class arena_memory_resource_impl;

/// Memory resource implementing an arena allocation scheme
class VECMEM_CORE_EXPORT arena_memory_resource : public memory_resource_base {

public:
    /// Construct the memory resource on top of an upstream memory resource
    ///
    /// @param[in] upstream The @c vecmem::memory_resource to use for "upstream"
    ///                     memory allocations
    /// @param[in] initial_size Initial memory memory allocation from
    ///                         @c upstream
    /// @param[in] maximum_size The maximal allowed allocation from @c upstream
    ///
    arena_memory_resource(memory_resource& upstream, std::size_t initial_size,
                          std::size_t maximum_size);
    /// Move constructor
    arena_memory_resource(arena_memory_resource&& parent);
    /// Disallow copying the memory resource
    arena_memory_resource(const arena_memory_resource&) = delete;

    /// Destructor
    ~arena_memory_resource();

    /// Move assignment operator
    arena_memory_resource& operator=(arena_memory_resource&& rhs);
    /// Disallow copying the memory resource
    arena_memory_resource& operator=(const arena_memory_resource&) = delete;

protected:
    /// @name Function(s) implementing @c vecmem::details::memory_resource_base
    /// @{

    /// Allocate memory in the arena
    virtual void* mr_allocate(std::size_t bytes, std::size_t) override;
    /// De-allocate a previously allocated memory block
    virtual void mr_deallocate(void* p, std::size_t bytes,
                               std::size_t) override;

    /// @}

private:
    /// Object performing the heavy lifting for the memory resource
    arena_memory_resource_impl* m_impl;

};  // class arena_memory_resource

}  // namespace vecmem::details
