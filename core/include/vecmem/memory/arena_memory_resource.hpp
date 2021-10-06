/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/vecmem_core_export.hpp"

// System include(s).
#include <memory>

namespace vecmem {

// Forward declaration(s).
namespace details {
class arena;
}

/// Memory resource implementing an arena allocation scheme
class VECMEM_CORE_EXPORT arena_memory_resource final : public memory_resource {

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

    /// Destructor
    ~arena_memory_resource();

private:
    /// @name Function(s) implemented from @c vecmem::memory_resource
    /// @{

    /// Allocate memory in the arena
    virtual void* do_allocate(std::size_t bytes, std::size_t) override;
    /// De-allocate a previously allocated memory block
    virtual void do_deallocate(void* p, std::size_t bytes,
                               std::size_t) override;
    /// Compares @c *this for equality with @c other
    virtual bool do_is_equal(
        const memory_resource& other) const noexcept override;

    /// @}

    /// Object performing the heavy lifting for the memory resource
    std::unique_ptr<details::arena> m_arena;

};  // class arena_memory_resource

}  // namespace vecmem
