/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/details/memory_resource_base.hpp"
#include "vecmem/vecmem_core_export.hpp"

// System include(s).
#include <memory>

// Disable the warning(s) about inheriting from/using standard library types
// with an exported class.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4251)
#endif  // MSVC

namespace vecmem {

// Forward declaration(s).
namespace details {
class arena;
}

/// Memory resource implementing an arena allocation scheme
class VECMEM_CORE_EXPORT arena_memory_resource final
    : public details::memory_resource_base {

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

    /// @}

    /// Object performing the heavy lifting for the memory resource
    std::unique_ptr<details::arena> m_arena;

};  // class arena_memory_resource

}  // namespace vecmem

// Re-enable the warning(s).
#ifdef _MSC_VER
#pragma warning(pop)
#endif  // MSVC
