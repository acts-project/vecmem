/**
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
#pragma warning(disable : 4275)
#pragma warning(disable : 4251)
#endif  // MSVC

namespace vecmem {

// Forward declaration(s).
namespace details {
struct binary_page_memory_resource_impl;
}

/**
 * @brief A memory manager using power-of-two pages that can be split to
 * deal with allocation requests of various sizes.
 *
 * This is a non-terminal memory resource which relies on an upstream
 * allocator to do the actual allocation. The allocator will allocate only
 * large blocks with sizes power of two from the upstream allocator. These
 * blocks can then be split in half and allocated, split in half again. This
 * creates a binary tree of pages which can be either vacant, occupied, or
 * split.
 */
class VECMEM_CORE_EXPORT binary_page_memory_resource
    : public details::memory_resource_base {
public:
    /**
     * @brief Initialize a binary page memory manager depending on an
     * upstream memory resource.
     */
    binary_page_memory_resource(memory_resource &);

    /**
     * @brief Deconstruct a binary page memory manager, freeing all
     * allocated blocks upstream.
     *
     * The destructor is explicitly implemented to not require clients of the
     * class to know how to destruct
     * @c vecmem::details::binary_page_memory_resource_impl.
     */
    ~binary_page_memory_resource();

private:
    /// @name Functions implemented from @c vecmem::memory_resource
    /// @{

    /// Allocate a blob of memory
    virtual void *do_allocate(std::size_t, std::size_t) override;
    /// De-allocate a previously allocated memory blob
    virtual void do_deallocate(void *p, std::size_t, std::size_t) override;

    /// @}

    /// Object implementing the memory resource's logic
    std::unique_ptr<details::binary_page_memory_resource_impl> m_impl;

};  // class binary_page_memory_resource

}  // namespace vecmem

// Re-enable the warning(s).
#ifdef _MSC_VER
#pragma warning(pop)
#endif  // MSVC
