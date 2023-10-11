/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/details/memory_resource_base.hpp"
#include "vecmem/vecmem_core_export.hpp"

namespace vecmem::details {

/**
 * @brief Memory resource which wraps standard library memory allocation calls.
 *
 * This is probably the simplest memory resource you can possibly write. It
 * is a terminal resource which does nothing but wrap @c std::aligned_alloc and
 * @c std::free. It is state-free (on the relevant levels of abstraction).
 */
class VECMEM_CORE_EXPORT host_memory_resource : public memory_resource_base {

protected:
    /// @name Function(s) implemented from
    ///       @c vecmem::details::memory_resource_base
    /// @{

    /// Allocate standard host memory
    virtual void* mr_allocate(std::size_t size, std::size_t alignment) override;
    /// De-allocate a block of previously allocated memory
    virtual void mr_deallocate(void* p, std::size_t size,
                               std::size_t alignment) override;
    /// Compares @c *this for equality with @c other
    virtual bool mr_is_equal(
        const memory_resource& other) const noexcept override;

    /// @}

};  // class host_memory_resource

}  // namespace vecmem::details
