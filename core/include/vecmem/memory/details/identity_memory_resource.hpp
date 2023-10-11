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

// System include(s).
#include <cstddef>

namespace vecmem::details {

/**
 * @brief This memory resource forwards allocation and deallocation requests to
 * the upstream resource.
 *
 * This allocator is here to act as the unit in the monoid of memory resources.
 * It serves only a niche practical purpose.
 */
class VECMEM_CORE_EXPORT identity_memory_resource
    : public details::memory_resource_base {

public:
    /**
     * @brief Constructs the identity memory resource.
     *
     * @param[in] upstream The upstream memory resource to use.
     */
    identity_memory_resource(memory_resource& upstream);

protected:
    /// @name Function(s) implementing @c vecmem::details::memory_resource_base
    /// @{

    /// Allocate memory with the upstream resource
    virtual void* mr_allocate(std::size_t, std::size_t) override;
    /// De-allocate a previously allocated memory block
    virtual void mr_deallocate(void* p, std::size_t, std::size_t) override;
    /// Compare the equality of @c *this memory resource with another
    virtual bool mr_is_equal(const memory_resource&) const noexcept override;

    /// @}

private:
    /// The upstream memory resource to use
    memory_resource& m_upstream;

};  // class identity_memory_resource

}  // namespace vecmem::details
