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

// System include(s).
#include <cstddef>

namespace vecmem::details {

/**
 * @brief This memory resource does nothing, but it does nothing for a purpose.
 *
 * This allocator has little practical use, but can be useful for defining some
 * conditional allocation schemes.
 *
 * Reimplementation of @c std::pmr::null_memory_resource but can accept another
 * memory resource in its constructor.
 */
class VECMEM_CORE_EXPORT terminal_memory_resource
    : public memory_resource_base {

public:
    /**
     * @brief Constructs the terminal memory resource, without an upstream
     * resource.
     */
    terminal_memory_resource(void);

    /**
     * @brief Constructs the terminal memory resource, with an upstream
     * resource.
     *
     * @param[in] upstream The upstream memory resource to use.
     */
    terminal_memory_resource(memory_resource& upstream);

protected:
    /// @name Function(s) implementing @c vecmem::details::memory_resource_base
    /// @{

    /// Throw @c std::bad_alloc.
    virtual void* mr_allocate(std::size_t, std::size_t) override;
    /// Do nothing.
    virtual void mr_deallocate(void* p, std::size_t, std::size_t) override;
    /// Check whether the other resource is also a terminal resource.
    virtual bool mr_is_equal(
        const memory_resource& other) const noexcept override;

    /// @}

};  // class terminal_memory_resource

}  // namespace vecmem::details
