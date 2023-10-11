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
#include <functional>

namespace vecmem::details {

// Forward declaration(s).
class choice_memory_resource_impl;

/**
 * @brief This memory resource conditionally allocates memory. It is
 * constructed with a function that determines which upstream resource to use.
 *
 * This resource can be used to construct complex conditional allocation
 * schemes.
 */
class VECMEM_CORE_EXPORT choice_memory_resource : public memory_resource_base {

public:
    /**
     * @brief Construct the choice memory resource.
     *
     * @param[in] upstreams The upstream memory resources to use.
     * @param[in] decision The function which picks the upstream memory
     * resource to use by index.
     */
    choice_memory_resource(
        std::function<memory_resource&(std::size_t, std::size_t)> decision);
    /// Move constructor
    choice_memory_resource(choice_memory_resource&& parent);
    /// Disallow copying the memory resource
    choice_memory_resource(const choice_memory_resource&) = delete;

    /// Destructor
    ~choice_memory_resource();

    /// Move assignment operator
    choice_memory_resource& operator=(choice_memory_resource&& rhs);
    /// Disallow copying the memory resource
    choice_memory_resource& operator=(const choice_memory_resource&) = delete;

protected:
    /// @name Function(s) implementing @c vecmem::details::memory_resource_base
    /// @{

    /// Allocate memory with one of the underlying resources
    virtual void* mr_allocate(std::size_t, std::size_t) override;
    /// De-allocate a previously allocated memory block
    virtual void mr_deallocate(void* p, std::size_t, std::size_t) override;

    /// @}

private:
    /// The implementation of the choice memory resource.
    choice_memory_resource_impl* m_impl;

};  // class choice_memory_resource

}  // namespace vecmem::details
