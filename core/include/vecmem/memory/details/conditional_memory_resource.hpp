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
#include <functional>

namespace vecmem::details {

// Forward declaration(s).
class conditional_memory_resource_impl;

/**
 * @brief This memory resource conditionally allocates memory. It is
 * constructed with a predicate function that determines whether an allocation
 * should succeed or not.
 *
 * This resource can be used to construct complex conditional allocation
 * schemes.
 */
class VECMEM_CORE_EXPORT conditional_memory_resource
    : public memory_resource_base {

public:
    /**
     * @brief Constructs the conditional memory resource.
     *
     * @param[in] upstream The upstream memory resource to use.
     * @param[in] pred The predicate function that determines whether the
     * allocation should succeed.
     */
    conditional_memory_resource(
        memory_resource& upstream,
        std::function<bool(std::size_t, std::size_t)> pred);
    /// Move constructor
    conditional_memory_resource(conditional_memory_resource&& parent);
    /// Disallow copying the memory resource
    conditional_memory_resource(const conditional_memory_resource&) = delete;

    /// Destructor
    ~conditional_memory_resource();

    /// Move assignment operator
    conditional_memory_resource& operator=(conditional_memory_resource&& rhs);
    /// Disallow copying the memory resource
    conditional_memory_resource& operator=(const conditional_memory_resource&) =
        delete;

protected:
    /// @name Function(s) implementing @c vecmem::details::memory_resource_base
    /// @{

    /// Allocate memory with one of the underlying resources
    virtual void* mr_allocate(std::size_t, std::size_t) override;
    /// De-allocate a previously allocated memory block
    virtual void mr_deallocate(void* p, std::size_t, std::size_t) override;

    /// @}

private:
    /// The implementation of the conditional memory resource.
    conditional_memory_resource_impl* m_impl;

};  // class conditional_memory_resource

}  // namespace vecmem::details
