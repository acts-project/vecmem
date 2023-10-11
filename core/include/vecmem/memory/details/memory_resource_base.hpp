/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/vecmem_core_export.hpp"

// System include(s).
#include <cstddef>

namespace vecmem::details {

/// Base class for implementations of the @c vecmem::memory_resource interface
///
/// This helper class is mainly meant to help with mitigating compiler warnings
/// about exporting types that inherit from standard library types. But at the
/// very least it also provides a default/conservative implementation for the
/// @c vecmem::memory_resource::do_is_equal(...) function.
///
class VECMEM_CORE_EXPORT memory_resource_base {

public:
    /// Virtual destructor, to make this into a proper base class
    virtual ~memory_resource_base() = default;

protected:
    /// @name Function(s) mimicking @c vecmem::memory_resource
    ///
    /// These functions are one-by-one mimicking the interface of the
    /// @c vecmem::memory_resource class. The idea is that concrete memory
    /// resource classes would inherit from this class, and not from
    /// @c vecmem::memory_resource directly. Thereby not encoding the exact
    /// standard library version into the vecmem libraries at compile time.
    ///
    /// @{

    /// Allocate memory using the memory resource
    ///
    /// By convention, the function is supposed to throw some kind of exception
    /// if the memory allocation fails.
    ///
    /// @param bytes     The number of bytes to allocate
    /// @param alignment The alignment of the allocated memory
    /// @returns A pointer to the allocated memory
    ///
    virtual void* mr_allocate(std::size_t bytes, std::size_t alignment) = 0;

    /// De-allocate memory using the memory resource
    ///
    /// By convention, the function is supposed to throw some kind of exception
    /// if the memory de-allocation fails.
    ///
    /// @param p         The pointer to the memory to de-allocate
    /// @param bytes     The number of bytes to de-allocate
    /// @param alignment The alignment of the allocated memory
    ///
    virtual void mr_deallocate(void* p, std::size_t bytes,
                               std::size_t alignment) = 0;

    /// Compare the equality of @c *this memory resource with another
    ///
    /// @param other The other memory resource to compare with
    /// @returns @c true if the two memory resources are equal, @c false
    ///          otherwise
    ///
    virtual bool mr_is_equal(const memory_resource& other) const noexcept;

    /// @}

};  // class memory_resource_base

}  // namespace vecmem::details
