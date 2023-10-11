/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/vecmem_core_export.hpp"

namespace vecmem::details {

/// Helper class for providing concrete memory resource implementations
///
/// This class is used throughout the project to provide memory resources
/// implementing the @c vecmem::memory_resource interface, without making the
/// libraries depend on the exact standard library version that was used to
/// compile the project.
///
/// @tparam MR_IMPL The implementation class to use
///
template <typename MR_IMPL>
class memory_resource_adaptor final : public memory_resource, public MR_IMPL {

public:
    /// Inherit the constructors of the implementation class
    using MR_IMPL::MR_IMPL;

    /// Type of the memory resource implementation class
    using memory_resource_impl_type = MR_IMPL;

protected:
    /// @name Implementation of the @c vecmem::memory_resource interface
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
    virtual void* do_allocate(std::size_t bytes,
                              std::size_t alignment) override {
        return MR_IMPL::mr_allocate(bytes, alignment);
    }

    /// De-allocate memory using the memory resource
    ///
    /// By convention, the function is supposed to throw some kind of exception
    /// if the memory de-allocation fails.
    ///
    /// @param p         The pointer to the memory to de-allocate
    /// @param bytes     The number of bytes to de-allocate
    /// @param alignment The alignment of the allocated memory
    ///
    virtual void do_deallocate(void* p, std::size_t bytes,
                               std::size_t alignment) override {
        MR_IMPL::mr_deallocate(p, bytes, alignment);
    }

    /// Compare the equality of @c *this memory resource with another
    ///
    /// @param other The other memory resource to compare with
    /// @returns @c true if the two memory resources are equal, @c false
    ///          otherwise
    ///
    virtual bool do_is_equal(
        const memory_resource& other) const noexcept override {
        return MR_IMPL::mr_is_equal(other);
    }

};  // class memory_resource_adaptor

}  // namespace vecmem::details
