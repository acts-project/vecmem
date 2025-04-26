/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// VecMem include(s).
#include "vecmem/utils/copy.hpp"
#include "vecmem/utils/sycl/details/queue_holder.hpp"
#include "vecmem/vecmem_sycl_export.hpp"

// System include(s).
#include <memory>

namespace vecmem::sycl {

/// Specialisation of @c vecmem::copy for SYCL
///
/// Unlike @c vecmem::cuda::copy and @c vecmem::hip::copy, this object does
/// have a state. As USM memory operations in SYCL happen through a
/// @c ::sycl::queue object. So this object needs to point to a valid
/// queue object itself.
///
/// Different to @c vecmem::sycl::copy, this type performs operations
/// asynchronously, requiring users to introduce synchronisation points
/// explicitly into their code as needed.
///
class async_copy final : public vecmem::copy, public details::queue_holder {

public:
    /// Constructor on top of a user-provided queue
    VECMEM_SYCL_EXPORT
    explicit async_copy(const queue_wrapper& queue);
    /// Destructor
    VECMEM_SYCL_EXPORT
    ~async_copy();

private:
    /// Perform a memory copy using SYCL
    VECMEM_SYCL_EXPORT
    void do_copy(std::size_t size, const void* from, void* to,
                 type::copy_type cptype) const override;
    /// Fill a memory area using SYCL
    VECMEM_SYCL_EXPORT
    void do_memset(std::size_t size, void* ptr, int value) const override;
    /// Create an event for synchronization
    VECMEM_SYCL_EXPORT
    event_type create_event() const override;

    /// Type holding on to internal data for @c vecmem::sycl::async_copy
    struct impl;
    /// Internal data for the object
    std::unique_ptr<impl> m_impl;

};  // class async_copy

}  // namespace vecmem::sycl
