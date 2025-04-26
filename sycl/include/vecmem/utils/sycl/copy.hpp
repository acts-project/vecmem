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

namespace vecmem::sycl {

/// Specialisation of @c vecmem::copy for SYCL
///
/// Unlike @c vecmem::cuda::copy and @c vecmem::hip::copy, this object does
/// have a state. As USM memory operations in SYCL happen through a
/// @c ::sycl::queue object. So this object needs to point to a valid
/// queue object itself.
///
class copy final : public vecmem::copy, public details::queue_holder {

public:
    /// Inherit the constructor(s) from @c vecmem::sycl::details::queue_holder
    using details::queue_holder::queue_holder;

private:
    /// Perform a memory copy using SYCL
    VECMEM_SYCL_EXPORT
    void do_copy(std::size_t size, const void* from, void* to,
                 type::copy_type cptype) const override;
    /// Fill a memory area using SYCL
    VECMEM_SYCL_EXPORT
    void do_memset(std::size_t size, void* ptr, int value) const override;

};  // class copy

}  // namespace vecmem::sycl
