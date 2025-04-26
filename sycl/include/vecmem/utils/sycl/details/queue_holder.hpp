/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/utils/sycl/queue_wrapper.hpp"
#include "vecmem/vecmem_sycl_export.hpp"

// System include(s).
#include <memory>

namespace vecmem::sycl::details {

/// Base class for all user-facing, @c sycl::queue using classes
class queue_holder {

public:
    /// Constructor on top of a user-provided queue
    VECMEM_SYCL_EXPORT
    explicit queue_holder(const queue_wrapper& queue = {});
    /// Destructor
    VECMEM_SYCL_EXPORT
    ~queue_holder();

protected:
    /// Get the held queue
    queue_wrapper& queue() const;

private:
    /// Type holding on to internal data for
    /// @c vecmem::sycl::details::queue_holder
    struct impl;
    /// Internal data for the object
    std::unique_ptr<impl> m_impl;

};  // queue_holder

}  // namespace vecmem::sycl::details