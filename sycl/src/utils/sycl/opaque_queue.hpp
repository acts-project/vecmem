/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// SYCL include(s).
#include <CL/sycl.hpp>

namespace vecmem::sycl::details {

/// Helper class for managing queue objects in memory
class opaque_queue : public cl::sycl::queue {
public:
    using cl::sycl::queue::queue;
};

}  // namespace vecmem::sycl::details
