/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/utils/sycl/queue_wrapper.hpp"

// SYCL include(s).
#include <CL/sycl.hpp>

namespace vecmem::sycl::details {

   /// Helper function for getting a @c cl::sycl::queue out of
   /// @c vecmem::sycl::queue_wrapper (non-const)
   cl::sycl::queue& get_queue( vecmem::sycl::queue_wrapper& queue );

   /// Helper function for getting a @c cl::sycl::queue out of
   /// @c vecmem::sycl::queue_wrapper (const)
   const cl::sycl::queue&
   get_queue( const vecmem::sycl::queue_wrapper& queue );

} // namespace vecmem::sycl::details
