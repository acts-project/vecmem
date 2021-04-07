/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/vector_view.hpp"

// System include(s).
#include <cstddef>

namespace vecmem::cuda::details {

   /// Type of the memory copy
   enum memory_copy_type {
      host_to_device   = 0, ///< Copy operation between the host and a device
      device_to_host   = 1, ///< Copy operation between a device and the host
      host_to_host     = 2, ///< Copy operation on the host
      device_to_device = 3, ///< Copy operation between two devices
      unknown          = 4  ///< Unknown copy type, determined at runtime
   };

   /// Helper function performing the copy of a 1D array/vector
   void copy_raw( std::size_t size, const void* from, void* to,
                  memory_copy_type type = unknown );

   /// Helper function performing the copy of a jagged array/vector
   template< typename TYPE >
   void copy_views( std::size_t size, const data::vector_view< TYPE >* from,
                    data::vector_view< TYPE >* to,
                    memory_copy_type type = unknown );

} // namespace vecmem::cuda::details

// Include the implementation.
#include "vecmem/utils/cuda/impl/copy_impl.ipp"
