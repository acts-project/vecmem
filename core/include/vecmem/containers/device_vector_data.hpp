/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/utils/types.hpp"

// System include(s).
#include <cstddef>
#include <type_traits>
#include <vector>

namespace vecmem {

   /// Simple struct holding data for @c vecmem::const_device_vector
   ///
   /// This type is meant to "formalise" the communication of data between
   /// @c vecmem::vector (a "host type") and @c vecmem::device_vector
   /// (a "device types").
   ///
   template< typename TYPE >
   struct device_vector_data {

      /// Non-constant value type used in the host vector
      typedef typename std::remove_cv< TYPE >::type value_type;
      /// Pointer type to the array
      typedef TYPE* pointer;

      /// Default constructor
      device_vector_data() = default;
      /// Constructor from "raw data"
      VECMEM_HOST_AND_DEVICE
      device_vector_data( std::size_t size, pointer ptr );

      /// Constructor from a non-const vector
      template< typename ALLOC >
      VECMEM_HOST
      device_vector_data( std::vector< value_type, ALLOC >& vec );
      /// Constructor from a const vector
      template< typename ALLOC >
      VECMEM_HOST
      device_vector_data( const std::vector< value_type, ALLOC >& vec );

      /// Size of the array in memory
      std::size_t m_size;
      /// Pointer to the start of the memory block/array
      pointer m_ptr;

   }; // struct device_vector_data

} // namespace vecmem

// Include the implementation.
#include "vecmem/containers/device_vector_data.ipp"
