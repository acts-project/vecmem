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

      /// Type of the objects in the vector/array
      typedef TYPE value_type;
      /// Constant pointer type to the array
      typedef value_type* pointer;

      /// Constructor from any vector type
      template< typename ALLOC >
      VECMEM_HOST
      device_vector_data( std::vector< TYPE, ALLOC >& vec );
      /// Constructor from "raw data"
      VECMEM_HOST_AND_DEVICE
      device_vector_data( std::size_t size, pointer ptr );

      /// Copy constructor
      VECMEM_HOST_AND_DEVICE
      device_vector_data( const device_vector_data& parent );
      /// Assignment operator
      VECMEM_HOST_AND_DEVICE
      device_vector_data&
      operator=( const device_vector_data& rhs );

      /// Size of the array in memory
      std::size_t m_size;
      /// Pointer to the start of the memory block/array
      pointer m_ptr;

   }; // struct device_vector_data

} // namespace vecmem

// Include the implementation.
#include "vecmem/containers/device_vector_data.ipp"
