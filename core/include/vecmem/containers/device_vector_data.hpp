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

      /// Constructor from another type of @c device_vector_data object
      ///
      /// Only enabled if the wrapped type is different, but only by const-ness.
      /// This complication is necessary to avoid problems from SYCL. Which is
      /// very particular about having default copy constructors for the types
      /// that it sends to kernels.
      ///
      template< typename OTHERTYPE,
                std::enable_if_t<
                   ( ! std::is_same< TYPE, OTHERTYPE >::value ) &&
                   std::is_same< TYPE,
                                 typename std::add_const< OTHERTYPE >::type >::value,
                   bool > = true >
      VECMEM_HOST_AND_DEVICE
      device_vector_data( const device_vector_data< OTHERTYPE >& parent );

      /// Size of the array in memory
      std::size_t m_size;
      /// Pointer to the start of the memory block/array
      pointer m_ptr;

   }; // struct device_vector_data

} // namespace vecmem

// Include the implementation.
#include "vecmem/containers/device_vector_data.ipp"
