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

namespace vecmem { namespace details {

   /// Simple struct holding data about a 1 dimensional vector/array
   ///
   /// This type is meant to "formalise" the communication of data between
   /// @c vecmem::vector, @c vecmem::array ("host types") and
   /// @c vecmem::(const_)device_vector, @c vecmem::(const_)device_array
   /// ("device types").
   ///
   /// This type does not own the data that it points to. It merely provides a
   /// "view" of that data.
   ///
   template< typename TYPE >
   struct vector_view {

      /// Size type used in the class
      typedef std::size_t size_type;
      /// Pointer type to the array
      typedef TYPE* pointer;

      /// Default constructor
      vector_view() = default;
      /// Constructor from "raw data"
      VECMEM_HOST_AND_DEVICE
      vector_view( size_type size, pointer ptr );

      /// Constructor from a "slightly different" @c vecmem::details::vector_view object
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
      vector_view( const vector_view< OTHERTYPE >& parent );

      /// Size of the array in memory
      size_type m_size;
      /// Pointer to the start of the memory block/array
      pointer m_ptr;

   }; // struct vector_view

} } // namespace vecmem::details

// Include the implementation.
#include "vecmem/containers/details/vector_view.ipp"
