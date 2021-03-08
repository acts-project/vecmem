/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/details/vector_data.hpp"
#include "vecmem/memory/resources/memory_resource.hpp"

// System include(s).
#include <vector>

namespace vecmem {
   /**
    * @brief Alias type for vectors with our polymorphic allocator
    *
    * This type serves as an alias for a common type pattern, namely a
    * host-accessible vector with a memory resource which is not known at
    * compile time, which could be host memory or shared memory.
    *
    * @warning This type should only be used with host-accessible memory
    * resources.
    */
   template<typename T>
   using vector = std::vector<T, vecmem::polymorphic_allocator<T>>;

   /// Helper function creating a @c vecmem::details::vector_data object
   template< typename TYPE, typename ALLOC >
   VECMEM_HOST
   details::vector_data< TYPE >
   get_data( std::vector< TYPE, ALLOC >& vec );

   /// Helper function creating a @c vecmem::details::vector_data object
   template< typename TYPE, typename ALLOC >
   VECMEM_HOST
   details::vector_data< const TYPE >
   get_data( const std::vector< TYPE, ALLOC >& vec );

} // namespace vecmem

// Include the implementation.
#include "vecmem/containers/vector.ipp"
