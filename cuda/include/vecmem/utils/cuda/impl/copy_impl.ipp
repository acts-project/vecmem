/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <cassert>

namespace vecmem::cuda::details {

   template< typename TYPE >
   void copy_views( std::size_t size, const data::vector_view< TYPE >* from,
                    data::vector_view< TYPE >* to, memory_copy_type type ) {

      // Helper variables used in the copy.
      const TYPE* from_ptr = nullptr;
      TYPE* to_ptr = nullptr;
      std::size_t copy_size = 0, copy_ops = 0;

      // Perform the copy in multiple steps.
      for( std::size_t i = 0; i < size; ++i ) {

         // Skip empty "inner vectors".
         if( ( from[ i ].m_size == 0 ) && ( to[ i ].m_size == 0 ) ) {
            continue;
         }

         // Some sanity checks.
         assert( from[ i ].m_ptr != nullptr );
         assert( to[ i ].m_ptr != nullptr );
         assert( from[ i ].m_size != 0 );
         assert( from[ i ].m_size == to[ i ].m_size );

         // Set/update the helper variables.
         if( ( from_ptr == nullptr ) && ( to_ptr == nullptr ) &&
             ( copy_size == 0 ) ) {
            from_ptr = from[ i ].m_ptr;
            to_ptr = to[ i ].m_ptr;
            copy_size = from[ i ].m_size * sizeof( TYPE );
         } else {
            assert( from_ptr != nullptr );
            assert( to_ptr != nullptr );
            assert( copy_size != 0 );
            copy_size += from[ i ].m_size * sizeof( TYPE );
         }

         // Check if the next vector element connects to this one. If not,
         // perform the copy now.
         if( ( ( i + 1 ) >= size ) ||
             ( ( from[ i ].m_ptr + from[ i ].m_size ) != from[ i + 1 ].m_ptr ) ||
             ( ( to[ i ].m_ptr + to[ i ].m_size ) != to[ i + 1 ].m_ptr ) ) {

            // Perform the copy.
            copy_raw( copy_size, from_ptr, to_ptr, type );

            // Reset/update the variables.
            from_ptr = nullptr;
            to_ptr = nullptr;
            copy_size = 0;
            copy_ops += 1;
         }
      }

      // Let the user know what happened.
      VECMEM_DEBUG_MSG( 2, "Copied the payload of a jagged vector of type "
                        "\"%s\" with %lu copy operations",
                        typeid( TYPE ).name(), copy_ops );
   }

} // namespace vecmem::cuda::details
