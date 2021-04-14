/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// VecMem include(s).
#include "vecmem/utils/debug.hpp"
#include "vecmem/utils/type_traits.hpp"

// System include(s).
#include <cassert>
#include <type_traits>

namespace vecmem {

   template< typename TYPE >
   data::vector_buffer< TYPE >
   copy::to( const vecmem::data::vector_view< TYPE >& data,
             memory_resource& resource, type::copy_type cptype ) {

      data::vector_buffer< TYPE > result( data.m_size, resource );
      do_copy( data.m_size * sizeof( TYPE ), data.m_ptr, result.m_ptr, cptype );
      VECMEM_DEBUG_MSG( 2, "Created a vector buffer of type \"%s\" with "
                        "size %lu", typeid( TYPE ).name(), data.m_size );
      return result;
   }

   template< typename TYPE >
   void copy::operator()( const data::vector_view< TYPE >& from,
                          data::vector_view< TYPE >& to,
                          type::copy_type cptype ) {

      assert( from.m_size == to.m_size );
      do_copy( from.m_size * sizeof( TYPE ), from.m_ptr, to.m_ptr, cptype );
   }

   template< typename TYPE1, typename TYPE2, typename ALLOC >
   void copy::operator()( const data::vector_view< TYPE1 >& from,
                          std::vector< TYPE2, ALLOC >& to,
                          type::copy_type cptype ) {

      // The input and output types are allowed to be different, but only by
      // const-ness.
      static_assert( std::is_same< TYPE1, TYPE2 >::value ||
                     details::is_same_nc< TYPE1, TYPE2 >::value ||
                     details::is_same_nc< TYPE2, TYPE1 >::value,
                     "Can only use compatible types in the copy" );
      // Make the target vector the correct size.
      to.resize( from.m_size );
      // Perform the memory copy.
      do_copy( from.m_size * sizeof( TYPE1 ), from.m_ptr, to.data(), cptype );
   }

   template< typename TYPE >
   void copy::setup( data::jagged_vector_buffer< TYPE >& data ) {

      // Check if anything needs to be done.
      if( ( data.m_ptr == data.host_ptr() ) || ( data.m_size == 0 ) ) {
         return;
      }

      // Copy the description of the inner vectors of the buffer.
      do_copy(
         data.m_size * sizeof(
            typename vecmem::data::jagged_vector_buffer< TYPE >::value_type ),
         data.host_ptr(), data.m_ptr, type::host_to_device );
      VECMEM_DEBUG_MSG( 2, "Prepared a jagged device vector buffer of size %lu "
                        "for use on a device", data.m_size );
   }

   template< typename TYPE >
   data::jagged_vector_buffer< TYPE >
   copy::to( const data::jagged_vector_view< TYPE >& data,
             memory_resource& resource, memory_resource* host_access_resource,
             type::copy_type cptype ) {

      // Create the result buffer object.
      data::jagged_vector_buffer< TYPE > result( data, resource,
                                                 host_access_resource );
      assert( result.m_size == data.m_size );

      // Copy the description of the "inner vectors" if necessary.
      setup( result );

      // Copy the payload of the inner vectors.
      copy_views( data.m_size, data.m_ptr, result.host_ptr(), cptype );

      // Return the newly created object.
      return result;
   }

   template< typename TYPE >
   data::jagged_vector_buffer< TYPE >
   copy::to( const data::jagged_vector_buffer< TYPE >& data,
             memory_resource& resource, memory_resource* host_access_resource,
             type::copy_type cptype ) {

      // Create the result buffer object.
      data::jagged_vector_buffer< TYPE > result( data, resource,
                                                 host_access_resource );
      assert( result.m_size == data.m_size );

      // Copy the description of the "inner vectors" if necessary.
      setup( result );

      // Copy the payload of the inner vectors.
      copy_views( data.m_size, data.host_ptr(), result.host_ptr(), cptype );

      // Return the newly created object.
      return result;
   }

   template< typename TYPE >
   void copy::operator()( const data::jagged_vector_view< TYPE >& from,
                          data::jagged_vector_view< TYPE >& to,
                          type::copy_type cptype ) {

      // A sanity check.
      assert( from.m_size == to.m_size );

      // Copy the payload of the inner vectors.
      copy_views( from.m_size, from.m_ptr, to.m_ptr, cptype );
   }

   template< typename TYPE >
   void copy::operator()( const data::jagged_vector_view< TYPE >& from,
                          data::jagged_vector_buffer< TYPE >& to,
                          type::copy_type cptype ) {

      // A sanity check.
      assert( from.m_size == to.m_size );

      // Copy the payload of the inner vectors.
      copy_views( from.m_size, from.m_ptr, to.host_ptr(), cptype );
   }

   template< typename TYPE >
   void copy::operator()( const data::jagged_vector_buffer< TYPE >& from,
                          data::jagged_vector_view< TYPE >& to,
                          type::copy_type cptype ) {

      // A sanity check.
      assert( from.m_size == to.m_size );

      // Copy the payload of the inner vectors.
      copy_views( from.m_size, from.host_ptr(), to.m_ptr, cptype );
   }

   template< typename TYPE >
   void copy::operator()( const data::jagged_vector_buffer< TYPE >& from,
                          data::jagged_vector_buffer< TYPE >& to,
                          type::copy_type cptype ) {

      // A sanity check.
      assert( from.m_size == to.m_size );

      // Copy the payload of the inner vectors.
      copy_views( from.m_size, from.host_ptr(), to.host_ptr(), cptype );
   }

   template< typename TYPE >
   void copy::copy_views( std::size_t size,
                          const data::vector_view< TYPE >* from,
                          data::vector_view< TYPE >* to,
                          type::copy_type cptype ) {

      // Helper variables used in the copy.
      const TYPE* from_ptr = nullptr;
      TYPE* to_ptr = nullptr;
      std::size_t copy_size = 0, copy_ops = 0;

      // Helper lambda for figuring out if the next vector element is
      // connected to the currently processed one or not.
      auto next_is_connected = [ size ]( const data::vector_view< TYPE >* array,
                                         std::size_t i ) {
            // Check if the next non-empty vector element is connected to the
            // current one.
            std::size_t j = i + 1;
            while( j < size ) {
               if( array[ j ].m_size == 0 ) {
                  ++j;
                  continue;
               }
               return ( ( array[ i ].m_ptr + array[ i ].m_size ) ==
                        array[ j ].m_ptr );
            }
            // If we got here, then the answer is no...
            return false;
         };

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
         if( ( ! next_is_connected( from, i ) ) ||
             ( ! next_is_connected( to, i ) ) ) {

            // Perform the copy.
            do_copy( copy_size, from_ptr, to_ptr, cptype );

            // Reset/update the variables.
            from_ptr = nullptr;
            to_ptr = nullptr;
            copy_size = 0;
            copy_ops += 1;
         }
      }

      // Let the user know what happened.
      VECMEM_DEBUG_MSG( 2, "Copied the payload of a jagged vector of type "
                        "\"%s\" with %lu copy operation(s)",
                        typeid( TYPE ).name(), copy_ops );
   }

} // namespace vecmem
