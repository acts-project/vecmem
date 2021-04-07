/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/utils/debug.hpp"
#include "vecmem/utils/cuda/impl/copy_impl.hpp"

// System include(s).
#include <cassert>

namespace vecmem::cuda {

   template< typename TYPE >
   vecmem::data::vector_buffer< TYPE >
   copy_to_device( const vecmem::data::vector_view< TYPE >& host,
                   memory_resource& resource ) {

      vecmem::data::vector_buffer< TYPE > device( host.m_size, resource );
      details::copy_raw( host.m_size * sizeof( TYPE ), host.m_ptr, device.m_ptr,
                         details::host_to_device );
      VECMEM_DEBUG_MSG( 2, "Created a device vector buffer of type \"%s\" with "
                        "size %lu", typeid( TYPE ).name(), host.m_size );
      return device;
   }

   template< typename TYPE >
   vecmem::data::vector_buffer< TYPE >
   copy_to_host( const vecmem::data::vector_view< TYPE >& device,
                 memory_resource& resource ) {

      vecmem::data::vector_buffer< TYPE > host( device.m_size, resource );
      details::copy_raw( device.m_size * sizeof( TYPE ), device.m_ptr,
                         host.m_ptr, details::device_to_host );
      VECMEM_DEBUG_MSG( 2, "Created a host vector buffer of type \"%s\" with "
                        "size %lu", typeid( TYPE ).name(), device.m_size );
      return host;
   }

   template< typename TYPE >
   void copy( const vecmem::data::vector_view< TYPE >& from,
              vecmem::data::vector_view< TYPE >& to ) {

      assert( from.m_size == to.m_size );
      details::copy_raw( from.m_size * sizeof( TYPE ), from.m_ptr, to.m_ptr );
   }

   template< typename TYPE >
   void prepare_for_device( vecmem::data::jagged_vector_buffer< TYPE >& data ) {

      // Check if anything needs to be done.
      if( ( data.m_ptr == data.host_ptr() ) || ( data.m_size == 0 ) ) {
         return;
      }

      // Copy the description of the inner vectors of the buffer.
      details::copy_raw(
         data.m_size * sizeof(
            typename vecmem::data::jagged_vector_buffer< TYPE >::value_type ),
         data.host_ptr(), data.m_ptr, details::host_to_device );
      VECMEM_DEBUG_MSG( 2, "Prepared a jagged device vector buffer of size %lu "
                        "for use on a device", data.m_size );
   }

   template< typename TYPE >
   vecmem::data::jagged_vector_buffer< TYPE >
   copy_to_device( const vecmem::data::jagged_vector_view< TYPE >& host,
                   memory_resource& device_resource,
                   memory_resource& host_resource ) {

      // Create the result buffer object.
      vecmem::data::jagged_vector_buffer< TYPE > result( host, device_resource,
                                                         &host_resource );
      assert( result.m_size == host.m_size );

      // Copy the description of the "inner vectors" if necessary.
      prepare_for_device( result );

      // Copy the payload of the inner vectors.
      details::copy_views( host.m_size, host.m_ptr, result.host_ptr(),
                           details::host_to_device );

      // Return the newly created object.
      return result;
   }

   template< typename TYPE >
   vecmem::data::jagged_vector_buffer< TYPE >
   copy_to_host( const vecmem::data::jagged_vector_buffer< TYPE >& device,
                 memory_resource& host_resource ) {

      // Create the result buffer object.
      vecmem::data::jagged_vector_buffer< TYPE >
         result( device, host_resource );
      assert( result.m_size == device.m_size );
      assert( result.m_ptr == result.host_ptr() );

      // Copy the payload of the inner vectors.
      details::copy_views( device.m_size, device.host_ptr(), result.m_ptr,
                           details::device_to_host );

      // Return the newly created object.
      return result;
   }

   template< typename TYPE >
   void copy( const vecmem::data::jagged_vector_view< TYPE >& from,
              vecmem::data::jagged_vector_view< TYPE >& to ) {

      // A sanity check.
      assert( from.m_size == to.m_size );

      // Copy the payload of the inner vectors.
      details::copy_views( from.m_size, from.m_ptr, to.m_ptr );
   }

   template< typename TYPE >
   void copy( const vecmem::data::jagged_vector_buffer< TYPE >& from,
              vecmem::data::jagged_vector_view< TYPE >& to ) {

      // A sanity check.
      assert( from.m_size == to.m_size );

      // Copy the payload of the inner vectors.
      details::copy_views( from.m_size, from.host_ptr(), to.m_ptr );
   }

   template< typename TYPE >
   void copy( const vecmem::data::jagged_vector_view< TYPE >& from,
              vecmem::data::jagged_vector_buffer< TYPE >& to ) {

      // A sanity check.
      assert( from.m_size == to.m_size );

      // Copy the payload of the inner vectors.
      details::copy_views( from.m_size, from.m_ptr, to.host_ptr() );
   }

   template< typename TYPE >
   void copy( const vecmem::data::jagged_vector_buffer< TYPE >& from,
              vecmem::data::jagged_vector_buffer< TYPE >& to ) {

      // A sanity check.
      assert( from.m_size == to.m_size );

      // Copy the payload of the inner vectors.
      details::copy_views( from.m_size, from.host_ptr(), to.host_ptr() );
   }

} // namespace vecmem::cuda
