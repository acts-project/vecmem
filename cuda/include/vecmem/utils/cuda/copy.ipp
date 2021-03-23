/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <cstddef>

namespace vecmem::cuda {

   namespace details {

      /// Helper function performing the actual H->D copy
      void copy_to_device( std::size_t size, const void* hostPtr,
                           void* devicePtr );
      /// Helper function performing the actual D->H copy
      void copy_to_host( std::size_t size, const void* devicePtr,
                         void* hostPtr );
      /// Helper function performing an unspecified type of copy
      void copy( std::size_t size, const void* from, void* to );

   } // namespace details

   template< typename TYPE >
   vecmem::data::vector_buffer< TYPE >
   copy_to_device( const vecmem::data::vector_view< TYPE >& host,
                   memory_resource& resource ) {

      vecmem::data::vector_buffer< TYPE > device( host.m_size, resource );
      details::copy_to_device( host.m_size * sizeof( TYPE ), host.m_ptr,
                               device.m_ptr );
      return device;
   }

   template< typename TYPE >
   vecmem::data::vector_buffer< TYPE >
   copy_to_host( const vecmem::data::vector_view< TYPE >& device,
                 memory_resource& resource ) {

      vecmem::data::vector_buffer< TYPE > host( device.m_size, resource );
      details::copy_to_host( device.m_size * sizeof( TYPE ), device.m_ptr,
                             host.m_ptr );
      return host;
   }

   template< typename TYPE >
   void copy( const vecmem::data::vector_view< TYPE >& from,
              vecmem::data::vector_view< TYPE >& to ) {

      details::copy( from.m_size * sizeof( TYPE ), from.m_ptr, to.m_ptr );
   }

} // namespace vecmem::cuda
