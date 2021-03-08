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

      /// Helper function performing the actual copy
      void copy_to_host( std::size_t size, const void* devicePtr,
                         void* hostPtr );

   } // namespace details

   template< typename TYPE >
   vecmem::details::owning_vector_data< typename std::remove_cv< TYPE >::type >
   copy_to_host( const vecmem::details::vector_data< TYPE >& device,
                 memory_resource& resource ) {

      vecmem::details::owning_vector_data<
         typename std::remove_cv< TYPE >::type >
            host( device.m_size, resource );
      details::copy_to_host( device.m_size * sizeof( TYPE ), device.m_ptr,
                             host.m_ptr );
      return host;
   }

   template< typename TYPE >
   void copy_to_host( const vecmem::details::vector_data< TYPE >& device,
                      vecmem::details::vector_data<
                         typename std::remove_cv< TYPE >::type >& host ) {

      details::copy_to_host( device.m_size * sizeof( TYPE ), device.m_ptr,
                             host.m_ptr );
   }

} // namespace vecmem::cuda
