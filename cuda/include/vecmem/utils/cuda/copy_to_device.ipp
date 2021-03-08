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
      void copy_to_device( std::size_t size, const void* hostPtr,
                           void* devicePtr );

   } // namespace details

   template< typename TYPE >
   vecmem::details::owning_vector_data< typename std::remove_cv< TYPE >::type >
   copy_to_device( const vecmem::details::vector_data< TYPE >& host,
                   memory_resource& resource ) {

      vecmem::details::owning_vector_data<
         typename std::remove_cv< TYPE >::type >
            device( host.m_size, resource );
      details::copy_to_device( host.m_size * sizeof( TYPE ), host.m_ptr,
                               device.m_ptr );
      return device;
   }

   template< typename TYPE >
   void copy_to_device( const vecmem::details::vector_data< TYPE >& host,
                        vecmem::details::vector_data<
                           typename std::remove_cv< TYPE >::type >& device ) {

      details::copy_to_device( host.m_size * sizeof( TYPE ), host.m_ptr,
                               device.m_ptr );
   }

} // namespace vecmem::cuda
