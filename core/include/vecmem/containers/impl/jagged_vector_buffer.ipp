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
#include <numeric>
#include <vector>

namespace {

   /// Helper conversion function
   template< typename TYPE >
   std::vector< std::size_t >
   get_sizes( const vecmem::data::jagged_vector_view< TYPE >& jvv ) {

      std::vector< std::size_t > result( jvv.m_size );
      for( std::size_t i = 0; i < jvv.m_size; ++i ) {
         result[ i ] = jvv.m_ptr[ i ].size();
      }
      return result;
   }

   /// Function allocating memory for @c vecmem::data::jagged_vector_buffer
   template< typename TYPE >
   std::unique_ptr<
      typename vecmem::data::jagged_vector_buffer< TYPE >::value_type,
      vecmem::details::deallocator >
   allocate_jagged_buffer_outer_memory(
      typename vecmem::data::jagged_vector_buffer< TYPE >::size_type size,
      vecmem::memory_resource& resource ) {

      const std::size_t byteSize =
         size *
         sizeof(
            typename vecmem::data::jagged_vector_buffer< TYPE >::value_type );
      return { size == 0 ? nullptr :
               static_cast<
                  typename vecmem::data::jagged_vector_buffer< TYPE >::pointer >(
                     resource.allocate( byteSize ) ),
               { byteSize, resource } };
   }

   /// Function allocating memory for @c vecmem::data::jagged_vector_buffer
   template< typename TYPE >
   std::unique_ptr< TYPE, vecmem::details::deallocator >
   allocate_jagged_buffer_inner_memory(
      const std::vector< std::size_t >& sizes,
      vecmem::memory_resource& resource ) {

      const typename vecmem::data::jagged_vector_buffer< TYPE >::size_type
         byteSize = std::accumulate( sizes.begin(), sizes.end(), 0 ) *
                    sizeof( TYPE );
      return { byteSize == 0 ? nullptr :
               static_cast< TYPE* >( resource.allocate( byteSize ) ),
               { byteSize, resource } };
   }

} // private namespace

namespace vecmem { namespace data {

   template< typename TYPE >
   template< typename OTHERTYPE,
             std::enable_if_t< std::is_convertible< TYPE, OTHERTYPE >::value,
                               bool > >
   jagged_vector_buffer< TYPE >::
   jagged_vector_buffer( const jagged_vector_view< OTHERTYPE >& other,
                         memory_resource& resource,
                         memory_resource* host_access_resource )
   : jagged_vector_buffer( ::get_sizes( other ), resource,
                           host_access_resource ) {

   }

   template< typename TYPE >
   jagged_vector_buffer< TYPE >::
   jagged_vector_buffer( const std::vector< std::size_t >& sizes,
                         memory_resource& resource,
                         memory_resource* host_access_resource )
   : base_type( sizes.size(), nullptr ),
     m_outer_memory(
        ::allocate_jagged_buffer_outer_memory< TYPE >(
           ( host_access_resource == nullptr ? 0 : sizes.size() ),
           resource ) ),
     m_outer_host_memory(
        ::allocate_jagged_buffer_outer_memory< TYPE >( sizes.size(),
           ( host_access_resource == nullptr ? resource :
             *host_access_resource ) ) ),
     m_inner_memory(
        ::allocate_jagged_buffer_inner_memory< TYPE >( sizes, resource ) ) {

      // Point the base class at the newly allocated memory.
      base_type::m_ptr = ( ( host_access_resource != nullptr ) ?
                           m_outer_memory.get() : m_outer_host_memory.get() );

      // Set up the host accessible memory array.
      std::ptrdiff_t ptrdiff = 0;
      for( std::size_t i = 0; i < sizes.size(); ++i ) {
         new( host_ptr() + i ) value_type();
         host_ptr()[ i ] =
            value_type( sizes[ i ], m_inner_memory.get() + ptrdiff );
         ptrdiff += sizes[ i ];
      }
   }

   template< typename TYPE >
   typename jagged_vector_buffer< TYPE >::pointer
   jagged_vector_buffer< TYPE >::host_ptr() const {

      return m_outer_host_memory.get();
   }

} // namespace data

   template< typename TYPE >
   data::jagged_vector_view< TYPE >&
   get_data( data::jagged_vector_buffer< TYPE >& data ) {

      return data;
   }

   template< typename TYPE >
   const data::jagged_vector_view< TYPE >&
   get_data( const data::jagged_vector_buffer< TYPE >& data ) {

      return data;
   }

} // namespace vecmem
