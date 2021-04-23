/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

namespace vecmem {

   namespace details {

      /// Helper function used in the @c vecmem::array constructors
      template< typename T, std::size_t N >
      std::unique_ptr< typename vecmem::array< T, N >::value_type,
                       typename vecmem::array< T, N >::deleter >
      allocate_array_memory( vecmem::memory_resource& resource,
                             typename vecmem::array< T, N >::size_type size ) {

         return { size == 0 ? nullptr :
                  static_cast< typename vecmem::array< T, N >::pointer >(
                     resource.allocate(
                        size * sizeof( typename
                                       vecmem::array< T, N >::value_type ) ) ),
                  { size, resource } };
      }

      /// Helper function used in the @c vecmem::array constructors
      template< typename T, std::size_t N >
      std::unique_ptr< typename vecmem::array< T, N >::value_type,
                       typename vecmem::array< T, N >::deleter >
      initialize_array_memory(
         std::unique_ptr< typename vecmem::array< T, N >::value_type,
                          typename vecmem::array< T, N >::deleter > memory,
         typename vecmem::array< T, N >::size_type size ) {

         typename vecmem::array< T, N >::size_type i = 0;
         for( typename vecmem::array< T, N >::pointer ptr = memory.get();
              i < size; ++i, ++ptr ) {
            new( ptr ) typename vecmem::array< T, N >::value_type();
         }
         return memory;
      }

   } // namespace details

   template< typename T, std::size_t N >
   array< T, N >::deleter::deleter( size_type size,
                                    memory_resource& resource )
   : m_size( size ), m_resource( &resource ) {

   }

   template< typename T, std::size_t N >
   void array< T, N >::deleter::operator()( void* ptr ) {

      // Call the destructor on all objects.
      size_type i = 0;
      for( pointer p = reinterpret_cast< pointer >( ptr ); i < m_size;
           ++i, ++p ) {
         p->~value_type();
      }
      // De-allocate the array's memory.
      if( ( m_size != 0 ) && ( ptr != nullptr ) ) {
         m_resource->deallocate( ptr, m_size * sizeof( value_type ) );
      }
   }

   template< typename T, std::size_t N >
   array< T, N >::array( memory_resource& resource )
   : m_size( N ),
     m_memory( details::initialize_array_memory< T, N >(
        details::allocate_array_memory< T, N >( resource, m_size ), m_size ) ) {

      static_assert( N != details::array_invalid_size,
                     "Can only use the 'compile time constructor' if a size "
                     "was provided as a template argument" );
   }

   template< typename T, std::size_t N >
   array< T, N >::array( memory_resource& resource, size_type size )
   : m_size( size ),
     m_memory( details::initialize_array_memory< T, N >(
        details::allocate_array_memory< T, N >( resource, m_size ), m_size ) ) {

      static_assert( N == details::array_invalid_size,
                     "Can only use the 'runtime constructor' if a size was not "
                     "provided as a template argument" );
   }

   template< typename T, std::size_t N >
   typename array< T, N >::reference
   array< T, N >::at( size_type pos ) {

      if( pos >= m_size ) {
         throw std::out_of_range( "Requested element " + std::to_string( pos ) +
                                  " from a " + std::to_string( m_size ) +
                                  " sized vecmem::array" );
      }
      return m_memory.get()[ pos ];
   }

   template< typename T, std::size_t N >
   typename array< T, N >::const_reference
   array< T, N >::at( size_type pos ) const {

      if( pos >= m_size ) {
         throw std::out_of_range( "Requested element " + std::to_string( pos ) +
                                  " from a " + std::to_string( m_size ) +
                                  " sized vecmem::array" );
      }
      return m_memory.get()[ pos ];
   }

   template< typename T, std::size_t N >
   typename array< T, N >::reference
   array< T, N >::operator[]( size_type pos ) {

      return m_memory.get()[ pos ];
   }

   template< typename T, std::size_t N >
   typename array< T, N >::const_reference
   array< T, N >::operator[]( size_type pos ) const {

      return m_memory.get()[ pos ];
   }

   template< typename T, std::size_t N >
   typename array< T, N >::reference
   array< T, N >::front() {

      if( m_size == 0 ) {
         throw std::out_of_range( "Called vecmem::array::front() on an empty "
                                  "array" );
      }
      return ( *m_memory );
   }

   template< typename T, std::size_t N >
   typename array< T, N >::const_reference
   array< T, N >::front() const {

      if( m_size == 0 ) {
         throw std::out_of_range( "Called vecmem::array::front() on an empty "
                                  "array" );
      }
      return ( *m_memory );
   }

   template< typename T, std::size_t N >
   typename array< T, N >::reference
   array< T, N >::back() {

      if( m_size == 0 ) {
         throw std::out_of_range( "Called vecmem::array::back() on an empty "
                                  "array" );
      }
      return m_memory.get()[ m_size - 1 ];
   }

   template< typename T, std::size_t N >
   typename array< T, N >::const_reference
   array< T, N >::back() const {

      if( m_size == 0 ) {
         throw std::out_of_range( "Called vecmem::array::back() on an empty "
                                  "array" );
      }
      return m_memory.get()[ m_size - 1 ];
   }

   template< typename T, std::size_t N >
   typename array< T, N >::pointer
   array< T, N >::data() {

      return m_memory.get();
   }

   template< typename T, std::size_t N >
   typename array< T, N >::const_pointer
   array< T, N >::data() const {

      return m_memory.get();
   }

   template< typename T, std::size_t N >
   typename array< T, N >::iterator
   array< T, N >::begin() {

      return m_memory.get();
   }

   template< typename T, std::size_t N >
   typename array< T, N >::const_iterator
   array< T, N >::begin() const {

      return m_memory.get();
   }

   template< typename T, std::size_t N >
   typename array< T, N >::const_iterator
   array< T, N >::cbegin() const {

      return m_memory.get();
   }

   template< typename T, std::size_t N >
   typename array< T, N >::iterator
   array< T, N >::end() {

      return ( m_memory.get() + m_size );
   }

   template< typename T, std::size_t N >
   typename array< T, N >::const_iterator
   array< T, N >::end() const {

      return ( m_memory.get() + m_size );
   }

   template< typename T, std::size_t N >
   typename array< T, N >::const_iterator
   array< T, N >::cend() const {

      return ( m_memory.get() + m_size );
   }

   template< typename T, std::size_t N >
   typename array< T, N >::reverse_iterator
   array< T, N >::rbegin() {

      return reverse_iterator( end() );
   }

   template< typename T, std::size_t N >
   typename array< T, N >::const_reverse_iterator
   array< T, N >::rbegin() const {

      return const_reverse_iterator( end() );
   }

   template< typename T, std::size_t N >
   typename array< T, N >::const_reverse_iterator
   array< T, N >::crbegin() const {

      return const_reverse_iterator( end() );
   }

   template< typename T, std::size_t N >
   typename array< T, N >::reverse_iterator
   array< T, N >::rend() {

      return reverse_iterator( begin() );
   }

   template< typename T, std::size_t N >
   typename array< T, N >::const_reverse_iterator
   array< T, N >::rend() const {

      return const_reverse_iterator( begin() );
   }

   template< typename T, std::size_t N >
   typename array< T, N >::const_reverse_iterator
   array< T, N >::crend() const {

      return const_reverse_iterator( begin() );
   }

   template< typename T, std::size_t N >
   bool array< T, N >::empty() const noexcept {

      return ( m_size == 0 );
   }

   template< typename T, std::size_t N >
   typename array< T, N >::size_type
   array< T, N >::size() const noexcept {

      return m_size;
   }

   template< typename T, std::size_t N >
   void array< T, N >::fill( const_reference value ) {

      std::fill( begin(), end(), value );
   }

   template< typename T, std::size_t N >
   VECMEM_HOST
   data::vector_view< T >
   get_data( array< T, N >& a ) {

      return {
         static_cast<
            typename data::vector_view< T >::size_type >( a.size() ),
         a.data() };
   }

   template< typename T, std::size_t N >
   VECMEM_HOST
   data::vector_view< const T >
   get_data( const array< T, N >& a ) {

      return {
         static_cast<
            typename data::vector_view< const T >::size_type >( a.size() ),
         a.data() };
   }

} // namespace vecmem
