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
      template< typename T, std::size_t N, typename A >
      typename vecmem::array< T, N, A >::memory_type
      allocate_array_memory(
         typename vecmem::array< T, N, A >::allocator_type& alloc,
         typename vecmem::array< T, N, A >::size_type size ) {

         return { size == 0 ? nullptr : alloc.allocate( size ),
                  { size, alloc } };
      }

   } // namespace details

   template< typename T, std::size_t N, typename Allocator >
   array< T, N, Allocator >::array( allocator_type alloc )
   : m_size( N ),
     m_memory( details::allocate_array_memory< T, N, Allocator >( alloc,
                                                                  m_size ) ) {

      static_assert( N != details::array_invalid_size,
                     "Can only use the 'compile time constructor' if a size "
                     "was provided as a template argument" );
   }

   template< typename T, std::size_t N, typename Allocator >
   array< T, N, Allocator >::array( allocator_type alloc,
                                    size_type size )
   : m_size( size ),
     m_memory( details::allocate_array_memory< T, N, Allocator >( alloc,
                                                                  m_size ) ) {

      static_assert( N == details::array_invalid_size,
                     "Can only use the 'runtime constructor' if a size was not "
                     "provided as a template argument" );
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::reference_type
   array< T, N, Allocator >::at( size_type pos ) {

      if( pos >= m_size ) {
         throw std::out_of_range( "Requested element " + std::to_string( pos ) +
                                  " from a " + std::to_string( m_size ) +
                                  " sized vecmem::array" );
      }
      return m_memory.get()[ pos ];
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::const_reference_type
   array< T, N, Allocator >::at( size_type pos ) const {

      if( pos >= m_size ) {
         throw std::out_of_range( "Requested element " + std::to_string( pos ) +
                                  " from a " + std::to_string( m_size ) +
                                  " sized vecmem::array" );
      }
      return m_memory.get()[ pos ];
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::reference_type
   array< T, N, Allocator >::operator[]( size_type pos ) {

      return m_memory.get()[ pos ];
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::const_reference_type
   array< T, N, Allocator >::operator[]( size_type pos ) const {

      return m_memory.get()[ pos ];
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::reference_type
   array< T, N, Allocator >::front() {

      if( m_size == 0 ) {
         throw std::out_of_range( "Called vecmem::array::front() on an empty "
                                  "array" );
      }
      return ( *m_memory );
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::const_reference_type
   array< T, N, Allocator >::front() const {

      if( m_size == 0 ) {
         throw std::out_of_range( "Called vecmem::array::front() on an empty "
                                  "array" );
      }
      return ( *m_memory );
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::reference_type
   array< T, N, Allocator >::back() {

      if( m_size == 0 ) {
         throw std::out_of_range( "Called vecmem::array::back() on an empty "
                                  "array" );
      }
      return m_memory.get()[ m_size - 1 ];
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::const_reference_type
   array< T, N, Allocator >::back() const {

      if( m_size == 0 ) {
         throw std::out_of_range( "Called vecmem::array::back() on an empty "
                                  "array" );
      }
      return m_memory.get()[ m_size - 1 ];
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::pointer_type
   array< T, N, Allocator >::data() {

      return m_memory.get();
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::const_pointer_type
   array< T, N, Allocator >::data() const {

      return m_memory.get();
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::iterator
   array< T, N, Allocator >::begin() {

      return m_memory.get();
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::const_iterator
   array< T, N, Allocator >::begin() const {

      return m_memory.get();
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::const_iterator
   array< T, N, Allocator >::cbegin() const {

      return m_memory.get();
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::iterator
   array< T, N, Allocator >::end() {

      return ( m_memory.get() + m_size );
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::const_iterator
   array< T, N, Allocator >::end() const {

      return ( m_memory.get() + m_size );
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::const_iterator
   array< T, N, Allocator >::cend() const {

      return ( m_memory.get() + m_size );
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::reverse_iterator
   array< T, N, Allocator >::rbegin() {

      return reverse_iterator( end() );
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::const_reverse_iterator
   array< T, N, Allocator >::rbegin() const {

      return const_reverse_iterator( end() );
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::const_reverse_iterator
   array< T, N, Allocator >::crbegin() const {

      return const_reverse_iterator( end() );
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::reverse_iterator
   array< T, N, Allocator >::rend() {

      return reverse_iterator( begin() );
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::const_reverse_iterator
   array< T, N, Allocator >::rend() const {

      return const_reverse_iterator( begin() );
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::const_reverse_iterator
   array< T, N, Allocator >::crend() const {

      return const_reverse_iterator( begin() );
   }

   template< typename T, std::size_t N, typename Allocator >
   bool array< T, N, Allocator >::empty() const noexcept {

      return ( m_size == 0 );
   }

   template< typename T, std::size_t N, typename Allocator >
   typename array< T, N, Allocator >::size_type
   array< T, N, Allocator >::size() const noexcept {

      return m_size;
   }

   template< typename T, std::size_t N, typename Allocator >
   void array< T, N, Allocator >::fill( const_reference_type value ) {

      std::fill( begin(), end(), value );
   }

   template< typename T, std::size_t N, typename Allocator >
   VECMEM_HOST
   details::vector_view< T >
   get_data( array< T, N, Allocator >& a ) {

      return { a.size(), a.data() };
   }

   template< typename T, std::size_t N, typename Allocator >
   VECMEM_HOST
   details::vector_view< const T >
   get_data( const array< T, N, Allocator >& a ) {

      return { a.size(), a.data() };
   }

} // namespace vecmem
