/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <cassert>

namespace vecmem {

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   device_array< T, N >::
   device_array( const data::vector_view< value_type >& data )
   : m_ptr( data.ptr() ) {

      assert( data.size() >= N );
   }

   template< typename T, std::size_t N >
   template< typename OTHERTYPE,
             std::enable_if_t<
                details::is_same_nc< T, OTHERTYPE >::value,
                bool > >
   VECMEM_HOST_AND_DEVICE
   device_array< T, N >::
   device_array( const data::vector_view< OTHERTYPE >& data )
   : m_ptr( data.ptr() ) {

      assert( data.size() >= N );
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   device_array< T, N >::device_array( const device_array& parent )
   : m_ptr( parent.m_ptr ) {

   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   device_array< T, N >&
   device_array< T, N >::operator=( const device_array& rhs ) {

      // Prevent self-assignment.
      if( this == &rhs ) {
         return *this;
      }

      // Copy the other object's payload.
      m_ptr = rhs.m_ptr;

      // Return a reference to this object.
      return *this;
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::reference
   device_array< T, N >::at( size_type pos ) {

      // Check if the index is valid.
      assert( pos < N );

      // Return a reference to the vector element.
      return m_ptr[ pos ];
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::const_reference
   device_array< T, N >::at( size_type pos ) const {

      // Check if the index is valid.
      assert( pos < N );

      // Return a reference to the vector element.
      return m_ptr[ pos ];
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::reference
   device_array< T, N >::operator[]( size_type pos ) {

      // Return a reference to the vector element.
      return m_ptr[ pos ];
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::const_reference
   device_array< T, N >::operator[]( size_type pos ) const {

      // Return a reference to the vector element.
      return m_ptr[ pos ];
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::reference
   device_array< T, N >::front() {

      // Make sure that there is at least one element in the vector.
      static_assert( N > 0, "Cannot return first element of empty array" );

      // Return a reference to the first element of the vector.
      return m_ptr[ 0 ];
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::const_reference
   device_array< T, N >::front() const {

      // Make sure that there is at least one element in the vector.
      static_assert( N > 0, "Cannot return first element of empty array" );

      // Return a reference to the first element of the vector.
      return m_ptr[ 0 ];
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::reference
   device_array< T, N >::back() {

      // Make sure that there is at least one element in the vector.
      static_assert( N > 0, "Cannot return last element of empty array" );

      // Return a reference to the last element of the vector.
      return m_ptr[ N - 1 ];
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::const_reference
   device_array< T, N >::back() const {

      // Make sure that there is at least one element in the vector.
      static_assert( N > 0, "Cannot return last element of empty array" );

      // Return a reference to the last element of the vector.
      return m_ptr[ N - 1 ];
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::pointer
   device_array< T, N >::data() {

      return m_ptr;
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::const_pointer
   device_array< T, N >::data() const {

      return m_ptr;
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::iterator
   device_array< T, N >::begin() {

      return iterator( m_ptr );
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::const_iterator
   device_array< T, N >::begin() const {

      return const_iterator( m_ptr );
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::const_iterator
   device_array< T, N >::cbegin() const {

      return begin();
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::iterator
   device_array< T, N >::end() {

      return iterator( m_ptr + N );
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::const_iterator
   device_array< T, N >::end() const {

      return const_iterator( m_ptr + N );
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::const_iterator
   device_array< T, N >::cend() const {

      return end();
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::reverse_iterator
   device_array< T, N >::rbegin() {

      return reverse_iterator( end() );
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::const_reverse_iterator
   device_array< T, N >::rbegin() const {

      return const_reverse_iterator( end() );
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::const_reverse_iterator
   device_array< T, N >::crbegin() const {

      return rbegin();
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::reverse_iterator
   device_array< T, N >::rend() {

      return reverse_iterator( begin() );
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::const_reverse_iterator
   device_array< T, N >::rend() const {

      return const_reverse_iterator( begin() );
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   typename device_array< T, N >::const_reverse_iterator
   device_array< T, N >::crend() const {

      return rend();
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   constexpr bool device_array< T, N >::empty() const {

      return N == 0;
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   constexpr typename device_array< T, N >::size_type
   device_array< T, N >::size() const {

      return N;
   }

   template< typename T, std::size_t N >
   VECMEM_HOST_AND_DEVICE
   constexpr typename device_array< T, N >::size_type
   device_array< T, N >::max_size() const {

      return size();
   }

} // namespace vecmem
