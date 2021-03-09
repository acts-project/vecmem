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

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   device_vector< TYPE >::
   device_vector( details::vector_view< value_type > data )
   : m_size( data.m_size ), m_ptr( data.m_ptr ) {

   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   device_vector< TYPE >::device_vector( const device_vector& parent )
   : m_size( parent.m_size ), m_ptr( parent.m_ptr ) {

   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   device_vector< TYPE >&
   device_vector< TYPE >::operator=( const device_vector& rhs ) {

      // Prevent self-assignment.
      if( this == &rhs ) {
         return *this;
      }

      // Copy the other object's payload.
      m_size = rhs.m_size;
      m_ptr = rhs.m_ptr;

      // Return a reference to this object.
      return *this;
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::reference
   device_vector< TYPE >::at( size_type pos ) {

      // Check if the index is valid.
      assert( pos < m_size );

      // Return a reference to the vector element.
      return m_ptr[ pos ];
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::const_reference
   device_vector< TYPE >::at( size_type pos ) const {

      // Check if the index is valid.
      assert( pos < m_size );

      // Return a reference to the vector element.
      return m_ptr[ pos ];
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::reference
   device_vector< TYPE >::operator[]( size_type pos ) {

      // Return a reference to the vector element.
      return m_ptr[ pos ];
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::const_reference
   device_vector< TYPE >::operator[]( size_type pos ) const {

      // Return a reference to the vector element.
      return m_ptr[ pos ];
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::reference
   device_vector< TYPE >::front() {

      // Make sure that there is at least one element in the vector.
      assert( m_size > 0 );

      // Return a reference to the first element of the vector.
      return m_ptr[ 0 ];
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::const_reference
   device_vector< TYPE >::front() const {

      // Make sure that there is at least one element in the vector.
      assert( m_size > 0 );

      // Return a reference to the first element of the vector.
      return m_ptr[ 0 ];
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::reference
   device_vector< TYPE >::back() {

      // Make sure that there is at least one element in the vector.
      assert( m_size > 0 );

      // Return a reference to the last element of the vector.
      return m_ptr[ m_size - 1 ];
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::const_reference
   device_vector< TYPE >::back() const {

      // Make sure that there is at least one element in the vector.
      assert( m_size > 0 );

      // Return a reference to the last element of the vector.
      return m_ptr[ m_size - 1 ];
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::pointer
   device_vector< TYPE >::data() {

      return m_ptr;
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::const_pointer
   device_vector< TYPE >::data() const {

      return m_ptr;
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::iterator
   device_vector< TYPE >::begin() {

      return iterator( m_ptr );
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::const_iterator
   device_vector< TYPE >::begin() const {

      return const_iterator( m_ptr );
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::const_iterator
   device_vector< TYPE >::cbegin() const {

      return begin();
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::iterator
   device_vector< TYPE >::end() {

      return iterator( m_ptr + m_size );
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::const_iterator
   device_vector< TYPE >::end() const {

      return const_iterator( m_ptr + m_size );
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::const_iterator
   device_vector< TYPE >::cend() const {

      return end();
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::reverse_iterator
   device_vector< TYPE >::rbegin() {

      return reverse_iterator( end() );
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::const_reverse_iterator
   device_vector< TYPE >::rbegin() const {

      return const_reverse_iterator( end() );
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::const_reverse_iterator
   device_vector< TYPE >::crbegin() const {

      return rbegin();
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::reverse_iterator
   device_vector< TYPE >::rend() {

      return reverse_iterator( begin() );
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::const_reverse_iterator
   device_vector< TYPE >::rend() const {

      return const_reverse_iterator( begin() );
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::const_reverse_iterator
   device_vector< TYPE >::crend() const {

      return rend();
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   bool device_vector< TYPE >::empty() const {

      return m_size == 0;
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::size_type
   device_vector< TYPE >::size() const {

      return m_size;
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::size_type
   device_vector< TYPE >::max_size() const {

      return size();
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::size_type
   device_vector< TYPE >::capacity() const {

      return size();
   }

} // namespace vecmem
