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
   const_device_vector< TYPE >::
   const_device_vector( const_device_vector_data< value_type > data )
   : m_size( data.m_size ), m_ptr( data.m_ptr ) {

   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   const_device_vector< TYPE >::
   const_device_vector( const const_device_vector& parent )
   : m_size( parent.m_size ), m_ptr( parent.m_ptr ) {

   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   const_device_vector< TYPE >&
   const_device_vector< TYPE >::operator=( const const_device_vector& rhs ) {

      // Prevent self-assignment.
      if( this == &rhs ) {
         return *this;
      }

      // Copy the payload from the other object.
      m_size = rhs.m_size;
      m_ptr = rhs.m_ptr;

      // Return a reference to this object.
      return *this;
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename const_device_vector< TYPE >::const_reference
   const_device_vector< TYPE >::at( size_type pos ) const {

      // Check if the index is valid.
      assert( pos < m_size );

      // Return a reference to the vector element.
      return m_ptr[ pos ];
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename const_device_vector< TYPE >::const_reference
   const_device_vector< TYPE >::operator[]( size_type pos ) const {

      // Return a reference to the vector element.
      return m_ptr[ pos ];
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename const_device_vector< TYPE >::const_reference
   const_device_vector< TYPE >::front() const {

      // Make sure that there is at least one element in the vector.
      assert( m_size > 0 );

      // Return a reference to the first element of the vector.
      return m_ptr[ 0 ];
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename const_device_vector< TYPE >::const_reference
   const_device_vector< TYPE >::back() const {

      // Make sure that there is at least one element in the vector.
      assert( m_size > 0 );

      // Return a reference to the last element of the vector.
      return m_ptr[ m_size - 1 ];
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename const_device_vector< TYPE >::const_pointer
   const_device_vector< TYPE >::data() const {

      return m_ptr;
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename const_device_vector< TYPE >::const_iterator
   const_device_vector< TYPE >::begin() const {

      return const_iterator( m_ptr );
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename const_device_vector< TYPE >::const_iterator
   const_device_vector< TYPE >::cbegin() const {

      return begin();
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename const_device_vector< TYPE >::const_iterator
   const_device_vector< TYPE >::end() const {

      return const_iterator( m_ptr + m_size );
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename const_device_vector< TYPE >::const_iterator
   const_device_vector< TYPE >::cend() const {

      return end();
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename const_device_vector< TYPE >::const_reverse_iterator
   const_device_vector< TYPE >::rbegin() const {

      return const_reverse_iterator( end() );
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename const_device_vector< TYPE >::const_reverse_iterator
   const_device_vector< TYPE >::crbegin() const {

      return rbegin();
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename const_device_vector< TYPE >::const_reverse_iterator
   const_device_vector< TYPE >::rend() const {

      return const_reverse_iterator( begin() );
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename const_device_vector< TYPE >::const_reverse_iterator
   const_device_vector< TYPE >::crend() const {

      return rend();
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   bool const_device_vector< TYPE >::empty() const {

      return m_size == 0;
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename const_device_vector< TYPE >::size_type
   const_device_vector< TYPE >::size() const {

      return m_size;
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename const_device_vector< TYPE >::size_type
   const_device_vector< TYPE >::max_size() const {

      return size();
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename const_device_vector< TYPE >::size_type
   const_device_vector< TYPE >::capacity() const {

      return size();
   }

} // namespace vecmem
