/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem { namespace details {

   template< typename TYPE >
   jagged_device_vector_iterator< TYPE >::jagged_device_vector_iterator()
   : m_ptr( nullptr ), m_value_is_valid( false ), m_value( data_type() ) {

   }

   template< typename TYPE >
   jagged_device_vector_iterator< TYPE >::
   jagged_device_vector_iterator( data_pointer data )
   : m_ptr( data ), m_value_is_valid( false ), m_value( data_type() ) {

   }

   template< typename TYPE >
   template< typename OTHERTYPE,
             std::enable_if_t<
                details::is_same_nc< TYPE, OTHERTYPE >::value,
                bool > >
   jagged_device_vector_iterator< TYPE >::
   jagged_device_vector_iterator( data::vector_view< OTHERTYPE >* data )
   : m_ptr( data ), m_value_is_valid( false ), m_value( data_type() ) {

   }

   template< typename TYPE >
   jagged_device_vector_iterator< TYPE >::
   jagged_device_vector_iterator( const jagged_device_vector_iterator& parent )
   : m_ptr( parent.m_ptr ), m_value_is_valid( false ), m_value( data_type() ) {

   }

   template< typename TYPE >
   template< typename T >
   jagged_device_vector_iterator< TYPE >::
   jagged_device_vector_iterator(
      const jagged_device_vector_iterator< T >& parent )
   : m_ptr( parent.m_ptr ), m_value_is_valid( false ), m_value( data_type() ) {

   }

   template< typename TYPE >
   jagged_device_vector_iterator< TYPE >&
   jagged_device_vector_iterator< TYPE >::
   operator=( const jagged_device_vector_iterator& rhs ) {

      // Check if anything needs to be done.
      if( this == &rhs ) {
         return *this;
      }

      // Perform the copy.
      m_ptr = rhs.m_ptr;
      m_value_is_valid = false;

      // Return this object.
      return *this;
   }

   template< typename TYPE >
   typename jagged_device_vector_iterator< TYPE >::reference
   jagged_device_vector_iterator< TYPE >::operator*() const {

      ensure_valid();
      return m_value;
   }

   template< typename TYPE >
   typename jagged_device_vector_iterator< TYPE >::pointer
   jagged_device_vector_iterator< TYPE >::operator->() const {

      ensure_valid();
      return &m_value;
   }

   template< typename TYPE >
   jagged_device_vector_iterator< TYPE >&
   jagged_device_vector_iterator< TYPE >::operator++() {

      ++m_ptr;
      m_value_is_valid = false;
      return *this;
   }

   template< typename TYPE >
   jagged_device_vector_iterator< TYPE >
   jagged_device_vector_iterator< TYPE >::operator++( int ) {

      jagged_device_vector_iterator tmp = *this;
      ++m_ptr;
      m_value_is_valid = false;
      return tmp;
   }

   template< typename TYPE >
   jagged_device_vector_iterator< TYPE >&
   jagged_device_vector_iterator< TYPE >::operator--() {

      --m_ptr;
      m_value_is_valid = false;
      return *this;
   }

   template< typename TYPE >
   jagged_device_vector_iterator< TYPE >
   jagged_device_vector_iterator< TYPE >::operator--( int ) {

      jagged_device_vector_iterator tmp = *this;
      --m_ptr;
      m_value_is_valid = false;
      return tmp;
   }

   template< typename TYPE >
   jagged_device_vector_iterator< TYPE >
   jagged_device_vector_iterator< TYPE >::operator+( difference_type n ) const {

      return jagged_device_vector_iterator( m_ptr + n );
   }

   template< typename TYPE >
   jagged_device_vector_iterator< TYPE >&
   jagged_device_vector_iterator< TYPE >::operator+=( difference_type n ) {

      m_ptr += n;
      m_value_is_valid = false;
      return *this;
   }

   template< typename TYPE >
   jagged_device_vector_iterator< TYPE >
   jagged_device_vector_iterator< TYPE >::operator-( difference_type n ) const {

      return jagged_device_vector_iterator( m_ptr - n );
   }

   template< typename TYPE >
   jagged_device_vector_iterator< TYPE >&
   jagged_device_vector_iterator< TYPE >::operator-=( difference_type n ) {

      m_ptr -= n;
      m_value_is_valid = false;
      return *this;
   }

   template< typename TYPE >
   bool jagged_device_vector_iterator< TYPE >::
   operator==( const jagged_device_vector_iterator& other ) const {

      return ( m_ptr == other.m_ptr );
   }

   template< typename TYPE >
   bool jagged_device_vector_iterator< TYPE >::
   operator!=( const jagged_device_vector_iterator& other ) const {

      return !( *this == other );
   }

   template< typename TYPE >
   void jagged_device_vector_iterator< TYPE >::ensure_valid() const {

      if( m_value_is_valid ) {
         return;
      }
      m_value = value_type( *m_ptr );
      m_value_is_valid = true;
      return;
   }

} } // namespace vecmem::details
