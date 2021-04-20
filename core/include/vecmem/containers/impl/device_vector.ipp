/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/utils/debug.hpp"

// System include(s).
#include <cassert>

namespace vecmem {

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   device_vector< TYPE >::
   device_vector( const data::vector_view< value_type >& data )
   : m_capacity( data.capacity() ), m_size( data.size_ptr() ),
     m_ptr( data.ptr() ) {

      VECMEM_DEBUG_MSG( 5, "Created vecmem::device_vector with capacity %u and "
                        "size pointer %p from pointer %p", m_capacity,
                        static_cast< const void* >( m_size ),
                        static_cast< const void* >( m_ptr ) );
   }

   template< typename TYPE >
   template< typename OTHERTYPE,
             std::enable_if_t<
                details::is_same_nc< TYPE, OTHERTYPE >::value,
                bool > >
   VECMEM_HOST_AND_DEVICE
   device_vector< TYPE >::
   device_vector( const data::vector_view< OTHERTYPE >& data )
   : m_capacity( data.capacity() ), m_size( data.size_ptr() ),
     m_ptr( data.ptr() ) {

      VECMEM_DEBUG_MSG( 5, "Created vecmem::device_vector with capacity %u and "
                        "size pointer %p from pointer %p", m_capacity,
                        static_cast< const void* >( m_size ),
                        static_cast< const void* >( m_ptr ) );
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   device_vector< TYPE >::device_vector( const device_vector& parent )
   : m_capacity( parent.m_capacity ), m_size( parent.m_size ),
     m_ptr( parent.m_ptr ) {

      VECMEM_DEBUG_MSG( 5, "Created vecmem::device_vector with capacity %u and "
                        "size pointer %p from pointer %p", m_capacity,
                        static_cast< const void* >( m_size ),
                        static_cast< const void* >( m_ptr ) );
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
      m_capacity = rhs.m_capacity;
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
      assert( pos < size() );

      // Return a reference to the vector element.
      return m_ptr[ pos ];
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::const_reference
   device_vector< TYPE >::at( size_type pos ) const {

      // Check if the index is valid.
      assert( pos < size() );

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
      assert( size() > 0 );

      // Return a reference to the first element of the vector.
      return m_ptr[ 0 ];
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::const_reference
   device_vector< TYPE >::front() const {

      // Make sure that there is at least one element in the vector.
      assert( size() > 0 );

      // Return a reference to the first element of the vector.
      return m_ptr[ 0 ];
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::reference
   device_vector< TYPE >::back() {

      // Make sure that there is at least one element in the vector.
      assert( size() > 0 );

      // Return a reference to the last element of the vector.
      return m_ptr[ size() - 1 ];
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::const_reference
   device_vector< TYPE >::back() const {

      // Make sure that there is at least one element in the vector.
      assert( size() > 0 );

      // Return a reference to the last element of the vector.
      return m_ptr[ size() - 1 ];
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
   void device_vector< TYPE >::
   assign( size_type count, const_reference value ) {

      // This can only be done on a sufficiently large, resizable vector.
      assert( m_size != nullptr );
      assert( m_capacity >= count );

      // Remove all previous elements.
      clear();
      // Set the assigned size of the vector.
      atomic< size_type > asize( m_size );
      asize.store( count );

      // Create the required number of identical elements.
      for( size_type i = 0; i < count; ++i ) {
         construct( i, value );
      }
   }

   template< typename TYPE >
   template< typename... Args >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::reference
   device_vector< TYPE >::emplace_back( Args&&... args ) {

      // This can only be done on a resizable vector.
      assert( m_size != nullptr );

      // Increment the size of the vector at first. So that we would "claim" the
      // index from other threads.
      atomic< size_type > asize( m_size );
      const size_type index = asize.fetch_add( 1 );
      assert( index < m_capacity );

      // Instantiate the new value.
      new( m_ptr + index ) value_type( std::forward< Args >( args )... );

      // Return a reference to the newly created object.
      return m_ptr[ index ];
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::size_type
   device_vector< TYPE >::push_back( const_reference value ) {

      // This can only be done on a resizable vector.
      assert( m_size != nullptr );

      // Increment the size of the vector at first. So that we would "claim" the
      // index from other threads.
      atomic< size_type > asize( m_size );
      const size_type index = asize.fetch_add( 1 );
      assert( index < m_capacity );

      // Instantiate the new value.
      construct( index, value );

      // Return the index under which the element was inserted:
      return index;
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::size_type
   device_vector< TYPE >::pop_back() {

      // This can only be done on a resizable vector.
      assert( m_size != nullptr );

      // Decrement the size of the vector, and remember this new size.
      atomic< size_type > asize( m_size );
      const size_type new_size = asize.fetch_sub( 1 ) - 1;

      // Remove the last element.
      destruct( new_size );

      // Return the vector's new size to the user.
      return new_size;
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   void device_vector< TYPE >::clear() {

      // This can only be done on a resizable vector.
      assert( m_size != nullptr );

      // Destruct all of the elements that the vector has "at the moment".
      atomic< size_type > asize( m_size );
      const size_type current_size = asize.load();
      for( size_type i = 0; i < current_size; ++i ) {
         destruct( i );
      }

      // Set the vector to be empty now.
      asize.store( 0 );
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   void device_vector< TYPE >::resize( size_type new_size ) {

      resize( new_size, value_type() );
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   void device_vector< TYPE >::resize( size_type new_size,
                                       const_reference value ) {

      // This can only be done on a resizable vector.
      assert( m_size != nullptr );

      // Get the current size of the vector.
      atomic< size_type > asize( m_size );
      const size_type current_size = asize.load();

      // Check if anything needs to be done.
      if( new_size == current_size ) {
         return;
      }

      // If the new size is smaller than the current size, remove the unwanted
      // elements.
      if( new_size < current_size ) {
         for( size_type i = new_size; i < current_size; ++i ) {
            destruct( i );
         }
      }
      // If the new size is larger than the current size, insert extra elements.
      else {
         for( size_type i = current_size; i < new_size; ++i ) {
            construct( i, value );
         }
      }

      // Set the new size for the vector.
      asize.store( new_size );
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

      return iterator( m_ptr + size() );
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::const_iterator
   device_vector< TYPE >::end() const {

      return const_iterator( m_ptr + size() );
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

      return ( size() == 0 );
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::size_type
   device_vector< TYPE >::size() const {

      if( m_size == nullptr ) {
         return m_capacity;
      } else {
         // For the host, CUDA and HIP it would be possible to use the type
         // atomic<const size_type> here, and avoid any const-casting. But with
         // SYCL we must pass a non-const pointer to the sycl::atomic object
         // that performs the load operation. And for that we need a non-const
         // pointer...
         atomic< size_type > asize( const_cast< size_type* >( m_size ) );
         return asize.load();
      }
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::size_type
   device_vector< TYPE >::max_size() const {

      return capacity();
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   typename device_vector< TYPE >::size_type
   device_vector< TYPE >::capacity() const {

      return m_capacity;
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   void device_vector< TYPE >::
   construct( size_type pos, const_reference value ) {

      // Make sure that the position is available.
      assert( pos < m_capacity );

      // Use the constructor of the type.
      new( m_ptr + pos ) value_type( value );
   }

   template< typename TYPE >
   VECMEM_HOST_AND_DEVICE
   void device_vector< TYPE >::destruct( size_type pos ) {

      // Make sure that the position is available.
      assert( pos < m_capacity );

      // Use the destructor of the type.
      pointer ptr = m_ptr + pos;
      ptr->~value_type();
   }

} // namespace vecmem
