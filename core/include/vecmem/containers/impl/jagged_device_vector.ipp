/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <cassert>

namespace vecmem {

    template<typename T>
    VECMEM_HOST_AND_DEVICE
    jagged_device_vector<T>::jagged_device_vector(
        const data::jagged_vector_view<T> & data
    ) :
        m_size(data.m_size),
        m_ptr(data.m_ptr)
    {
    }

    template< typename T >
    template< typename OTHERTYPE,
              std::enable_if_t<
                  details::is_same_nc< T, OTHERTYPE >::value,
                  bool > >
    VECMEM_HOST_AND_DEVICE
    jagged_device_vector< T >::
    jagged_device_vector( const data::jagged_vector_view< OTHERTYPE >& data )
    : m_size( data.m_size ),
      // This looks scarier than it really is. We "just" reinterpret a
      // vecmem::data::vector_view<T> pointer to be seen as
      // vecmem::data::vector_view<const T> instead.
      m_ptr( reinterpret_cast<
          typename data::jagged_vector_view< T >::pointer >( data.m_ptr ) ) {

    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    jagged_device_vector< T >::
    jagged_device_vector( const jagged_device_vector& parent )
    : m_size( parent.m_size ), m_ptr( parent.m_ptr ) {

    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    jagged_device_vector< T >&
    jagged_device_vector< T >::operator=( const jagged_device_vector& rhs ) {

        // Check if anything needs to be done.
        if( this == &rhs ) {
            return *this;
        }

        // Make this object point at the same data in memory as the one we're
        // copying from.
        m_size = rhs.m_size;
        m_ptr = rhs.m_ptr;

        // Return this object.
        return *this;
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::reference
    jagged_device_vector< T >::at( size_type pos ) {

        // Check if the index is valid.
        assert( pos < m_size );

        // Return a reference to the vector element.
        return m_ptr[ pos ];
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::const_reference
    jagged_device_vector< T >::at( size_type pos ) const {

        // Check if the index is valid.
        assert( pos < m_size );

        // Return a reference to the vector element.
        return m_ptr[ pos ];
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::reference
    jagged_device_vector< T >::operator[]( size_type pos ) {

        // Return a reference to the vector element.
        return m_ptr[ pos ];
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::const_reference
    jagged_device_vector< T >::operator[]( size_type pos ) const {

        // Return a reference to the vector element.
        return m_ptr[ pos ];
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::reference
    jagged_device_vector< T >::front() {

        // Make sure that there is at least one element in the outer vector.
        assert( m_size > 0 );

        // Return a reference to the first element of the vector.
        return m_ptr[ 0 ];
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::const_reference
    jagged_device_vector< T >::front() const {

        // Make sure that there is at least one element in the outer vector.
        assert( m_size > 0 );

        // Return a reference to the first element of the vector.
        return m_ptr[ 0 ];
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::reference
    jagged_device_vector< T >::back() {

        // Make sure that there is at least one element in the outer vector.
        assert( m_size > 0 );

        // Return a reference to the last element of the vector.
        return m_ptr[ m_size - 1 ];
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::const_reference
    jagged_device_vector< T >::back() const {

        // Make sure that there is at least one element in the outer vector.
        assert( m_size > 0 );

        // Return a reference to the last element of the vector.
        return m_ptr[ m_size - 1 ];
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::iterator
    jagged_device_vector< T >::begin() {

        return iterator( m_ptr );
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::const_iterator
    jagged_device_vector< T >::begin() const {

        return const_iterator( m_ptr );
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::const_iterator
    jagged_device_vector< T >::cbegin() const {

        return begin();
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::iterator
    jagged_device_vector< T >::end() {

        return iterator( m_ptr + m_size );
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::const_iterator
    jagged_device_vector< T >::end() const {

        return const_iterator( m_ptr + m_size );
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::const_iterator
    jagged_device_vector< T >::cend() const {

        return end();
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::reverse_iterator
    jagged_device_vector< T >::rbegin() {

        return reverse_iterator( end() );
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::const_reverse_iterator
    jagged_device_vector< T >::rbegin() const {

        return const_reverse_iterator( end() );
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::const_reverse_iterator
    jagged_device_vector< T >::crbegin() const {

        return rbegin();
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::reverse_iterator
    jagged_device_vector< T >::rend() {

        return reverse_iterator( begin() );
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::const_reverse_iterator
    jagged_device_vector< T >::rend() const {

        return const_reverse_iterator( begin() );
    }

    template< typename T >
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector< T >::const_reverse_iterator
    jagged_device_vector< T >::crend() const {

        return rend();
    }

    template<typename T>
    VECMEM_HOST_AND_DEVICE
    bool jagged_device_vector<T>::empty(
        void
    ) const {
        return m_size == 0;
    }

    template<typename T>
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector<T>::size_type
    jagged_device_vector<T>::size(
        void
    ) const {
        return m_size;
    }

    template<typename T>
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector<T>::size_type
    jagged_device_vector<T>::max_size() const {

        return m_size;
    }

    template<typename T>
    VECMEM_HOST_AND_DEVICE
    typename jagged_device_vector<T>::size_type
    jagged_device_vector<T>::capacity() const {

        return m_size;
    }

} // namespace vecmem
