/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace vecmem { namespace data {

    template<typename T>
    VECMEM_HOST_AND_DEVICE
    jagged_vector_view<T>::jagged_vector_view(
        size_type size,
        pointer ptr
    ) :
        m_size(size),
        m_ptr(ptr)
    {
    }

    template< typename T >
    template< typename OTHERTYPE,
              std::enable_if_t<
                 details::is_same_nc< T, OTHERTYPE >::value,
                 bool > >
    VECMEM_HOST_AND_DEVICE
    jagged_vector_view< T >::
    jagged_vector_view( const jagged_vector_view< OTHERTYPE >& parent )
    : m_size( parent.m_size ),
      // This looks scarier than it really is. We "just" reinterpret a
      // vecmem::data::vector_view<T> pointer to be seen as
      // vecmem::data::vector_view<const T> instead.
      m_ptr( reinterpret_cast< pointer >( parent.m_ptr ) ) {

    }

}} // namespace vemcem::data
