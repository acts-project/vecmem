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
    : m_size( parent.m_size ), m_ptr( parent.m_ptr ) {

    }

}} // namespace vemcem::data
