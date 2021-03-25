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
                 ( ! std::is_same< T, OTHERTYPE >::value ) &&
                 std::is_same< T,
                               typename std::add_const< OTHERTYPE >::type >::value,
                 bool > >
    jagged_vector_view< T >::
    jagged_vector_view( const jagged_vector_view< OTHERTYPE >& parent )
    : m_size( parent.m_size ), m_ptr( parent.m_ptr ) {

    }

}} // namespace vemcem::data
