/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace vecmem { namespace details {
    template<typename T>
    jagged_vector_view<T>::jagged_vector_view(
        std::size_t size,
        vector_view<T> * ptr
    ) :
        m_size(size),
        m_ptr(ptr)
    {
    }
}}
