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
namespace data {

template <typename TYPE>
VECMEM_HOST_AND_DEVICE vector_view<TYPE>::vector_view(size_type size,
                                                      pointer ptr)
    : m_capacity(size), m_size(nullptr), m_ptr(ptr) {}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE vector_view<TYPE>::vector_view(size_type capacity,
                                                      size_pointer size,
                                                      pointer ptr)
    : m_capacity(capacity), m_size(size), m_ptr(ptr) {}

template <typename TYPE>
template <typename OTHERTYPE,
          std::enable_if_t<details::is_same_nc<TYPE, OTHERTYPE>::value, bool> >
VECMEM_HOST_AND_DEVICE vector_view<TYPE>::vector_view(
    const vector_view<OTHERTYPE>& parent)
    : m_capacity(parent.capacity()),
      m_size(parent.size_ptr()),
      m_ptr(parent.ptr()) {}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::size() const -> size_type {

    return (m_size == nullptr ? m_capacity : *m_size);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::capacity() const -> size_type {

    return m_capacity;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::size_ptr() -> size_pointer {

    return m_size;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::size_ptr() const
    -> const_size_pointer {

    return m_size;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::ptr() -> pointer {

    return m_ptr;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::ptr() const -> const_pointer {

    return m_ptr;
}

}  // namespace data
}  // namespace vecmem
