/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

namespace vecmem {
namespace data {

template <typename T>
VECMEM_HOST_AND_DEVICE jagged_vector_view<T>::jagged_vector_view(
    size_type size, pointer ptr, pointer host_ptr)
    : m_size(size),
      m_ptr(ptr),
      m_host_ptr(host_ptr != nullptr ? host_ptr : ptr) {}

template <typename T>
template <typename OTHERTYPE,
          std::enable_if_t<details::is_same_nc<T, OTHERTYPE>::value, bool> >
VECMEM_HOST_AND_DEVICE jagged_vector_view<T>::jagged_vector_view(
    jagged_vector_view<OTHERTYPE> parent)
    : m_size(parent.size()),
      // This looks scarier than it really is. We "just" reinterpret a
      // vecmem::data::vector_view<T> pointer to be seen as
      // vecmem::data::vector_view<const T> instead.
      m_ptr(reinterpret_cast<pointer>(parent.ptr())),
      m_host_ptr(reinterpret_cast<pointer>(parent.host_ptr())) {}

template <typename T>
template <typename OTHERTYPE,
          std::enable_if_t<details::is_same_nc<T, OTHERTYPE>::value, bool> >
VECMEM_HOST_AND_DEVICE jagged_vector_view<T>& jagged_vector_view<T>::operator=(
    jagged_vector_view<OTHERTYPE> rhs) {

    // Avoid self-assignment.
    if (this == &rhs) {
        return *this;
    }

    // Perform the assignment.
    m_size = rhs.m_size;
    m_ptr = reinterpret_cast<pointer>(rhs.m_ptr);
    m_host_ptr = reinterpret_cast<pointer>(rhs.m_host_ptr);
}

template <typename T>
VECMEM_HOST_AND_DEVICE typename jagged_vector_view<T>::size_type
jagged_vector_view<T>::size() const {

    return m_size;
}

template <typename T>
VECMEM_HOST_AND_DEVICE typename jagged_vector_view<T>::size_type
jagged_vector_view<T>::capacity() const {

    return m_size;
}

template <typename T>
VECMEM_HOST_AND_DEVICE typename jagged_vector_view<T>::pointer
jagged_vector_view<T>::ptr() {

    return m_ptr;
}

template <typename T>
VECMEM_HOST_AND_DEVICE typename jagged_vector_view<T>::const_pointer
jagged_vector_view<T>::ptr() const {

    return m_ptr;
}

template <typename T>
VECMEM_HOST_AND_DEVICE typename jagged_vector_view<T>::pointer
jagged_vector_view<T>::host_ptr() {

    return m_host_ptr;
}

template <typename T>
VECMEM_HOST_AND_DEVICE typename jagged_vector_view<T>::const_pointer
jagged_vector_view<T>::host_ptr() const {

    return m_host_ptr;
}

}  // namespace data
}  // namespace vecmem
