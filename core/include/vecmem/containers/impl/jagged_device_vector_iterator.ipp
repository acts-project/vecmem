/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem {
namespace details {

template <typename TYPE>
VECMEM_HOST_AND_DEVICE jagged_device_vector_iterator<TYPE>::pointer::pointer(
    const data_pointer data)
    : m_vec(*data) {}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto
jagged_device_vector_iterator<TYPE>::pointer::operator->() -> value_type* {

    return &m_vec;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto
jagged_device_vector_iterator<TYPE>::pointer::operator->() const
    -> const value_type* {

    return &m_vec;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE
jagged_device_vector_iterator<TYPE>::jagged_device_vector_iterator()
    : m_ptr(nullptr) {}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE
jagged_device_vector_iterator<TYPE>::jagged_device_vector_iterator(
    data_pointer data)
    : m_ptr(data) {}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE
jagged_device_vector_iterator<TYPE>::jagged_device_vector_iterator(
    const jagged_device_vector_iterator& parent)
    : m_ptr(parent.m_ptr) {}

template <typename TYPE>
template <typename T>
VECMEM_HOST_AND_DEVICE
jagged_device_vector_iterator<TYPE>::jagged_device_vector_iterator(
    const jagged_device_vector_iterator<T>& parent)
    : m_ptr(parent.m_ptr) {}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE jagged_device_vector_iterator<TYPE>&
jagged_device_vector_iterator<TYPE>::operator=(
    const jagged_device_vector_iterator& rhs) {

    // Check if anything needs to be done.
    if (this == &rhs) {
        return *this;
    }

    // Perform the copy.
    m_ptr = rhs.m_ptr;

    // Return this object.
    return *this;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector_iterator<TYPE>::operator*()
    const -> reference {

    return *m_ptr;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto
jagged_device_vector_iterator<TYPE>::operator->() const -> pointer {

    return m_ptr;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto jagged_device_vector_iterator<TYPE>::operator[](
    difference_type n) const -> reference {

    return *(*this + n);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE jagged_device_vector_iterator<TYPE>&
jagged_device_vector_iterator<TYPE>::operator++() {

    ++m_ptr;
    return *this;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE jagged_device_vector_iterator<TYPE>
jagged_device_vector_iterator<TYPE>::operator++(int) {

    jagged_device_vector_iterator tmp = *this;
    ++m_ptr;
    return tmp;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE jagged_device_vector_iterator<TYPE>&
jagged_device_vector_iterator<TYPE>::operator--() {

    --m_ptr;
    return *this;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE jagged_device_vector_iterator<TYPE>
jagged_device_vector_iterator<TYPE>::operator--(int) {

    jagged_device_vector_iterator tmp = *this;
    --m_ptr;
    return tmp;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE jagged_device_vector_iterator<TYPE>
jagged_device_vector_iterator<TYPE>::operator+(difference_type n) const {

    return jagged_device_vector_iterator(m_ptr + n);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE jagged_device_vector_iterator<TYPE>&
jagged_device_vector_iterator<TYPE>::operator+=(difference_type n) {

    m_ptr += n;
    return *this;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE jagged_device_vector_iterator<TYPE>
jagged_device_vector_iterator<TYPE>::operator-(difference_type n) const {

    return jagged_device_vector_iterator(m_ptr - n);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE jagged_device_vector_iterator<TYPE>&
jagged_device_vector_iterator<TYPE>::operator-=(difference_type n) {

    m_ptr -= n;
    return *this;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE bool jagged_device_vector_iterator<TYPE>::operator==(
    const jagged_device_vector_iterator& other) const {

    return (m_ptr == other.m_ptr);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE bool jagged_device_vector_iterator<TYPE>::operator!=(
    const jagged_device_vector_iterator& other) const {

    return !(*this == other);
}

}  // namespace details
}  // namespace vecmem
