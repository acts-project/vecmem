/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include <endian.h>
#include "vecmem/utils/debug.hpp"

// System include(s).
#include <cassert>
#include <cwctype>
namespace vecmem {

template <typename TYPE>
VECMEM_HOST_AND_DEVICE device_vector<TYPE>::device_vector(
    const data::vector_view<value_type>& data)
    : m_capacity(data.capacity()), m_size(0), m_start_index(reinterpret_cast<size_t>(data.ptr())) {

    // Copy the size of the vector if given
    if (data.size_ptr() != nullptr) {
        m_size = *data.size_ptr();
        is_resizable = true;
    } else {
        m_size = m_capacity;
        is_resizable = false;
    }

    VECMEM_DEBUG_MSG(5,
                     "Created vecmem::device_vector with capacity %u and "
                     "size %u from start index %u",
                     m_capacity, m_size,
                     m_start_index);
}

template <typename TYPE>
template <typename OTHERTYPE,
          std::enable_if_t<details::is_same_nc<TYPE, OTHERTYPE>::value, bool> >
VECMEM_HOST_AND_DEVICE device_vector<TYPE>::device_vector(
    const data::vector_view<OTHERTYPE>& data)
    : m_capacity(data.capacity()), m_size(0), m_start_index(reinterpret_cast<size_t>(data.ptr())) {

    // Copy the size of the vector if given
    if (data.size_ptr() != nullptr) {
        m_size = *data.size_ptr();
        is_resizable = true;
    } else {
        m_size = m_capacity;
        is_resizable = false;
    }
    VECMEM_DEBUG_MSG(5,
                     "Created vecmem::device_vector with capacity %u and "
                     "size %u from start index %u",
                     m_capacity, m_size,
                     m_start_index);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE device_vector<TYPE>::device_vector(
    const device_vector& parent)
    : m_capacity(parent.m_capacity),
      m_size(parent.m_size),
      m_start_index(parent.m_start_index),
      is_resizable(parent.is_resizable) {

    VECMEM_DEBUG_MSG(5,
                     "Created vecmem::device_vector with capacity %u and "
                     "size %u from start index %u",
                     m_capacity, m_size,
                     m_start_index);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE device_vector<TYPE>& device_vector<TYPE>::operator=(
    const device_vector& rhs) {

    // Prevent self-assignment.
    if (this == &rhs) {
        return *this;
    }

    // Copy the other object's payload.
    m_capacity = rhs.m_capacity;
    m_size = rhs.m_size;
    m_start_index = rhs.m_start_index;
    is_resizable = rhs.is_resizable;

    // Return a reference to this object.
    return *this;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::at(size_type pos)
    -> reference {

    // Check if the index is valid.
    assert(pos < size());

    // Return a reference to the vector element.
    pointer m_ptr = get_pointer();
    return m_ptr[pos];
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::at(size_type pos) const
    -> const_reference {

    // Check if the index is valid.
    assert(pos < size());

    // Return a reference to the vector element.
    pointer m_ptr = get_pointer();
    return m_ptr[pos];
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::operator[](size_type pos)
    -> reference {

    // Return a reference to the vector element.
    pointer m_ptr = get_pointer();
    return m_ptr[pos];
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::operator[](size_type pos) const
    -> const_reference {

    // Return a reference to the vector element.
    pointer m_ptr = get_pointer();
    return m_ptr[pos];
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::front() -> reference {

    // Make sure that there is at least one element in the vector.
    assert(size() > 0);

    // Return a reference to the first element of the vector.
    pointer m_ptr = get_pointer();
    return m_ptr[0];
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::front() const
    -> const_reference {

    // Make sure that there is at least one element in the vector.
    assert(size() > 0);

    // Return a reference to the first element of the vector.
    pointer m_ptr = get_pointer();
    return m_ptr[0];
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::back() -> reference {

    // Make sure that there is at least one element in the vector.
    assert(size() > 0);

    // Return a reference to the last element of the vector.
    pointer m_ptr = get_pointer();
    return m_ptr[size() - 1];
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::back() const
    -> const_reference {

    // Make sure that there is at least one element in the vector.
    assert(size() > 0);

    // Return a reference to the last element of the vector.
    pointer m_ptr = get_pointer();
    return m_ptr[size() - 1];
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::data() -> pointer {

    pointer m_ptr = get_pointer();
    return m_ptr;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::data() const -> const_pointer {

    pointer m_ptr = get_pointer();
    return m_ptr;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE void device_vector<TYPE>::assign(size_type count,
                                                        const_reference value) {

    // This can only be done on a sufficiently large, resizable vector.
    assert(is_resizable);
    assert(m_capacity >= count);

    // Remove all previous elements.
    clear();
    // Set the assigned size of the vector.
    m_size = count;

    // Create the required number of identical elements.
    for (size_type i = 0; i < count; ++i) {
        construct(i, value);
    }
}

template <typename TYPE>
template <typename... Args>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::emplace_back(Args&&... args)
    -> reference {

    // This can only be done on a resizable vector.
    assert(is_resizable);

    // Increment the size of the vector at first. So that we would "claim" the
    // index from other threads.
    const size_type index = m_size++;
    assert(index < m_capacity);

    // Instantiate the new value.
    pointer m_ptr = get_pointer();
    new (m_ptr + index) value_type(std::forward<Args>(args)...);

    // Return a reference to the newly created object.
    return m_ptr[index];
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::push_back(
    const_reference value) -> size_type {

    // This can only be done on a resizable vector.
    assert(is_resizable);

    // Increment the size of the vector at first. So that we would "claim" the
    // index from other threads.
    size_type index = m_size++;
    assert(index < m_capacity);

    // Instantiate the new value.
    construct(index, value);

    // Return the index under which the element was inserted:
    return index;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::pop_back() -> size_type {

    // This can only be done on a resizable vector.
    assert(is_resizable);

    // Decrement the size of the vector, and remember this new size.
    const size_type new_size = --m_size;

    // Remove the last element.
    destruct(new_size);

    // Return the vector's new size to the user.
    return new_size;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE void device_vector<TYPE>::clear() {

    // This can only be done on a resizable vector.
    assert(is_resizable);

    // Destruct all of the elements that the vector has "at the moment".
    const size_type current_size = m_size;
    for (size_type i = 0; i < current_size; ++i) {
        destruct(i);
    }

    m_size = 0;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE void device_vector<TYPE>::resize(size_type new_size) {

    resize(new_size, value_type());
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE void device_vector<TYPE>::resize(size_type new_size,
                                                        const_reference value) {

    // This can only be done on a resizable vector.
    assert(is_resizable);

    // Get the current size of the vector.
    const size_type current_size = m_size;

    // Check if anything needs to be done.
    if (new_size == current_size) {
        return;
    }

    // If the new size is smaller than the current size, remove the unwanted
    // elements.
    if (new_size < current_size) {
        for (size_type i = new_size; i < current_size; ++i) {
            destruct(i);
        }
    }
    // If the new size is larger than the current size, insert extra elements.
    else {
        for (size_type i = current_size; i < new_size; ++i) {
            construct(i, value);
        }
    }

    // Set the new size for the vector.
    m_size = new_size;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::begin() -> iterator {

    pointer m_ptr = get_pointer();
    return iterator(m_ptr);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::begin() const
    -> const_iterator {

    pointer m_ptr = get_pointer();
    return const_iterator(m_ptr);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::cbegin() const
    -> const_iterator {

    return begin();
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::end() -> iterator {

    pointer m_ptr = get_pointer();
    return iterator(m_ptr + size());
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::end() const -> const_iterator {

    pointer m_ptr = get_pointer();
    return const_iterator(m_ptr + size());
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::cend() const
    -> const_iterator {

    return end();
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::rbegin() -> reverse_iterator {

    return reverse_iterator(end());
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::rbegin() const
    -> const_reverse_iterator {

    return const_reverse_iterator(end());
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::crbegin() const
    -> const_reverse_iterator {

    return rbegin();
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::rend() -> reverse_iterator {

    return reverse_iterator(begin());
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::rend() const
    -> const_reverse_iterator {

    return const_reverse_iterator(begin());
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::crend() const
    -> const_reverse_iterator {

    return rend();
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE bool device_vector<TYPE>::empty() const {

    return (size() == 0);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::size() const -> size_type {
    return m_size;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::max_size() const -> size_type {

    return capacity();
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto device_vector<TYPE>::capacity() const -> size_type {

    return m_capacity;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE void device_vector<TYPE>::construct(
    size_type pos, const_reference value) {

    // Make sure that the position is available.
    assert(pos < m_capacity);

    // Use the constructor of the type.
    pointer m_ptr = get_pointer();
    new (m_ptr + pos) value_type(value);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE void device_vector<TYPE>::destruct(size_type pos) {

    // Make sure that the position is available.
    assert(pos < m_capacity);

    // Use the destructor of the type.
    pointer m_ptr = get_pointer();
    pointer ptr = m_ptr + pos;
    ptr->~value_type();
}

template <typename TYPE>
typename vecmem::device_vector<TYPE>::pointer vecmem::device_vector<TYPE>::get_pointer() const {
    return reinterpret_cast<pointer>(memory_buffer + m_start_index);
}
}
