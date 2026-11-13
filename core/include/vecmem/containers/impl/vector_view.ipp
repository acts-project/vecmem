/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
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
          std::enable_if_t<details::is_same_nc<TYPE, OTHERTYPE>::value, bool>>
VECMEM_HOST_AND_DEVICE vector_view<TYPE>::vector_view(
    const vector_view<OTHERTYPE>& parent)
    : m_capacity(parent.capacity()),
      m_size(parent.size_ptr()),
      m_ptr(parent.ptr()) {}

template <typename TYPE>
template <typename OTHERTYPE,
          std::enable_if_t<details::is_same_nc<TYPE, OTHERTYPE>::value, bool>>
VECMEM_HOST_AND_DEVICE vector_view<TYPE>& vector_view<TYPE>::operator=(
    const vector_view<OTHERTYPE>& rhs) {

    // Self-assignment is not dangerous for this type. But putting in
    // extra checks into the code would not be great.
    m_capacity = rhs.capacity();
    m_size = rhs.size_ptr();
    m_ptr = rhs.ptr();

    // Return this (updated) object.
    return *this;
}

template <typename TYPE>
template <typename OTHERTYPE,
          std::enable_if_t<std::is_same<std::remove_cv_t<TYPE>,
                                        std::remove_cv_t<OTHERTYPE>>::value,
                           bool>>
VECMEM_HOST_AND_DEVICE bool vector_view<TYPE>::operator==(
    const vector_view<OTHERTYPE>& rhs) const {

    return ((m_capacity == rhs.capacity()) && (m_size == rhs.size_ptr()) &&
            (static_cast<const void*>(m_ptr) ==
             static_cast<const void*>(rhs.ptr())));
}

template <typename TYPE>
template <typename OTHERTYPE,
          std::enable_if_t<std::is_same<std::remove_cv_t<TYPE>,
                                        std::remove_cv_t<OTHERTYPE>>::value,
                           bool>>
VECMEM_HOST_AND_DEVICE bool vector_view<TYPE>::operator!=(
    const vector_view<OTHERTYPE>& rhs) const {

    return !(*this == rhs);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::at(size_type pos) -> reference {

    // Check if the index is valid.
    assert(pos < size());

    // Return a reference to the vector element.
    return m_ptr[pos];
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::at(size_type pos) const
    -> const_reference {

    // Check if the index is valid.
    assert(pos < size());

    // Return a reference to the vector element.
    return m_ptr[pos];
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::operator[](size_type pos)
    -> reference {

    // Return a reference to the vector element.
    return m_ptr[pos];
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::operator[](size_type pos) const
    -> const_reference {

    // Return a reference to the vector element.
    return m_ptr[pos];
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::front() -> reference {

    // Make sure that there is at least one element in the vector.
    assert(size() > 0);

    // Return a reference to the first element of the vector.
    return m_ptr[0];
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::front() const
    -> const_reference {

    // Make sure that there is at least one element in the vector.
    assert(size() > 0);

    // Return a reference to the first element of the vector.
    return m_ptr[0];
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::back() -> reference {

    // Make sure that there is at least one element in the vector.
    assert(size() > 0);

    // Return a reference to the last element of the vector.
    return m_ptr[size() - 1];
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::back() const -> const_reference {

    // Make sure that there is at least one element in the vector.
    assert(size() > 0);

    // Return a reference to the last element of the vector.
    return m_ptr[size() - 1];
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::data() -> pointer {

    return m_ptr;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::data() const -> const_pointer {

    return m_ptr;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE void vector_view<TYPE>::assign(size_type count,
                                                      const_reference value) {

    // This can only be done on a sufficiently large, resizable vector.
    assert(m_size != nullptr);
    assert(m_capacity >= count);

    // Remove all previous elements.
    clear();
    // Set the assigned size of the vector.
    device_atomic_ref<size_type> asize(*m_size);
    asize.store(count);

    // Create the required number of identical elements.
    for (size_type i = 0; i < count; ++i) {
        construct(i, value);
    }
}

template <typename TYPE>
template <typename OTHERTYPE,
          std::enable_if_t<std::is_convertible<OTHERTYPE, TYPE>::value, bool>>
VECMEM_HOST_AND_DEVICE void vector_view<TYPE>::assign(
    const vector_view<OTHERTYPE>& rhs) {

    // Prevent self-assignment.
    if (this == &rhs) {
        return;
    }

    // Copy the other object's payload.
    resize(rhs.size());
    for (size_type i = 0; i < size(); ++i) {
        this->at(i) = rhs.at(i);
    }
}

template <typename TYPE>
template <typename... Args>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::emplace_back(Args&&... args)
    -> reference {

    // This can only be done on a resizable vector.
    assert(m_size != nullptr);

    // Increment the size of the vector at first. So that we would "claim" the
    // index from other threads.
    device_atomic_ref<size_type> asize(*m_size);
    const size_type index = asize.fetch_add(1u);
    assert(index < m_capacity);

    // Instantiate the new value.
    new (m_ptr + index) value_type(std::forward<Args>(args)...);

    // Return a reference to the newly created object.
    return m_ptr[index];
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::push_back(const_reference value)
    -> size_type {

    // This can only be done on a resizable vector.
    assert(m_size != nullptr);

    // Increment the size of the vector at first. So that we would "claim" the
    // index from other threads.
    device_atomic_ref<size_type> asize(*m_size);
    const size_type index = asize.fetch_add(1u);
    assert(index < m_capacity);

    // Instantiate the new value.
    construct(index, value);

    // Return the index under which the element was inserted:
    return index;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::bulk_append(size_type n)
    -> size_type {
    // This can only be done on a resizable vector.
    assert(m_size != nullptr);

    static_assert(std::is_default_constructible<TYPE>::value,
                  "Type `T` in `vector_view<T>::bulk_append` is not a "
                  "default-constructible type.");

    device_atomic_ref<size_type> asize(*m_size);
    const size_type index = asize.fetch_add(n);
    assert((index + n) <= m_capacity);

    for (size_type i = 0; i < n; ++i) {
        construct(index + i, value_type());
    }

    return index;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::bulk_append(size_type n,
                                                           const_reference v)
    -> size_type {
    // This can only be done on a resizable vector.
    assert(m_size != nullptr);

    static_assert(std::is_copy_constructible<TYPE>::value,
                  "Type `TYPE` in vector_view<T>::bulk_append(SIZE, TYPE)` "
                  "must be copy-constructible.");

    device_atomic_ref<size_type> asize(*m_size);
    const size_type index = asize.fetch_add(n);
    assert((index + n) <= m_capacity);

    for (size_type i = 0; i < n; ++i) {
        construct(index + i, value_type(v));
    }

    return index;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::bulk_append_implicit(size_type n)
    -> size_type {
    // This can only be done on a resizable vector.
    assert(m_size != nullptr);

    static_assert(details::is_implicit_lifetime<TYPE>::value,
                  "Type `TYPE` in `vector_view<T>::bulk_append_implicit` is "
                  "not an implicit lifetype type, so slots cannot be safely "
                  "reserved. Note that the definition of implicit lifetimes "
                  "differs between C++<=17, C++20, and C++>=23.");

    device_atomic_ref<size_type> asize(*m_size);
    const size_type index = asize.fetch_add(n);
    assert((index + n) <= m_capacity);

#if defined(__cpp_lib_start_lifetime_as) && \
    __cpp_lib_start_lifetime_as >= 202207L
    std::start_lifetime_as_array<TYPE>(m_ptr + index, n);
#endif

    return index;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::bulk_append_implicit_unsafe(
    size_type n) -> size_type {
    // This can only be done on a resizable vector.
    assert(m_size != nullptr);

    device_atomic_ref<size_type> asize(*m_size);
    const size_type index = asize.fetch_add(n);
    assert((index + n) <= m_capacity);

    return index;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::pop_back() -> size_type {

    // This can only be done on a resizable vector.
    assert(m_size != nullptr);

    // Decrement the size of the vector, and remember this new size.
    device_atomic_ref<size_type> asize(*m_size);
    const size_type new_size = asize.fetch_sub(1u) - 1;

    // Remove the last element.
    destruct(new_size);

    // Return the vector's new size to the user.
    return new_size;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE void vector_view<TYPE>::clear() {

    // This can only be done on a resizable vector.
    assert(m_size != nullptr);

    // Destruct all of the elements that the vector has "at the moment".
    device_atomic_ref<size_type> asize(*m_size);
    const size_type current_size = asize.load();
    for (size_type i = 0; i < current_size; ++i) {
        destruct(i);
    }

    // Set the vector to be empty now.
    asize.store(0);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE void vector_view<TYPE>::resize(size_type new_size) {

    resize(new_size, value_type());
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE void vector_view<TYPE>::resize(size_type new_size,
                                                      const_reference value) {

    // This can only be done on a resizable vector.
    assert(m_size != nullptr);

    // Get the current size of the vector.
    device_atomic_ref<size_type> asize(*m_size);
    const size_type current_size = asize.load();

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
    asize.store(new_size);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE void vector_view<TYPE>::resize_implicit(
    size_type new_size) {
    // This can only be done on a resizable vector.
    assert(m_size != nullptr);

    static_assert(details::is_implicit_lifetime<TYPE>::value,
                  "Type `TYPE` in `vector_view<T>::resize_implicit` is not "
                  "an implicit lifetype type, so slots cannot be safely "
                  "reserved. Note that the definition of implicit lifetimes "
                  "differs between C++<=17, C++20, and C++>=23.");

    device_atomic_ref<size_type> asize(*m_size);
    assert(new_size <= m_capacity);
    asize.store(new_size);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE void vector_view<TYPE>::resize_implicit_unsafe(
    size_type new_size) {
    // This can only be done on a resizable vector.
    assert(m_size != nullptr);

    device_atomic_ref<size_type> asize(*m_size);
    assert(new_size <= m_capacity);
    asize.store(new_size);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::begin() -> iterator {

    return iterator(m_ptr);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::begin() const -> const_iterator {

    return const_iterator(m_ptr);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::cbegin() const
    -> const_iterator {

    return begin();
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::end() -> iterator {

    return iterator(m_ptr + size());
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::end() const -> const_iterator {

    return const_iterator(m_ptr + size());
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::cend() const -> const_iterator {

    return end();
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::rbegin() -> reverse_iterator {

    return reverse_iterator(end());
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::rbegin() const
    -> const_reverse_iterator {

    return const_reverse_iterator(end());
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::crbegin() const
    -> const_reverse_iterator {

    return rbegin();
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::rend() -> reverse_iterator {

    return reverse_iterator(begin());
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::rend() const
    -> const_reverse_iterator {

    return const_reverse_iterator(begin());
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::crend() const
    -> const_reverse_iterator {

    return rend();
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE bool vector_view<TYPE>::empty() const {

    return (size() == 0);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::size() const -> size_type {

    if (m_size == nullptr) {
        return m_capacity;
    } else {
        // For the host, CUDA and HIP it would be possible to use the type
        // atomic<const size_type> here, and avoid any const-casting. But with
        // SYCL we must pass a non-const pointer to the sycl::atomic object
        // that performs the load operation. And for that we need a non-const
        // pointer...
        device_atomic_ref<size_type> asize(*(const_cast<size_type*>(m_size)));
        return asize.load();
    }
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::max_size() const -> size_type {

    return capacity();
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::capacity() const -> size_type {

    return m_capacity;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::size_ptr() const
    -> size_pointer {

    return m_size;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE auto vector_view<TYPE>::ptr() const -> pointer {

    return m_ptr;
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE void vector_view<TYPE>::construct(
    size_type pos, const_reference value) {

    // Make sure that the position is available.
    assert(pos < m_capacity);

    // Use the constructor of the type.
    new (m_ptr + pos) value_type(value);
}

template <typename TYPE>
VECMEM_HOST_AND_DEVICE void vector_view<TYPE>::destruct(size_type pos) {

    // Make sure that the position is available.
    assert(pos < m_capacity);

    // Use the destructor of the type.
    pointer ptr = m_ptr + pos;
    ptr->~value_type();
}

}  // namespace data
}  // namespace vecmem
