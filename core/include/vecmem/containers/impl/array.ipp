/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

namespace vecmem {

namespace details {

/// Helper function used in the @c vecmem::array constructors
template <typename T, std::size_t N>
std::unique_ptr<typename vecmem::array<T, N>::value_type,
                typename vecmem::array<T, N>::deleter>
allocate_array_memory(vecmem::memory_resource& resource,
                      typename vecmem::array<T, N>::size_type size) {

    return {
        size == 0
            ? nullptr
            : static_cast<typename vecmem::array<T, N>::pointer>(
                  resource.allocate(
                      size * sizeof(typename vecmem::array<T, N>::value_type))),
        {size, resource}};
}

/// Helper function used in the @c vecmem::array constructors
template <typename T, std::size_t N>
auto initialize_array_memory(
    std::unique_ptr<typename vecmem::array<T, N>::value_type,
                    typename vecmem::array<T, N>::deleter>
        memory,
    typename vecmem::array<T, N>::size_type size) {

    typename vecmem::array<T, N>::size_type i = 0;
    for (typename vecmem::array<T, N>::pointer ptr = memory.get(); i < size;
         ++i, ++ptr) {
        new (ptr) typename vecmem::array<T, N>::value_type();
    }
    return memory;
}

}  // namespace details

template <typename T, std::size_t N>
array<T, N>::deleter::deleter(size_type size, memory_resource& resource)
    : m_size(size), m_resource(&resource) {}

template <typename T, std::size_t N>
void array<T, N>::deleter::operator()(void* ptr) {

    // Call the destructor on all objects.
    size_type i = 0;
    for (pointer p = reinterpret_cast<pointer>(ptr); i < m_size; ++i, ++p) {
        p->~value_type();
    }
    // De-allocate the array's memory.
    if ((m_size != 0) && (ptr != nullptr)) {
        m_resource->deallocate(ptr, m_size * sizeof(value_type));
    }
}

template <typename T, std::size_t N>
array<T, N>::array(memory_resource& resource)
    : m_size(N),
      m_memory(details::initialize_array_memory<T, N>(
          details::allocate_array_memory<T, N>(resource, m_size), m_size)) {

    static_assert(N != details::array_invalid_size,
                  "Can only use the 'compile time constructor' if a size "
                  "was provided as a template argument");
}

template <typename T, std::size_t N>
array<T, N>::array(memory_resource& resource, size_type size)
    : m_size(size),
      m_memory(details::initialize_array_memory<T, N>(
          details::allocate_array_memory<T, N>(resource, m_size), m_size)) {

    static_assert(N == details::array_invalid_size,
                  "Can only use the 'runtime constructor' if a size was not "
                  "provided as a template argument");
}

template <typename T, std::size_t N>
auto array<T, N>::at(size_type pos) -> reference {

    if (pos >= m_size) {
        throw std::out_of_range("Requested element " + std::to_string(pos) +
                                " from a " + std::to_string(m_size) +
                                " sized vecmem::array");
    }
    return m_memory.get()[pos];
}

template <typename T, std::size_t N>
auto array<T, N>::at(size_type pos) const -> const_reference {

    if (pos >= m_size) {
        throw std::out_of_range("Requested element " + std::to_string(pos) +
                                " from a " + std::to_string(m_size) +
                                " sized vecmem::array");
    }
    return m_memory.get()[pos];
}

template <typename T, std::size_t N>
auto array<T, N>::operator[](size_type pos) -> reference {

    return m_memory.get()[pos];
}

template <typename T, std::size_t N>
auto array<T, N>::operator[](size_type pos) const -> const_reference {

    return m_memory.get()[pos];
}

template <typename T, std::size_t N>
auto array<T, N>::front() -> reference {

    if (m_size == 0) {
        throw std::out_of_range(
            "Called vecmem::array::front() on an empty "
            "array");
    }
    return (*m_memory);
}

template <typename T, std::size_t N>
auto array<T, N>::front() const -> const_reference {

    if (m_size == 0) {
        throw std::out_of_range(
            "Called vecmem::array::front() on an empty "
            "array");
    }
    return (*m_memory);
}

template <typename T, std::size_t N>
auto array<T, N>::back() -> reference {

    if (m_size == 0) {
        throw std::out_of_range(
            "Called vecmem::array::back() on an empty "
            "array");
    }
    return m_memory.get()[m_size - 1];
}

template <typename T, std::size_t N>
auto array<T, N>::back() const -> const_reference {

    if (m_size == 0) {
        throw std::out_of_range(
            "Called vecmem::array::back() on an empty "
            "array");
    }
    return m_memory.get()[m_size - 1];
}

template <typename T, std::size_t N>
auto array<T, N>::data() -> pointer {

    return m_memory.get();
}

template <typename T, std::size_t N>
auto array<T, N>::data() const -> const_pointer {

    return m_memory.get();
}

template <typename T, std::size_t N>
auto array<T, N>::begin() -> iterator {

    return m_memory.get();
}

template <typename T, std::size_t N>
auto array<T, N>::begin() const -> const_iterator {

    return m_memory.get();
}

template <typename T, std::size_t N>
auto array<T, N>::cbegin() const -> const_iterator {

    return m_memory.get();
}

template <typename T, std::size_t N>
auto array<T, N>::end() -> iterator {

    return (m_memory.get() + m_size);
}

template <typename T, std::size_t N>
auto array<T, N>::end() const -> const_iterator {

    return (m_memory.get() + m_size);
}

template <typename T, std::size_t N>
auto array<T, N>::cend() const -> const_iterator {

    return (m_memory.get() + m_size);
}

template <typename T, std::size_t N>
auto array<T, N>::rbegin() -> reverse_iterator {

    return reverse_iterator(end());
}

template <typename T, std::size_t N>
auto array<T, N>::rbegin() const -> const_reverse_iterator {

    return const_reverse_iterator(end());
}

template <typename T, std::size_t N>
auto array<T, N>::crbegin() const -> const_reverse_iterator {

    return const_reverse_iterator(end());
}

template <typename T, std::size_t N>
auto array<T, N>::rend() -> reverse_iterator {

    return reverse_iterator(begin());
}

template <typename T, std::size_t N>
auto array<T, N>::rend() const -> const_reverse_iterator {

    return const_reverse_iterator(begin());
}

template <typename T, std::size_t N>
auto array<T, N>::crend() const -> const_reverse_iterator {

    return const_reverse_iterator(begin());
}

template <typename T, std::size_t N>
bool array<T, N>::empty() const noexcept {

    return (m_size == 0);
}

template <typename T, std::size_t N>
auto array<T, N>::size() const noexcept -> size_type {

    return m_size;
}

template <typename T, std::size_t N>
void array<T, N>::fill(const_reference value) {

    std::fill(begin(), end(), value);
}

template <typename T, std::size_t N>
VECMEM_HOST data::vector_view<T> get_data(array<T, N>& a) {

    return {static_cast<typename data::vector_view<T>::size_type>(a.size()),
            a.data()};
}

template <typename T, std::size_t N>
VECMEM_HOST data::vector_view<const T> get_data(const array<T, N>& a) {

    return {
        static_cast<typename data::vector_view<const T>::size_type>(a.size()),
        a.data()};
}

}  // namespace vecmem
