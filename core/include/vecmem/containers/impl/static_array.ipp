/**
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include "vecmem/utils/types.hpp"

namespace vecmem {
template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE static_array<T, N>::static_array(void) {
    /*
     * This function does quite literally nothing, and leaves the array's
     * contents uninitialized.
     */
}

template <typename T, std::size_t N>
VECMEM_HOST constexpr typename static_array<T, N>::reference
static_array<T, N>::at(size_type i) {
    /*
     * The at function is bounds-checking in the standard library, so we
     * do a boundary check in our code, too. This makes this method
     * incompatible with device code.
     */
    if (i >= N) {
        throw std::out_of_range("Index greater than size of static array.");
    }

    return operator[](i);
}

template <typename T, std::size_t N>
VECMEM_HOST constexpr typename static_array<T, N>::const_reference
static_array<T, N>::at(size_type i) const {
    /*
     * Same thing as with the other at function, we do a bounds check in
     * accordance with the standard library.
     */
    if (i >= N) {
        throw std::out_of_range("Index greater than size of static array.");
    }

    return operator[](i);
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr typename static_array<T, N>::reference
static_array<T, N>::operator[](size_type i) {
    /*
     * Non-bounds checking access, which could cause a segmentation
     * violation.
     */
    return m_array[i];
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr typename static_array<T, N>::const_reference
static_array<T, N>::operator[](size_type i) const {
    /*
     * Return an element as constant.
     */
    return m_array[i];
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr typename static_array<T, N>::reference
static_array<T, N>::front(void) {
    /*
     * Return the first element.
     */
    return m_array[0];
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr typename static_array<T, N>::const_reference
static_array<T, N>::front(void) const {
    /*
     * Return the first element, but it's const.
     */
    return m_array[0];
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr typename static_array<T, N>::reference
static_array<T, N>::back(void) {
    /*
     * Return the last element.
     */
    return m_array[N - 1];
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr typename static_array<T, N>::const_reference
static_array<T, N>::back(void) const {
    /*
     * Return the last element, but it's const.
     */
    return m_array[N - 1];
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr typename static_array<T, N>::pointer
static_array<T, N>::data(void) {
    /*
     * Return a pointer to the underlying data.
     */
    return m_array;
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr typename static_array<T, N>::const_pointer
static_array<T, N>::data(void) const {
    /*
     * Return a pointer to the underlying data, but the elements are const.
     */
    return m_array;
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr typename static_array<T, N>::iterator
static_array<T, N>::begin() {

    return m_array;
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr typename static_array<T, N>::const_iterator
static_array<T, N>::begin() const {

    return m_array;
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr typename static_array<T, N>::const_iterator
static_array<T, N>::cbegin() const {

    return begin();
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr typename static_array<T, N>::iterator
static_array<T, N>::end() {

    return m_array + N;
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr typename static_array<T, N>::const_iterator
static_array<T, N>::end() const {

    return m_array + N;
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr typename static_array<T, N>::const_iterator
static_array<T, N>::cend() const {

    return end();
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr typename static_array<T, N>::reverse_iterator
static_array<T, N>::rbegin() {

    return reverse_iterator(end());
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr
    typename static_array<T, N>::const_reverse_iterator
    static_array<T, N>::rbegin() const {

    return const_reverse_iterator(end());
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr
    typename static_array<T, N>::const_reverse_iterator
    static_array<T, N>::crbegin() const {

    return rbegin();
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr typename static_array<T, N>::reverse_iterator
static_array<T, N>::rend() {

    return reverse_iterator(begin());
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr
    typename static_array<T, N>::const_reverse_iterator
    static_array<T, N>::rend() const {

    return const_reverse_iterator(begin());
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr
    typename static_array<T, N>::const_reverse_iterator
    static_array<T, N>::crend() const {

    return rend();
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr bool static_array<T, N>::empty() const {

    return N == 0;
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr typename static_array<T, N>::size_type
static_array<T, N>::size() const {

    return N;
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE constexpr typename static_array<T, N>::size_type
static_array<T, N>::max_size() const {

    return N;
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE void static_array<T, N>::fill(const_reference value) {

    for (std::size_t i = 0; i < N; ++i) {
        m_array[i] = value;
    }
}

template <typename T, std::size_t N>
template <typename Tp1, typename... Tp>
VECMEM_HOST_AND_DEVICE void static_array<T, N>::static_array_impl(size_type i,
                                                                  Tp1&& a1,
                                                                  Tp&&... a) {

    m_array[i] = a1;
    static_array_impl(i + 1, std::forward<Tp>(a)...);
}

template <typename T, std::size_t N>
template <typename Tp1>
VECMEM_HOST_AND_DEVICE void static_array<T, N>::static_array_impl(size_type i,
                                                                  Tp1&& a1) {

    m_array[i] = a1;
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE bool operator==(const static_array<T, N>& lhs,
                                       const static_array<T, N>& rhs) {
    /*
     * Iterate over all elements in the arrays, if any of them are unequal
     * between them, the arrays are not equal.
     */
    for (typename static_array<T, N>::size_type i = 0; i < N; ++i) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }

    /*
     * If we have iterated over the entire array without finding a counter-
     * example to the equality, the arrays must be equal.
     */
    return true;
}

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE bool operator!=(const static_array<T, N>& lhs,
                                       const static_array<T, N>& rhs) {
    /*
     * Same thing as before, we check all element pairs, if any of them are
     * unequal, the entire array is unequal. We could also implement this as
     * return !(lhs == rhs).
     */
    for (typename static_array<T, N>::size_type i = 0; i < N; ++i) {
        if (lhs[i] != rhs[i]) {
            return true;
        }
    }

    /*
     * No counter example, so the arrays are not unequal.
     */
    return false;
}
}  // namespace vecmem
