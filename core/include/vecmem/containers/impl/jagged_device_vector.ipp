/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cassert>

namespace vecmem {
    template<typename T>
    jagged_device_vector<T>::jagged_device_vector(
        const details::jagged_vector_view<T> & data
    ) :
        m_size(data.m_size),
        m_ptr(data.m_ptr)
    {
    }

    template<typename T>
    bool jagged_device_vector<T>::empty(
        void
    ) const {
        return size() == 0;
    }

    template<typename T>
    std::size_t jagged_device_vector<T>::size(
        void
    ) const {
        return m_size;
    }

    template<typename T>
    device_vector<T> jagged_device_vector<T>::at(
        std::size_t i
    ) {
        assert(i < size());

        return device_vector<T>(m_ptr[i]);
    }

    template<typename T>
    const device_vector<T> jagged_device_vector<T>::at(
        std::size_t i
    ) const {
        assert(i < size());

        return device_vector<T>(m_ptr[i]);
    }

    template<typename T>
    device_vector<T> jagged_device_vector<T>::operator[](
        std::size_t i
    ) {
        return device_vector<T>(m_ptr[i]);
    }

    template<typename T>
    const device_vector<T> jagged_device_vector<T>::operator[](
        std::size_t i
    ) const {
        return device_vector<T>(m_ptr[i]);
    }

    template<typename T>
    T & jagged_device_vector<T>::at(
        std::size_t i,
        std::size_t j
    ) {
        return at(i).at(j);
    }

    template<typename T>
    const T & jagged_device_vector<T>::at(
        std::size_t i,
        std::size_t j
    ) const {
        return at(i).at(j);
    }
}
