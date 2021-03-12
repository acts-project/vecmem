/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/details/vector_view.hpp"

#include <cstddef>

namespace vecmem {
    namespace details {
        template<typename T>
        struct jagged_vector_view;
    }

    /**
     * @brief A view for jagged vectors.
     *
     * A jagged vector is a two-dimensional vector in which the inner vectors do
     * not necessarily have the same size. For example, a jagged vector might
     * look like this:
     *
     * [[0, 1, 2],
     *  [3, 4],
     *  [],
     *  [5, 6, 7]]
     *
     * This class is a view of existing two-dimensional vectors created using a
     * vector-of-vectors formalism. Elements cannot be added or removed through
     * this view, but individual elements can be accessed and modified.
     *
     * @warning This view class shares memory with the vectors from which it was
     * constructed. Operating on the underlying vectors while an instance of
     * this class exists deriving from it is undefined and may leave the view in
     * an undefined state.
     */
    template<typename T>
    class jagged_device_vector {
    public:
        /**
         * @brief Construct a jagged vector view from a jagged vector data
         * object.
         */
        VECMEM_HOST_AND_DEVICE
        jagged_device_vector(
            const details::jagged_vector_view<T> & data
        );

        /**
         * @brief Checks whether this view has no rows.
         *
         * Returns true if the jagged vector is empty, and false otherwise.
         *
         * @note A jagged vector of shape [[]] (that is to say, an empty row) is
         * not considered empty, but a jagged vector of shape [] is.
         */
        VECMEM_HOST_AND_DEVICE
        bool empty(
            void
        ) const;

        /**
         * @brief Get the number of rows in this view.
         */
        VECMEM_HOST_AND_DEVICE
        std::size_t size(
            void
        ) const;

        /**
         * @brief Get the row (vector) at a certain index.
         *
         * This method will assert that the element is in range if NDEBUG is
         * unset.
         *
         * @param[in] i The row number to fetch.
         */
        VECMEM_HOST_AND_DEVICE
        device_vector<T> at(
            std::size_t i
        );

        /**
         * @brief Get the row (vector) at a certain index in a const way.
         *
         * This method will assert that the element is in range if NDEBUG is
         * unset.
         *
         * @param[in] i The row number to fetch.
         */
        VECMEM_HOST_AND_DEVICE
        const device_vector<T> at(
            std::size_t i
        ) const;

        /**
         * @brief Get the row (vector) at a certain index.
         *
         * This method does no bounds checking at all.
         *
         * @param[in] i The row number to fetch.
         */
        VECMEM_HOST_AND_DEVICE
        device_vector<T> operator[](
            std::size_t i
        );

        /**
         * @brief Get the row (vector) at a certain index in a const way.
         *
         * This method does no bounds checking at all.
         *
         * @param[in] i The row number to fetch.
         */
        VECMEM_HOST_AND_DEVICE
        const device_vector<T> operator[](
            std::size_t i
        ) const;

        /**
         * @brief Retrieve the element at a given two-dimensional coordinate.
         *
         * This method can be used to directly retrieve an object from an inner
         * vector.
         *
         * @param[in] i The row index to fetch from.
         * @param[in] j The index in row i to use.
         */
        VECMEM_HOST_AND_DEVICE
        T & at(
            std::size_t i,
            std::size_t j
        );

        /**
         * @brief Retrieve the element at a given two-dimensional coordinate in
         * a const way.
         *
         * This method can be used to directly retrieve an object from an inner
         * vector.
         *
         * @param[in] i The row index to fetch from.
         * @param[in] j The index in row i to use.
         */
        VECMEM_HOST_AND_DEVICE
        const T & at(
            std::size_t i,
            std::size_t j
        ) const;

    private:
        /**
         * The number of rows in this jagged vector.
         */
        const std::size_t m_size;

        /**
         * The internal state of this jagged vector, which is heap-allocated by
         * the given memory manager.
         */
        details::vector_view<T> * const m_ptr;
    };
}

#include "jagged_device_vector.ipp"
