/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include "vecmem/containers/data/vector_view.hpp"

#include <cstddef>

namespace vecmem { namespace data {
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
    struct jagged_vector_view {
        jagged_vector_view(
            std::size_t size,
            vector_view<T> * ptr
        );

        /**
         * The number of rows in this jagged vector.
         */
        std::size_t m_size;

        /**
         * The internal state of this jagged vector, which is heap-allocated by
         * the given memory manager.
         */
        vector_view<T> * m_ptr;
    };
} } // namespace vecmem::data

// Include the implementation.
#include "vecmem/containers/impl/jagged_vector_view.ipp"
