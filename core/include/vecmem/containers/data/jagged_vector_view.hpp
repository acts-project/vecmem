/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/utils/type_traits.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <cstddef>
#include <type_traits>

namespace vecmem {
namespace data {
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
template <typename T>
struct jagged_vector_view {

    /// Size type used in the class
    typedef std::size_t size_type;
    /// Value type of the jagged array
    typedef vector_view<T> value_type;
    /// Pointer type to the jagged array
    typedef value_type* pointer;

    /**
     * Default constructor
     */
    jagged_vector_view() = default;
    /**
     * Constructor with all the information held by the object.
     */
    VECMEM_HOST_AND_DEVICE
    jagged_vector_view(size_type size, pointer ptr);

    /**
     * Constructor from a "slightly different" @c
     * vecmem::details::jagged_vector_view object
     *
     * Only enabled if the wrapped type is different, but only by
     * const-ness. This complication is necessary to avoid problems from
     * SYCL. Which is very particular about having default copy
     * constructors for the types that it sends to kernels.
     */
    template <
        typename OTHERTYPE,
        std::enable_if_t<details::is_same_nc<T, OTHERTYPE>::value, bool> = true>
    VECMEM_HOST_AND_DEVICE jagged_vector_view(
        const jagged_vector_view<OTHERTYPE>& parent);

    /**
     * The number of rows in this jagged vector.
     */
    size_type m_size;

    /**
     * The internal state of this jagged vector, which is heap-allocated by
     * the given memory manager.
     */
    pointer m_ptr;

};  // struct jagged_vector_view

}  // namespace data
}  // namespace vecmem

// Include the implementation.
#include "vecmem/containers/impl/jagged_vector_view.ipp"
