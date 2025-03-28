/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "vecmem/containers/details/jagged_device_vector_iterator.hpp"
#include "vecmem/containers/details/reverse_iterator.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/utils/type_traits.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <cstddef>
#include <type_traits>

namespace vecmem {

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
class jagged_device_vector {

public:
    /// @name Type definitions, mimicking @c std::vector
    /// @{

    /// Type of the "outer" array elements
    using value_type = device_vector<T>;
    /// Size type for the array
    using size_type = std::size_t;
    /// Pointer difference type
    using difference_type = std::ptrdiff_t;

    /// Value reference type
    using reference = device_vector<T>;
    /// Constant value reference type
    using const_reference = device_vector<std::add_const_t<T> >;

    /// Forward iterator type
    using iterator = details::jagged_device_vector_iterator<T>;
    /// Constant forward iterator type
    using const_iterator =
        details::jagged_device_vector_iterator<std::add_const_t<T> >;
    /// Reverse iterator type
    using reverse_iterator = details::reverse_iterator<iterator>;
    /// Constant reverse iterator type
    using const_reverse_iterator = details::reverse_iterator<const_iterator>;

    /// @}

    /**
     * @brief Construct a jagged device vector from a jagged vector view
     * object.
     */
    VECMEM_HOST_AND_DEVICE
    explicit jagged_device_vector(data::jagged_vector_view<T> data);

    /// Copy constructor
    VECMEM_HOST_AND_DEVICE
    jagged_device_vector(const jagged_device_vector& parent);

    /// Copy assignment operator
    VECMEM_HOST_AND_DEVICE
    jagged_device_vector& operator=(const jagged_device_vector& rhs);

    /// @name Vector element access functions
    /// @{

    /// Return a specific element of the vector in a "safe way" (non-const)
    VECMEM_HOST_AND_DEVICE
    reference at(size_type pos);
    /// Return a specific element of the vector in a "safe way" (const)
    VECMEM_HOST_AND_DEVICE
    const_reference at(size_type pos) const;

    /// Return a specific element of the vector (non-const)
    VECMEM_HOST_AND_DEVICE
    reference operator[](size_type pos);
    /// Return a specific element of the vector (const)
    VECMEM_HOST_AND_DEVICE
    const_reference operator[](size_type pos) const;

    /// Return the first element of the vector (non-const)
    VECMEM_HOST_AND_DEVICE
    reference front();
    /// Return the first element of the vector (const)
    VECMEM_HOST_AND_DEVICE
    const_reference front() const;

    /// Return the last element of the vector (non-const)
    VECMEM_HOST_AND_DEVICE
    reference back();
    /// Return the last element of the vector (const)
    VECMEM_HOST_AND_DEVICE
    const_reference back() const;

    /// @}

    /// @name Iterator providing functions
    /// @{

    /// Return a forward iterator pointing at the beginning of the vector
    VECMEM_HOST_AND_DEVICE
    iterator begin();
    /// Return a constant forward iterator pointing at the beginning of the
    /// vector
    VECMEM_HOST_AND_DEVICE
    const_iterator begin() const;
    /// Return a constant forward iterator pointing at the beginning of the
    /// vector
    VECMEM_HOST_AND_DEVICE
    const_iterator cbegin() const;

    /// Return a forward iterator pointing at the end of the vector
    VECMEM_HOST_AND_DEVICE
    iterator end();
    /// Return a constant forward iterator pointing at the end of the vector
    VECMEM_HOST_AND_DEVICE
    const_iterator end() const;
    /// Return a constant forward iterator pointing at the end of the vector
    VECMEM_HOST_AND_DEVICE
    const_iterator cend() const;

    /// Return a reverse iterator pointing at the end of the vector
    VECMEM_HOST_AND_DEVICE
    reverse_iterator rbegin();
    /// Return a constant reverse iterator pointing at the end of the vector
    VECMEM_HOST_AND_DEVICE
    const_reverse_iterator rbegin() const;
    /// Return a constant reverse iterator pointing at the end of the vector
    VECMEM_HOST_AND_DEVICE
    const_reverse_iterator crbegin() const;

    /// Return a reverse iterator pointing at the beginning of the vector
    VECMEM_HOST_AND_DEVICE
    reverse_iterator rend();
    /// Return a constant reverse iterator pointing at the beginning of the
    /// vector
    VECMEM_HOST_AND_DEVICE
    const_reverse_iterator rend() const;
    /// Return a constant reverse iterator pointing at the beginning of the
    /// vector
    VECMEM_HOST_AND_DEVICE
    const_reverse_iterator crend() const;

    /// @}

    /// @name Capacity checking functions
    /// @{

    /**
     * @brief Checks whether this view has no rows.
     *
     * Returns true if the jagged vector is empty, and false otherwise.
     *
     * @note A jagged vector of shape [[]] (that is to say, an empty row) is
     * not considered empty, but a jagged vector of shape [] is.
     */
    VECMEM_HOST_AND_DEVICE
    bool empty(void) const;

    /**
     * @brief Get the number of rows in this view.
     */
    VECMEM_HOST_AND_DEVICE
    size_type size(void) const;

    /// Return the maximum (fixed) number of elements in the vector
    VECMEM_HOST_AND_DEVICE
    size_type max_size() const;
    /// Return the current (fixed) capacity of the vector
    VECMEM_HOST_AND_DEVICE
    size_type capacity() const;

    /// @}

private:
    /**
     * The number of rows in this jagged vector.
     */
    size_type m_size;

    /**
     * Objects representing the "inner vectors" towards the user.
     */
    typename data::jagged_vector_view<T>::pointer m_ptr;

};  // class jagged_device_vector

}  // namespace vecmem

// Include the implementation.
#include "vecmem/containers/impl/jagged_device_vector.ipp"
