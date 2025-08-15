/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/containers/details/reverse_iterator.hpp"
#include "vecmem/memory/device_atomic_ref.hpp"
#include "vecmem/utils/type_traits.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <cstddef>
#include <type_traits>
#include <cstdint>

namespace vecmem {

static uint8_t * memory_buffer = nullptr;

// Forward declaration(s).
namespace edm {
template <typename T>
class device;
}

/// Class mimicking an @c std::vector in "device code"
///
/// This type can be used in "generic device code" as an @c std::vector that
/// allows modification of its elements. It does not allow the vector to be
/// resized, it just allows client code to access the data wrapped by the
/// vector with the same interface that @c std::vector provides.
///
template <typename TYPE>
class device_vector {

    // Make @c vecmem::edm::device a friend of this class.
    template <typename T>
    friend class edm::device;

public:
    /// @name Type definitions, mimicking @c std::vector
    /// @{

    /// Type of the array elements
    typedef TYPE value_type;
    /// Size type for the array
    typedef unsigned int size_type;
    /// Pointer difference type
    typedef std::ptrdiff_t difference_type;

    /// Pointer type to the size of the array
    typedef
        typename std::conditional<std::is_const<TYPE>::value, const size_type*,
                                  size_type*>::type size_pointer;

    /// Value reference type
    typedef value_type& reference;
    /// Constant value reference type
    typedef const value_type& const_reference;
    /// Value pointer type
    typedef value_type * pointer;
    /// Constant value pointer type
    typedef const value_type* const_pointer;

    /// Forward iterator type
    typedef pointer iterator;
    /// Constant forward iterator type
    typedef const_pointer const_iterator;
    /// Reverse iterator type
    typedef vecmem::details::reverse_iterator<iterator> reverse_iterator;
    /// Constant reverse iterator type
    typedef vecmem::details::reverse_iterator<const_iterator>
        const_reverse_iterator;

    /// @}

    /// Constructor, on top of a previously allocated/filled block of memory
    VECMEM_HOST_AND_DEVICE
    device_vector(const data::vector_view<value_type>& data);
    /// Construct a const device vector from a non-const data object
    template <typename OTHERTYPE,
              std::enable_if_t<details::is_same_nc<TYPE, OTHERTYPE>::value,
                               bool> = true>
    VECMEM_HOST_AND_DEVICE device_vector(
        const data::vector_view<OTHERTYPE>& data);
    /// Copy constructor
    VECMEM_HOST_AND_DEVICE
    device_vector(const device_vector& parent);

    /// Copy assignment operator
    VECMEM_HOST_AND_DEVICE
    device_vector& operator=(const device_vector& rhs);

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

    /// Access the underlying memory array (non-const)
    VECMEM_HOST_AND_DEVICE
    pointer data();
    /// Access the underlying memory array (const)
    VECMEM_HOST_AND_DEVICE
    const_pointer data() const;

    /// @}

    /// @name Payload modification functions
    /// @{

    /// Assign new values to the vector (not thread-safe)
    VECMEM_HOST_AND_DEVICE
    void assign(size_type count, const_reference value);
    /// Assign new values to the vector (not thread-safe)
    template <
        typename InputIt,
        std::enable_if_t<details::is_iterator_of<InputIt, value_type>::value,
                         bool> = true>
    VECMEM_HOST_AND_DEVICE void assign(InputIt other_begin, InputIt other_end) {

        // This can only be done on a resizable vector.
        assert(is_resizable);

        // Remove all previous elements.
        clear();

        // Create copies of all of the elements one-by-one. It's very
        // inefficient, but we can't make any assumptions about the type of the
        // input iterator received by this function.
        for (InputIt itr = other_begin; itr != other_end; ++itr) {
            construct(m_size++, *itr);
        }
    }

    /// Add a new element at the end of the vector (thread-safe)
    template <typename... Args>
    VECMEM_HOST_AND_DEVICE reference emplace_back(Args&&... args);
    /// Add a new element at the end of the vector (thread-safe)
    VECMEM_HOST_AND_DEVICE
    size_type push_back(const_reference value);

    /// Remove the last element of the vector (not thread-safe)
    VECMEM_HOST_AND_DEVICE
    size_type pop_back();

    /// Clear the vector (not thread-safe)
    VECMEM_HOST_AND_DEVICE
    void clear();
    /// Resize the vector (not thread-safe)
    VECMEM_HOST_AND_DEVICE
    void resize(size_type new_size);
    /// Resize the vector and fill any new elements with the specified value
    /// (not thread-safe)
    VECMEM_HOST_AND_DEVICE
    void resize(size_type new_size, const_reference value);

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

    /// Check whether the vector is empty
    VECMEM_HOST_AND_DEVICE
    bool empty() const;
    /// Return the number of elements in the vector
    VECMEM_HOST_AND_DEVICE
    size_type size() const;
    /// Return the maximum (fixed) number of elements in the vector
    VECMEM_HOST_AND_DEVICE
    size_type max_size() const;
    /// Return the current (fixed) capacity of the vector
    VECMEM_HOST_AND_DEVICE
    size_type capacity() const;

    /// @}

private:
    /// Construct a new vector element
    VECMEM_HOST_AND_DEVICE
    void construct(size_type pos, const_reference value);
    /// Destruct a vector element
    VECMEM_HOST_AND_DEVICE
    void destruct(size_type pos);

    pointer get_pointer() const; 
    /// Size of the array that this object looks at
    size_type m_capacity;
    /// Current size of the vector
    size_type m_size;

    /// Pointer to the start of the array
    size_type m_start_index;

    bool is_resizable;

};  // class device_vector

}  // namespace vecmem

// Include the implementation.
#include "vecmem/containers/impl/device_vector.ipp"
