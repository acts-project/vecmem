/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/details/reverse_iterator.hpp"
#include "vecmem/memory/device_atomic_ref.hpp"
#include "vecmem/utils/type_traits.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <cstddef>
#include <type_traits>

namespace vecmem {

/// @brief Namespace holding "data types"
///
/// These are types that either own, or only provide a view of data owned by
/// some other component. They are used for "non-interactive" data management
/// in the code.
///
namespace data {

/// Class holding data about a 1 dimensional vector/array
///
/// This type is meant to "formalise" the communication of data between
/// @c vecmem::vector, @c vecmem::array ("host types") and
/// @c vecmem::(const_)device_vector, @c vecmem::(const_)device_array
/// ("device types").
///
/// This type does not own the data that it points to. It merely provides a
/// "view" of that data.
///
template <typename TYPE>
class vector_view {

    /// We cannot use boolean types.
    static_assert(
        !std::is_same<typename std::remove_cv<TYPE>::type, bool>::value,
        "bool is not supported in VecMem containers");

public:
    /// @name Type definitions, mimicking @c std::vector
    /// @{

    /// Type of the array elements
    using value_type = TYPE;
    /// Size type used in the class
    using size_type = unsigned int;
    /// Pointer difference type
    using difference_type = std::ptrdiff_t;

    /// Pointer type to the size of the array
    using size_pointer =
        std::conditional_t<std::is_const<TYPE>::value,
                           std::add_pointer_t<std::add_const_t<size_type>>,
                           std::add_pointer_t<size_type>>;

    /// Value reference type
    using reference = std::add_lvalue_reference_t<value_type>;
    /// Constant value reference type
    using const_reference =
        std::add_lvalue_reference_t<std::add_const_t<value_type>>;
    /// Value pointer type
    using pointer = std::add_pointer_t<value_type>;
    /// Constant value pointer type
    using const_pointer = std::add_pointer_t<std::add_const_t<value_type>>;

    /// Forward iterator type
    using iterator = pointer;
    /// Constant forward iterator type
    using const_iterator = const_pointer;
    /// Reverse iterator type
    using reverse_iterator = vecmem::details::reverse_iterator<iterator>;
    /// Constant reverse iterator type
    using const_reverse_iterator =
        vecmem::details::reverse_iterator<const_iterator>;

    /// @}

    /// Default constructor
    vector_view() = default;
    /// Constant size data constructor
    VECMEM_HOST_AND_DEVICE
    vector_view(size_type size, pointer ptr);
    /// Resizable data constructor
    VECMEM_HOST_AND_DEVICE
    vector_view(size_type capacity, size_pointer size, pointer ptr);
    /// Copy constructor
    vector_view(const vector_view& parent) = default;
    /// Move constructor
    vector_view(vector_view&& parent) = default;

    /// Constructor from a "slightly different" @c vecmem::details::vector_view
    /// object
    ///
    /// Only enabled if the wrapped type is different, but only by const-ness.
    /// This complication is necessary to avoid problems from SYCL. Which is
    /// very particular about having default copy constructors for the types
    /// that it sends to kernels.
    ///
    template <typename OTHERTYPE,
              std::enable_if_t<details::is_same_nc<TYPE, OTHERTYPE>::value,
                               bool> = true>
    VECMEM_HOST_AND_DEVICE vector_view(const vector_view<OTHERTYPE>& parent);

    /// Copy assignment operator
    vector_view& operator=(const vector_view& rhs) = default;
    /// Move assignment operator
    vector_view& operator=(vector_view&& rhs) = default;

    /// Copy from a "slightly different" @c vecmem::details::vector_view object
    ///
    /// See the copy constructor for more details.
    ///
    template <typename OTHERTYPE,
              std::enable_if_t<details::is_same_nc<TYPE, OTHERTYPE>::value,
                               bool> = true>
    VECMEM_HOST_AND_DEVICE vector_view& operator=(
        const vector_view<OTHERTYPE>& rhs);

    /// @name Comparison operators
    /// @{

    /// Equality check. Two objects are only equal if they point at the same
    /// memory.
    template <typename OTHERTYPE,
              std::enable_if_t<std::is_same<std::remove_cv_t<TYPE>,
                                            std::remove_cv_t<OTHERTYPE>>::value,
                               bool> = true>
    VECMEM_HOST_AND_DEVICE bool operator==(
        const vector_view<OTHERTYPE>& rhs) const;

    /// Inequality check. Simply based on @c operator==.
    template <typename OTHERTYPE,
              std::enable_if_t<std::is_same<std::remove_cv_t<TYPE>,
                                            std::remove_cv_t<OTHERTYPE>>::value,
                               bool> = true>
    VECMEM_HOST_AND_DEVICE bool operator!=(
        const vector_view<OTHERTYPE>& rhs) const;

    /// @}

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
        assert(m_size != nullptr);

        // Remove all previous elements.
        clear();

        // Create copies of all of the elements one-by-one. It's very
        // inefficient, but we can't make any assumptions about the type of the
        // input iterator received by this function.
        device_atomic_ref<size_type> asize(*m_size);
        for (InputIt itr = other_begin; itr != other_end; ++itr) {
            construct(asize.fetch_add(1), *itr);
        }
    }
    /// Assign another vector_view's values to this view (not thread-safe)
    template <typename OTHERTYPE,
              std::enable_if_t<std::is_convertible<OTHERTYPE, TYPE>::value,
                               bool> = true>
    VECMEM_HOST_AND_DEVICE void assign(const vector_view<OTHERTYPE>& rhs);

    /// Add a new element at the end of the vector (thread-safe)
    template <typename... Args>
    VECMEM_HOST_AND_DEVICE reference emplace_back(Args&&... args);
    /// Add a new element at the end of the vector (thread-safe)
    VECMEM_HOST_AND_DEVICE
    size_type push_back(const_reference value);

    /// Default-construct a given number of elements at the end of the vector.
    ///
    /// @note This function runs in Θ(n) time, performing n default
    /// constructions.
    VECMEM_HOST_AND_DEVICE size_type bulk_append(size_type n);

    /// Copy-construct a given number of elements at the end of the vector,
    /// copying the given object.
    ///
    /// @note This function runs in Θ(n) time, performing n copy constructions.
    VECMEM_HOST_AND_DEVICE size_type bulk_append(size_type n,
                                                 const_reference v);

    /// Reserve a fixed number of slots in the array in a standards-conformant
    /// way.
    ///
    /// @note This function runs in Θ(1) time.
    ///
    /// @warning This method is only standards-conformant in C++20 and later.
    VECMEM_HOST_AND_DEVICE size_type bulk_append_implicit(size_type n);

    /// Reserve a fixed number of slots in the array in a way that technically
    /// is ill-formed.
    ///
    /// @note This function runs in Θ(1) time.
    ///
    /// @warning This method requires that the client guarantees that the array
    /// elements reserved are given a proper lifetime before they are used.
    VECMEM_HOST_AND_DEVICE size_type bulk_append_implicit_unsafe(size_type n);

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

    /// Resize a vector of implicit lifetime types
    ///
    /// @note This function runs in Θ(1) time.
    VECMEM_HOST_AND_DEVICE void resize_implicit(size_type new_size);

    /// Resize a vector in constant time, unsafely deallocating non-implicit
    /// lifetime types.
    ///
    /// @note This function runs in Θ(1) time.
    VECMEM_HOST_AND_DEVICE void resize_implicit_unsafe(size_type new_size);

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

    /// Get a pointer to the size of the vector
    VECMEM_HOST_AND_DEVICE
    size_pointer size_ptr() const;

    /// Get a pointer to the vector elements
    VECMEM_HOST_AND_DEVICE
    pointer ptr() const;

private:
    /// Construct a new vector element
    VECMEM_HOST_AND_DEVICE
    void construct(size_type pos, const_reference value);
    /// Destruct a vector element
    VECMEM_HOST_AND_DEVICE
    void destruct(size_type pos);

    /// Maximum capacity of the array
    size_type m_capacity;
    /// Pointer to the size of the array in memory
    size_pointer m_size;
    /// Pointer to the start of the memory block/array
    pointer m_ptr;

};  // struct vector_view

}  // namespace data
}  // namespace vecmem

namespace std {

/// @name Type trait specialization(s) for @c vecmem::data::vector_view
/// @{

template <typename TYPE>
struct add_const<vecmem::data::vector_view<TYPE>> {
    using type = const vecmem::data::vector_view<std::add_const_t<TYPE>>;
};

template <typename TYPE>
struct add_const<const vecmem::data::vector_view<TYPE>> {
    using type = const vecmem::data::vector_view<std::add_const_t<TYPE>>;
};

template <typename TYPE>
struct remove_const<vecmem::data::vector_view<TYPE>> {
    using type = vecmem::data::vector_view<std::remove_const_t<TYPE>>;
};

template <typename TYPE>
struct remove_const<const vecmem::data::vector_view<TYPE>> {
    using type = vecmem::data::vector_view<std::remove_const_t<TYPE>>;
};

template <typename TYPE>
struct remove_cv<vecmem::data::vector_view<TYPE>> {
    using type = vecmem::data::vector_view<std::remove_cv_t<TYPE>>;
};

template <typename TYPE>
struct remove_cv<const vecmem::data::vector_view<TYPE>> {
    using type = vecmem::data::vector_view<std::remove_cv_t<TYPE>>;
};

/// @}

}  // namespace std

// Include the implementation.
#include "vecmem/containers/impl/vector_view.ipp"
