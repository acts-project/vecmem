/**
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include "vecmem/utils/types.hpp"

namespace vecmem {
/**
 * @brief Simple statically-sized array-like class designed for use in
 * device code.
 *
 * This class is designed to be an almost-drop-in replacement for std::array
 * which can be used in device code.
 *
 * @tparam T The array type.
 * @tparam N The size of the array.
 */
template <typename T, std::size_t N>
class static_array {
    public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;

    /**
     * @brief Trivial constructor.
     *
     * This constructor does nothing, and leaves the inner array
     * uninitialized.
     */
    VECMEM_HOST_AND_DEVICE
    static_array(void);

    /**
     * @brief Construct an array from a parameter pack of arbitrary size.
     *
     * This constructor is of arbitrary arity, and inserts those elements in
     * order into the array.
     *
     * @tparam Tp The parameter pack for the arguments.
     *
     * @warning The std::array implementation of this constructor requires
     * the parameter list to be of size equal to or less than the array
     * itself. This class is slightly different, as it requires the argument
     * list to be exactly the same length. This protects the user from
     * accidentally initializing an array with fewer values than necessary.
     *
     * @note All parameters passed to this function must be convertible to
     * the array value type, but the values do not need to be homogeneous.
     */
    template <
        typename... Tp, typename = std::enable_if_t<sizeof...(Tp) == N>,
        typename = std::enable_if_t<(std::is_convertible_v<Tp, T> && ...)> >
    VECMEM_HOST_AND_DEVICE static_array(Tp &&... a)
        : static_array(std::index_sequence_for<Tp...>(),
                       std::forward<Tp>(a)...) {}

    /**
     * @brief Bounds-checked accessor method.
     *
     * Since this method can throw an exception, this is not usable on the
     * device side.
     *
     * @param[in] i The index to access.
     *
     * @return The value at index i if i less less than N, otherwise an
     * exception is thrown.
     */
    VECMEM_HOST
    constexpr reference at(size_type i);

    /**
     * @brief Constant bounds-checked accessor method.
     *
     * Since this method can throw an exception, this is not usable on the
     * device side.
     *
     * @param[in] i The index to access.
     *
     * @return The value at index i if i less less than N, otherwise an
     * exception is thrown.
     */
    VECMEM_HOST
    constexpr const_reference at(size_type i) const;

    /**
     * @brief Accessor method.
     *
     * @param[in] i The index to access.
     *
     * @return The value at index i if i less less than N, otherwise the
     * behaviour is undefined.
     */
    VECMEM_HOST_AND_DEVICE
    constexpr reference operator[](size_type i);

    /**
     * @brief Constant accessor method.
     *
     * @param[in] i The index to access.
     *
     * @return The value at index i if i less less than N, otherwise the
     * behaviour is undefined.
     */
    VECMEM_HOST_AND_DEVICE
    constexpr const_reference operator[](size_type i) const;

    /**
     * @brief Access the front element of the array.
     *
     * @return The first element of the array.
     */
    VECMEM_HOST_AND_DEVICE
    constexpr reference front(void);

    /**
     * @brief Access the front element of the array in a const fashion.
     *
     * @return The first element of the array.
     */
    VECMEM_HOST_AND_DEVICE
    constexpr const_reference front(void) const;

    /**
     * @brief Access the back element of the array.
     *
     * @return The last element of the array.
     */
    VECMEM_HOST_AND_DEVICE
    constexpr reference back(void);

    /**
     * @brief Access the back element of the array in a const fashion.
     *
     * @return The last element of the array.
     */
    VECMEM_HOST_AND_DEVICE
    constexpr const_reference back(void) const;

    /**
     * @brief Access the underlying data of the array.
     *
     * @return A pointer to the underlying memory.
     */
    VECMEM_HOST_AND_DEVICE
    constexpr pointer data(void);

    /**
     * @brief Access the underlying data of the array in a const fasion.
     *
     * @return A pointer to the underlying memory.
     */
    VECMEM_HOST_AND_DEVICE
    constexpr const_pointer data(void) const;

    private:
    /**
     * @brief Private helper-constructor for the parameter pack constructor.
     *
     * HACK: This template pack is defined as std::size_t instead of
     * size_type because the SYCL compiler refuses to accept it otherwise.
     */
    template <std::size_t... Is, typename... Tp>
    VECMEM_HOST_AND_DEVICE static_array(std::index_sequence<Is...>, Tp &&... a);

    T v[N];
};

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE bool operator==(const static_array<T, N> &lhs,
                                       const static_array<T, N> &rhs);

template <typename T, std::size_t N>
VECMEM_HOST_AND_DEVICE bool operator!=(const static_array<T, N> &lhs,
                                       const static_array<T, N> &rhs);
}  // namespace vecmem

#include "impl/static_array.ipp"
