/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/details/tuple.hpp"

// System include(s).
#include <cstddef>
#include <type_traits>

namespace vecmem {
namespace edm {
namespace details {

/// Struct used to implement @c vecmem::edm::details::get in a C++14 stype
///
/// @tparam I The index of the tuple element to get
///
template <std::size_t I>
struct get_impl {

    /// Get the I-th (const) tuple element recursively
    template <typename... Ts>
    VECMEM_HOST_AND_DEVICE static constexpr const auto &get(
        const tuple<Ts...> &t) {
        return get_impl<I - 1>::get(t.m_tail);
    }
    /// Get the I-th (non-const) tuple element recursively
    template <typename... Ts>
    VECMEM_HOST_AND_DEVICE static constexpr auto &get(tuple<Ts...> &t) {
        return get_impl<I - 1>::get(t.m_tail);
    }

};  // struct get_impl

/// Specialization of @c vecmem::edm::details::get_impl for the 0th element
///
/// @tparam ...Ts Types stored in the tuple of interest
///
template <>
struct get_impl<0> {

    /// Get the first (const) tuple element
    template <typename... Ts>
    VECMEM_HOST_AND_DEVICE static constexpr const auto &get(
        const tuple<Ts...> &t) {
        return t.m_head;
    }
    /// Get the first (non-const) tuple element
    template <typename... Ts>
    VECMEM_HOST_AND_DEVICE static constexpr auto &get(tuple<Ts...> &t) {
        return t.m_head;
    }

};  // struct get_impl

/// Get a constant element out of a tuple
///
/// @tparam I The index of the element to get
/// @tparam ...Ts The types held by the tuple
/// @return The I-th element of the tuple
///
template <std::size_t I, typename... Ts>
VECMEM_HOST_AND_DEVICE constexpr const auto &get(
    const tuple<Ts...> &t) noexcept {

    // Make sure that the requested index is valid.
    static_assert(I < sizeof...(Ts),
                  "Attempt to access index greater than tuple size.");

    // Return the correct element using the helper struct.
    return get_impl<I>::get(t);
}

/// Get a non-constant element out of a tuple
///
/// @tparam I The index of the element to get
/// @tparam ...Ts The types held by the tuple
/// @return The I-th element of the tuple
///
template <std::size_t I, typename... Ts>
VECMEM_HOST_AND_DEVICE constexpr auto &get(tuple<Ts...> &t) noexcept {

    // Make sure that the requested index is valid.
    static_assert(I < sizeof...(Ts),
                  "Attempt to access index greater than tuple size.");

    // Return the correct element using the helper struct.
    return get_impl<I>::get(t);
}

/// Tie references to existing objects, into a tuple
///
/// @tparam ...Ts Types to refer to with the resulting tuple
/// @param ...args References to the objects that the tuple should point to
/// @return A tuple of references to some existing objects
///
template <typename... Ts>
VECMEM_HOST_AND_DEVICE constexpr tuple<Ts &...> tie(Ts &... args) {

    return tuple<Ts &...>(args...);
}

/// Default/empty implementation for @c vecmem::edm::details::tuple_element
///
/// @tparam T Dummy template argument
/// @tparam I Dummy index argument
///
template <std::size_t I, class T>
struct tuple_element;

/// Get the type of the I-th element of a tuple
///
/// @tparam ...Ts The element types in the tuple
/// @tparam I     Index of the element to get the type of
///
template <std::size_t I, typename... Ts>
struct tuple_element<I, tuple<Ts...>> {

    /// Type of the I-th element of the specified tuple
    using type = std::decay_t<decltype(get<I>(std::declval<tuple<Ts...>>()))>;
};

/// Convenience accessor for the I-th element of a tuple
///
/// @tparam T The type of the tuple to investigate
/// @tparam I Index of the element to get the type of
///
template <std::size_t I, class T>
using tuple_element_t = typename tuple_element<I, T>::type;

/// Make a tuple with automatic type deduction
///
/// @tparam ...Ts Types deduced for the resulting tuple
/// @param args   Values to make a tuple out of
/// @return A tuple constructed from the received values
///
template <class... Ts>
VECMEM_HOST_AND_DEVICE inline constexpr tuple<typename std::decay<Ts>::type...>
make_tuple(Ts &&... args) {
    return tuple<typename std::decay<Ts>::type...>{std::forward<Ts>(args)...};
}

}  // namespace details
}  // namespace edm
}  // namespace vecmem
