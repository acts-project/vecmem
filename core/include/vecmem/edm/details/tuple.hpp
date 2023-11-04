/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/utils/type_traits.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <type_traits>

namespace vecmem {
namespace edm {
namespace details {

/// Default tuple type
///
/// Serving as the final node in the recursive implementation of this tuple
/// type.
///
template <typename... Ts>
struct tuple {

    // As long as we did everything correctly, this should only get instantiated
    // with an empty parameter list, for the implementation to work correctly.
    static_assert(sizeof...(Ts) == 0,
                  "There's a coding error in vecmem::edm::details::tuple!");

    /// Default constructor for the default tuple type
    VECMEM_HOST_AND_DEVICE tuple() {}

};  // struct tuple

/// Simple tuple implementation for the @c vecmem::edm classes
///
/// The @c vecmem::edm classes require something analogous to @c std::tuple,
/// but that type is not officially supported by CUDA in device code. Worse yet,
/// @c std::tuple actively generates invalid code with @c nvcc at the time of
/// writing (up to CUDA 12.3.0).
///
/// This is a very simple implementation for a tuple type, which can do exactly
/// as much as we need from it for @c vecmem::edm.
///
/// @tparam T     The first type to be stored in the tuple
/// @tparam ...Ts The rest of the types to be stored in the tuple
///
template <typename T, typename... Ts>
struct tuple<T, Ts...> {

    /// Default constructor
    VECMEM_HOST_AND_DEVICE constexpr tuple() : m_head{}, m_tail{} {}

    /// Copy constructor
    ///
    /// @param parent The parent to copy
    ///
    template <typename U, typename... Us,
              std::enable_if_t<sizeof...(Ts) == sizeof...(Us), bool> = true>
    VECMEM_HOST_AND_DEVICE constexpr tuple(const tuple<U, Us...> &parent)
        : m_head(parent.m_head), m_tail(parent.m_tail) {}

    /// Main constructor, from a list of tuple elements
    ///
    /// @param head The first element to be stored in the tuple
    /// @param tail The rest of the elements to be stored in the tuple
    ///
    template <typename U, typename... Us,
              std::enable_if_t<vecmem::details::conjunction<
                                   std::is_constructible<T, U &&>,
                                   std::is_constructible<Ts, Us &&>...>::value,
                               bool> = true>
    VECMEM_HOST_AND_DEVICE constexpr tuple(U &&head, Us &&... tail)
        : m_head(std::forward<U>(head)), m_tail(std::forward<Us>(tail)...) {}

    /// The first/head element of the tuple
    T m_head;
    /// The rest of the tuple elements
    tuple<Ts...> m_tail;

};  // struct tuple

}  // namespace details
}  // namespace edm
}  // namespace vecmem
