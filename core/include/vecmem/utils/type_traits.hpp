/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <iterator>
#include <type_traits>

namespace vecmem {
namespace details {

/// Helper trait for identifying input iterators
///
/// It comes in handy in some of the functions of the custom (device)
/// container types that use templated iterator values. Which could hide
/// overloads of the same function with the same number of (non-templated)
/// arguments.
///
/// The implementation is *very* simplistic at the moment. It could/should
/// be made more elaborate when the need arises.
///
template <typename iterator_type, typename value_type>
using is_iterator_of = std::is_convertible<
    typename std::iterator_traits<iterator_type>::value_type, value_type>;

/// Helper trait for detecting when a type is a non-const version of another
///
/// This comes into play multiple times to enable certain constructors
/// conditionally through SFINAE.
///
template <typename CTYPE, typename NCTYPE>
struct is_same_nc {
    static constexpr bool value = false;
};

template <typename TYPE>
struct is_same_nc<const TYPE, TYPE> {
    static constexpr bool value = true;
};

/// Implementation for @c std::conjunction
///
/// Since @c std::conjunction is only available starting with C++17, but it
/// comes in very handy in some places in the VecMem code, this is a naive
/// custom implementation for it.
///
template <class...>
struct conjunction : std::true_type {};

template <class B1>
struct conjunction<B1> : B1 {};

template <class B1, class... Bn>
struct conjunction<B1, Bn...>
    : std::conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};

template <class... B>
constexpr bool conjunction_v = conjunction<B...>::value;

}  // namespace details
}  // namespace vecmem
