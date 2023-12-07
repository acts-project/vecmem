/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/schema.hpp"
#include "vecmem/utils/type_traits.hpp"

// System include(s).
#include <type_traits>

namespace vecmem {
namespace edm {
namespace type {
namespace details {

/// @name Traits turning variable types into constant types
/// @{

template <typename T>
struct add_const;

template <typename TYPE>
struct add_const<type::scalar<TYPE>> {
    using type = type::scalar<std::add_const_t<TYPE>>;
};  // struct add_const

template <typename TYPE>
struct add_const<type::vector<TYPE>> {
    using type = type::vector<std::add_const_t<TYPE>>;
};  // struct add_const

template <typename TYPE>
struct add_const<type::jagged_vector<TYPE>> {
    using type = type::jagged_vector<std::add_const_t<TYPE>>;
};  // struct add_const

template <typename T>
using add_const_t = typename add_const<T>::type;

/// @}

/// @name Traits checking the type of a variable
/// @{

template <typename T>
struct is_scalar {
    static constexpr bool value = false;
};  // struct is_scalar

template <typename TYPE>
struct is_scalar<type::scalar<TYPE>> {
    static constexpr bool value = true;
};  // struct is_scalar

template <typename T>
constexpr bool is_scalar_v = is_scalar<T>::value;

template <typename T>
struct is_vector {
    static constexpr bool value = false;
};  // struct is_vector

template <typename TYPE>
struct is_vector<type::vector<TYPE>> {
    static constexpr bool value = true;
};  // struct is_vector

template <typename TYPE>
struct is_vector<type::jagged_vector<TYPE>> {
    static constexpr bool value = true;
};  // struct is_vector

template <typename T>
constexpr bool is_vector_v = is_vector<T>::value;

template <typename T>
struct is_jagged_vector {
    static constexpr bool value = false;
};  // struct is_jagged_vector

template <typename TYPE>
struct is_jagged_vector<type::jagged_vector<TYPE>> {
    static constexpr bool value = true;
};  // struct is_jagged_vector

template <typename T>
constexpr bool is_jagged_vector_v = is_jagged_vector<T>::value;

/// @}

/// @name Traits checking if two types are the same, except for constness
/// @{

template <typename TYPE1, typename TYPE2>
struct is_same_nc {
    static constexpr bool value = false;
};  // struct is_same_nc

template <typename TYPE1, typename TYPE2>
struct is_same_nc<type::scalar<TYPE1>, type::scalar<TYPE2>> {
    static constexpr bool value =
        vecmem::details::is_same_nc<TYPE1, TYPE2>::value;
};  // struct is_same_nc

template <typename TYPE1, typename TYPE2>
struct is_same_nc<type::vector<TYPE1>, type::vector<TYPE2>> {
    static constexpr bool value =
        vecmem::details::is_same_nc<TYPE1, TYPE2>::value;
};  // struct is_same_nc

template <typename TYPE1, typename TYPE2>
struct is_same_nc<type::jagged_vector<TYPE1>, type::jagged_vector<TYPE2>> {
    static constexpr bool value =
        vecmem::details::is_same_nc<TYPE1, TYPE2>::value;
};  // struct is_same_nc

template <typename TYPE1, typename TYPE2>
constexpr bool is_same_nc_v = is_same_nc<TYPE1, TYPE2>::value;

/// @}

}  // namespace details
}  // namespace type

namespace details {

/// @name Trait(s) making an entire schema into a constant one
/// @{

template <typename... VARTYPES>
struct add_const;

template <typename... VARTYPES>
struct add_const<schema<VARTYPES...>> {
    using type = schema<typename type::details::add_const<VARTYPES>::type...>;
};

template <typename... VARTYPES>
using add_const_t = typename add_const<VARTYPES...>::type;

/// @}

}  // namespace details
}  // namespace edm
}  // namespace vecmem
