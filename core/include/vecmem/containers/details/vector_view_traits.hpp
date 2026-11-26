/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Project include(s).
#include "vecmem/utils/type_traits.hpp"

namespace vecmem {
namespace data {

// Forward declaration of @c vecmem::data::vector_view
template <typename T>
class vector_view;

}  // namespace data

namespace details {

/// @name Type trait specialization(s) for @c vecmem::data::vector_view
/// @{

template <typename TYPE>
struct is_const<vecmem::data::vector_view<TYPE>> {
    static constexpr bool value = std::is_const<TYPE>::value;
};

template <typename TYPE>
struct is_const<const vecmem::data::vector_view<TYPE>> {
    static constexpr bool value = true;
};

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

/// @name Trait providing a const reference for @c data::vector_view
/// @{

template <typename T>
struct view_const_reference {
    using type = std::add_lvalue_reference_t<std::add_const_t<T>>;
};

template <typename T>
struct view_const_reference<data::vector_view<T>> {
    // This here is very much on purpose. Since we may return a different view
    // than what we have in memory, we cannot return a reference. We return
    // such objects by value instead.
    using type = vecmem::details::add_const_t<data::vector_view<T>>;
};

template <typename T>
using view_const_reference_t = typename view_const_reference<T>::type;

/// @}

}  // namespace details
}  // namespace vecmem
