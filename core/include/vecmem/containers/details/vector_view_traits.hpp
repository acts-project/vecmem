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

}  // namespace details
}  // namespace vecmem
