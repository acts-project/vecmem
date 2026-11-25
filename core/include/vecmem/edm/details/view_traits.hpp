/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/edm/schema.hpp"
#include "vecmem/utils/tuple.hpp"

// System include(s).
#include <type_traits>

namespace vecmem {
namespace edm {
namespace details {

/// @name Traits for the view types for the individual variables
/// @{

template <typename TYPE>
struct view_type_base {
    using payload_type = TYPE;
    using payload_ptr = std::add_pointer_t<payload_type>;
    using size_type = unsigned int;
    using size_ptr = std::add_pointer_t<size_type>;
};  // struct view_type_base

template <typename TYPE>
struct view_type : public view_type_base<TYPE> {};

template <typename TYPE>
struct view_type<type::scalar<TYPE>> : public view_type_base<TYPE> {
    using layout_type = int;
    using layout_ptr = std::add_pointer_t<layout_type>;
    using type = typename view_type_base<TYPE>::payload_ptr;
    using return_type = std::add_lvalue_reference_t<
        typename view_type_base<TYPE>::payload_type>;
    using const_return_type = std::add_lvalue_reference_t<
        std::add_const_t<typename view_type_base<TYPE>::payload_type>>;
};  // struct view_type

template <typename TYPE>
struct view_type<type::vector<TYPE>> : public view_type_base<TYPE> {
    using layout_type = int;
    using layout_ptr = std::add_pointer_t<layout_type>;
    using type = vecmem::data::vector_view<TYPE>;
    using return_type = std::add_lvalue_reference_t<type>;
    using const_return_type = vecmem::details::add_const_t<type>;
};  // struct view_type

template <typename TYPE>
struct view_type<type::jagged_vector<TYPE>> : public view_type_base<TYPE> {
    using layout_type = vecmem::data::vector_view<TYPE>;
    using layout_ptr = std::add_pointer_t<layout_type>;
    using type = vecmem::data::jagged_vector_view<TYPE>;
    using return_type = std::add_lvalue_reference_t<type>;
    using const_return_type = vecmem::details::add_const_t<type>;
};  // struct view_type

template <std::size_t INDEX, typename... VARTYPES>
struct view_type_at {
    using type =
        typename view_type<tuple_element_t<INDEX, tuple<VARTYPES...>>>::type;
    using return_type = typename view_type<
        tuple_element_t<INDEX, tuple<VARTYPES...>>>::return_type;
    using const_return_type = typename view_type<
        tuple_element_t<INDEX, tuple<VARTYPES...>>>::const_return_type;
};  // struct device_type_at

/// @}

/// @name Helper traits for the @c vecmem::edm::view::get functions
/// @{

template <typename TYPE>
struct view_get {
    VECMEM_HOST_AND_DEVICE
    static constexpr typename view_type<TYPE>::return_type get(
        typename view_type<TYPE>::type& variable) {

        return variable;
    }
    VECMEM_HOST_AND_DEVICE
    static constexpr typename view_type<TYPE>::const_return_type get(
        const typename view_type<TYPE>::type& variable) {

        return variable;
    }
};  // struct view_get

template <typename TYPE>
struct view_get<type::scalar<TYPE>> {
    VECMEM_HOST_AND_DEVICE
    static constexpr typename view_type<type::scalar<TYPE>>::return_type get(
        typename view_type<type::scalar<TYPE>>::type& variable) {

        return *variable;
    }
    VECMEM_HOST_AND_DEVICE
    static constexpr typename view_type<type::scalar<TYPE>>::const_return_type
    get(const typename view_type<type::scalar<TYPE>>::type& variable) {

        return *variable;
    }
};  // struct view_get

/// @}

}  // namespace details
}  // namespace edm
}  // namespace vecmem
