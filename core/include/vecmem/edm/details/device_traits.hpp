/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/jagged_device_vector.hpp"
#include "vecmem/edm/schema.hpp"
#include "vecmem/edm/view.hpp"
#include "vecmem/utils/tuple.hpp"

// System include(s).
#include <type_traits>
#include <utility>

namespace vecmem {
namespace edm {
namespace details {

/// @name Traits for the device types for the individual variables
/// @{

template <typename TYPE>
struct device_type;

template <typename TYPE>
struct device_type<type::scalar<TYPE>> {
    using type = std::add_pointer_t<TYPE>;
    using return_type = std::add_lvalue_reference_t<TYPE>;
    using const_return_type =
        std::add_lvalue_reference_t<std::add_const_t<TYPE>>;
};  // struct device_type

template <typename TYPE>
struct device_type<type::vector<TYPE>> {
    using type = device_vector<TYPE>;
    using return_type = std::add_lvalue_reference_t<type>;
    using const_return_type =
        std::add_lvalue_reference_t<std::add_const_t<type>>;
};  // struct device_type

template <typename TYPE>
struct device_type<type::jagged_vector<TYPE>> {
    using type = jagged_device_vector<TYPE>;
    using return_type = std::add_lvalue_reference_t<type>;
    using const_return_type =
        std::add_lvalue_reference_t<std::add_const_t<type>>;
};  // struct device_type

template <std::size_t INDEX, typename... VARTYPES>
struct device_type_at {
    using type =
        typename device_type<tuple_element_t<INDEX, tuple<VARTYPES...>>>::type;
    using return_type = typename device_type<
        tuple_element_t<INDEX, tuple<VARTYPES...>>>::return_type;
    using const_return_type = typename device_type<
        tuple_element_t<INDEX, tuple<VARTYPES...>>>::const_return_type;
};  // struct device_type_at

/// @}

/// @name Helper traits for the @c vecmem::edm::device::get functions
/// @{

template <typename TYPE>
struct device_get {
    VECMEM_HOST_AND_DEVICE
    static constexpr typename device_type<TYPE>::return_type get(
        typename device_type<TYPE>::type& variable) {

        return variable;
    }
    VECMEM_HOST_AND_DEVICE
    static constexpr typename device_type<TYPE>::const_return_type get(
        const typename device_type<TYPE>::type& variable) {

        return variable;
    }
};  // struct device_get

template <typename TYPE>
struct device_get<type::scalar<TYPE>> {
    VECMEM_HOST_AND_DEVICE
    static constexpr typename device_type<type::scalar<TYPE>>::return_type get(
        typename device_type<type::scalar<TYPE>>::type& variable) {

        return *variable;
    }
    VECMEM_HOST_AND_DEVICE
    static constexpr typename device_type<type::scalar<TYPE>>::const_return_type
    get(const typename device_type<type::scalar<TYPE>>::type& variable) {

        return *variable;
    }
};  // struct device_get

/// @}

/// Check whether a scalar variable has the right capacity (always true)
template <typename TYPE, typename SIZE_TYPE>
VECMEM_HOST_AND_DEVICE constexpr bool device_capacity_matches(
    SIZE_TYPE, const typename device_type<type::scalar<TYPE>>::type&) {
    return true;
}

/// Check whether a vector variable has the right capacity
template <typename TYPE>
VECMEM_HOST_AND_DEVICE constexpr bool device_capacity_matches(
    const typename device_type<type::vector<TYPE>>::type::size_type capacity,
    const typename device_type<type::vector<TYPE>>::type& variable) {

    return (capacity == variable.capacity());
}

/// Check whether a jagged vector variable has the right capacity
template <typename TYPE>
VECMEM_HOST_AND_DEVICE constexpr bool device_capacity_matches(
    const typename device_type<
        type::jagged_vector<TYPE>>::type::value_type::size_type capacity,
    const typename device_type<type::jagged_vector<TYPE>>::type& variable) {

    return (capacity == variable.capacity());
}

/// Terminal node for @c vecmem::edm::details::device_capacities_match
template <typename... VARTYPES>
VECMEM_HOST_AND_DEVICE constexpr bool device_capacities_match(
    const typename view<schema<VARTYPES...>>::size_type,
    const tuple<typename device_type<VARTYPES>::type...>&,
    std::index_sequence<>) {

    return true;
}

/// Helper function checking the capacities of all variables against a reference
///
/// This is only used in assertions in the code, to make sure that there would
/// be no coding mistakes with setting up the individual variables.
///
template <typename... VARTYPES, std::size_t INDEX, std::size_t... INDICES>
VECMEM_HOST_AND_DEVICE constexpr bool device_capacities_match(
    const typename view<schema<VARTYPES...>>::size_type capacity,
    const tuple<typename device_type<VARTYPES>::type...>& variables,
    std::index_sequence<INDEX, INDICES...>) {

    // Check the capacities recursively.
    return device_capacity_matches<
               typename tuple_element_t<INDEX, tuple<VARTYPES...>>::type>(
               capacity, get<INDEX>(variables)) &&
           device_capacities_match<VARTYPES...>(
               capacity, variables, std::index_sequence<INDICES...>{});
}

/// Helper trait for setting the @c m_size variable of a device object
template <typename SCHEMA, bool HAS_JAGGED_VECTOR>
struct device_size_pointer;

/// Helper trait for setting the @c m_size variable of a device object
/// (for a "jagged schema")
template <typename SCHEMA>
struct device_size_pointer<SCHEMA, true> {
    VECMEM_HOST_AND_DEVICE static constexpr typename view<SCHEMA>::size_pointer
    get(const typename view<SCHEMA>::memory_view_type&) {

        return nullptr;
    }
};

/// Helper trait for setting the @c m_size variable of a device object
/// (for a "non-jagged schema")
template <typename SCHEMA>
struct device_size_pointer<SCHEMA, false> {
    VECMEM_HOST_AND_DEVICE static constexpr typename view<SCHEMA>::size_pointer
    get(const typename view<SCHEMA>::memory_view_type& v) {

        // A sanity check.
        assert((v.ptr() == nullptr) ||
               (v.size() == sizeof(typename view<SCHEMA>::size_type)));
        // Do a forceful conversion.
        return reinterpret_cast<typename view<SCHEMA>::size_pointer>(v.ptr());
    }
};

}  // namespace details
}  // namespace edm
}  // namespace vecmem
