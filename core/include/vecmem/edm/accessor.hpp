/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/details/device_traits.hpp"
#include "vecmem/edm/details/schema_traits.hpp"
#include "vecmem/edm/device.hpp"
#include "vecmem/edm/schema.hpp"
#include "vecmem/utils/types.hpp"

#if __cplusplus >= 201700L
#include "vecmem/edm/details/host_traits.hpp"
#include "vecmem/edm/host.hpp"
#endif  // __cplusplus >= 201700L

// System include(s).
#include <cstddef>

namespace vecmem {
namespace edm {

/// Technical base type for @c accessor<INDEX,schema<VARTYPES...>>
template <std::size_t INDEX, typename... VARTYPES>
struct accessor {};

/// Accessor for a specific variable in an SoA container
///
/// Such accessors can be "named", making the access to the elements of an SoA
/// container easier to read/write.
///
/// @tparam INDEX The index of the variable to access
/// @tparam VARTYPES The variable types of the SoA container
///
template <std::size_t INDEX, typename... VARTYPES>
struct accessor<INDEX, schema<VARTYPES...>> {

    /// Schema that this accessor operates on
    using schema_type = schema<VARTYPES...>;
    /// Non-const device type handled by this accessor
    using device_type = device<schema_type>;
    /// Non-const return type for device access
    using device_return_type =
        typename details::device_type_at<INDEX, VARTYPES...>::return_type;
    /// Const return type for device access
    using device_const_return_type =
        typename details::device_type_at<INDEX, VARTYPES...>::const_return_type;

    /// Constant schema that this accessor operates on
    using const_schema_type = typename details::add_const<schema_type>::type;
    /// Constant device type handled by this constructor
    using const_device_type = device<const_schema_type>;
    /// Non-const return type for constant device access
    using const_device_return_type = typename details::device_type_at<
        INDEX,
        typename type::details::add_const<VARTYPES>::type...>::return_type;
    /// Const return type for constant device access
    using const_device_const_return_type = typename details::device_type_at<
        INDEX, typename type::details::add_const<VARTYPES>::type...>::
        const_return_type;

    /// @name Device container accessor function(s)
    /// @{

    /// (Non-const) Access a specific variable from an SoA device container
    VECMEM_HOST_AND_DEVICE static constexpr device_return_type get(
        device_type& obj);

    /// (const) Access a specific variable from an SoA device container
    VECMEM_HOST_AND_DEVICE static constexpr device_const_return_type get(
        const device_type& obj);

    /// (Non-const) Access a specific variable from an (const) SoA device
    /// container
    VECMEM_HOST_AND_DEVICE static constexpr const_device_return_type get(
        const_device_type& obj);

    /// (const) Access a specific variable from an (const) SoA device container
    VECMEM_HOST_AND_DEVICE static constexpr const_device_const_return_type get(
        const const_device_type& obj);

    /// @}

#if __cplusplus >= 201700L
    /// Host type handled by this accessor
    using host_type = host<schema_type>;
    /// Non-const return type for host access
    using host_return_type =
        typename details::host_type_at<INDEX, VARTYPES...>::return_type;
    /// Const return type for host access
    using host_const_return_type =
        typename details::host_type_at<INDEX, VARTYPES...>::const_return_type;

    /// @name Host container accessor function(s)
    /// @{

    /// (Non-const) Access a specific variable from an SoA host container
    VECMEM_HOST static constexpr host_return_type get(host_type& obj);

    /// (const) Access a specific variable from an SoA host container
    VECMEM_HOST static constexpr host_const_return_type get(
        const host_type& obj);

    /// @}
#endif  // __cplusplus >= 201700L

};  // struct accessor

}  // namespace edm
}  // namespace vecmem

// Include the implementation.
#include "vecmem/edm/impl/accessor.ipp"
