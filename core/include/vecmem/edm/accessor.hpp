/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/details/accessor_device_traits.hpp"
#include "vecmem/edm/details/schema_traits.hpp"
#include "vecmem/edm/device.hpp"
#include "vecmem/edm/schema.hpp"
#include "vecmem/utils/types.hpp"

#if __cplusplus >= 201700L
#include "vecmem/edm/details/accessor_host_traits.hpp"
#include "vecmem/edm/host.hpp"
#endif  // __cplusplus >= 201700L

// System include(s).
#include <cstddef>

namespace vecmem {
namespace edm {

/// Generic accessor template
template <std::size_t INDEX, typename... VARTYPES>
struct accessor {};

/// Accessor for a specific variable in an SoA host container
///
/// Such accessors can be "named", making the access to the elements of an SoA
/// container easier to read/write.
///
/// @tparam INDEX The index of the variable to access
///
template <std::size_t INDEX, typename... VARTYPES>
struct accessor<INDEX, schema<VARTYPES...>> {

#if __cplusplus >= 201700L
    /// @name Host container accessor function(s)
    /// @{

    /// (Non-const) Access a specific variable from an SoA host container
    VECMEM_HOST static constexpr
        typename details::accessor_host_type_at<INDEX, VARTYPES...>::return_type
        get(host<schema<VARTYPES...>>& obj);

    /// (const) Access a specific variable from an SoA host container
    VECMEM_HOST static constexpr
        typename details::accessor_host_type_at<INDEX,
                                                VARTYPES...>::const_return_type
        get(const host<schema<VARTYPES...>>& obj);

    /// @}
#endif  // __cplusplus >= 201700L

    /// @name Device container accessor function(s)
    /// @{

    /// (Non-const) Access a specific variable from an SoA device container
    VECMEM_HOST_AND_DEVICE static constexpr
        typename details::accessor_device_type_at<INDEX,
                                                  VARTYPES...>::return_type
        get(device<schema<VARTYPES...>>& obj);

    /// (const) Access a specific variable from an SoA device container
    VECMEM_HOST_AND_DEVICE static constexpr
        typename details::accessor_device_type_at<
            INDEX, VARTYPES...>::const_return_type
        get(const device<schema<VARTYPES...>>& obj);

    /// (Non-const) Access a specific variable from an (const) SoA device
    /// container
    VECMEM_HOST_AND_DEVICE static constexpr
        typename details::accessor_device_type_at<
            INDEX,
            typename type::details::add_const<VARTYPES>::type...>::return_type
        get(device<
            schema<typename type::details::add_const<VARTYPES>::type...>>& obj);

    /// (const) Access a specific variable from an (const) SoA device container
    VECMEM_HOST_AND_DEVICE static constexpr
        typename details::accessor_device_type_at<
            INDEX, typename type::details::add_const<
                       VARTYPES>::type...>::const_return_type
        get(const device<
            schema<typename type::details::add_const<VARTYPES>::type...>>& obj);

    /// @}

};  // struct accessor

}  // namespace edm
}  // namespace vecmem

// Include the implementation.
#include "vecmem/edm/impl/accessor.ipp"
