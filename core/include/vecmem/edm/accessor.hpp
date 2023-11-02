/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/details/accessor_traits.hpp"
#include "vecmem/edm/device.hpp"
#include "vecmem/edm/host.hpp"
#include "vecmem/edm/schema.hpp"

// System include(s).
#include <cstddef>

namespace vecmem::edm {

/// Accessor for a specific variable in an SoA host container
///
/// Such accessors can be "named", making the access to the elements of an SoA
/// container easier to read/write.
///
/// @tparam INDEX The index of the variable to access
///
template <std::size_t INDEX>
struct accessor {

    /// (Non-const) Access a specific variable from an SoA host container
    template <typename... VARTYPES>
    constexpr
        typename details::accessor_host_type_at<INDEX, VARTYPES...>::return_type
        operator()(host<schema<VARTYPES...>>& obj) const;

    /// (const) Access a specific variable from an SoA host container
    template <typename... VARTYPES>
    constexpr
        typename details::accessor_host_type_at<INDEX,
                                                VARTYPES...>::const_return_type
        operator()(const host<schema<VARTYPES...>>& obj) const;

    /// (Non-const) Access a specific variable from an SoA device container
    template <typename... VARTYPES>
    constexpr
        typename details::accessor_device_type_at<INDEX,
                                                  VARTYPES...>::return_type
        operator()(device<schema<VARTYPES...>>& obj) const;

    /// (const) Access a specific variable from an SoA device container
    template <typename... VARTYPES>
    constexpr typename details::accessor_device_type_at<
        INDEX, VARTYPES...>::const_return_type
    operator()(const device<schema<VARTYPES...>>& obj) const;

};  // struct accessor

}  // namespace vecmem::edm

// Include the implementation.
#include "vecmem/edm/impl/accessor.ipp"
