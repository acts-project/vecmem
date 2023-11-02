/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/details/accessor_traits.hpp"

namespace vecmem::edm {

template <std::size_t INDEX>
template <typename... VARTYPES>
VECMEM_HOST constexpr
    typename details::accessor_host_type_at<INDEX, VARTYPES...>::return_type
    accessor<INDEX>::get(host<schema<VARTYPES...>>& obj) {

    return details::accessor_host_get_at<INDEX, VARTYPES...>::get(
        obj.template get<INDEX>());
}

template <std::size_t INDEX>
template <typename... VARTYPES>
VECMEM_HOST constexpr
    typename details::accessor_host_type_at<INDEX,
                                            VARTYPES...>::const_return_type
    accessor<INDEX>::get(const host<schema<VARTYPES...>>& obj) {

    return details::accessor_host_get_at<INDEX, VARTYPES...>::get(
        obj.template get<INDEX>());
}

template <std::size_t INDEX>
template <typename... VARTYPES>
VECMEM_HOST_AND_DEVICE constexpr
    typename details::accessor_device_type_at<INDEX, VARTYPES...>::return_type
    accessor<INDEX>::get(device<schema<VARTYPES...>>& obj) {

    return details::accessor_device_get_at<INDEX, VARTYPES...>::get(
        obj.template get<INDEX>());
}

template <std::size_t INDEX>
template <typename... VARTYPES>
VECMEM_HOST_AND_DEVICE constexpr
    typename details::accessor_device_type_at<INDEX,
                                              VARTYPES...>::const_return_type
    accessor<INDEX>::get(const device<schema<VARTYPES...>>& obj) {

    return details::accessor_device_get_at<INDEX, VARTYPES...>::get(
        obj.template get<INDEX>());
}

}  // namespace vecmem::edm
