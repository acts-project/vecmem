/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem {
namespace edm {

#if __cplusplus >= 201700L

template <std::size_t INDEX, typename... VARTYPES>
VECMEM_HOST constexpr
    typename details::accessor_host_type_at<INDEX, VARTYPES...>::return_type
    accessor<INDEX, schema<VARTYPES...>>::get(host<schema<VARTYPES...>>& obj) {

    return details::accessor_host_get_at<INDEX, VARTYPES...>::get(
        obj.template get<INDEX>());
}

template <std::size_t INDEX, typename... VARTYPES>
VECMEM_HOST constexpr
    typename details::accessor_host_type_at<INDEX,
                                            VARTYPES...>::const_return_type
    accessor<INDEX, schema<VARTYPES...>>::get(
        const host<schema<VARTYPES...>>& obj) {

    return details::accessor_host_get_at<INDEX, VARTYPES...>::get(
        obj.template get<INDEX>());
}

#endif  // __cplusplus >= 201700L

template <std::size_t INDEX, typename... VARTYPES>
VECMEM_HOST_AND_DEVICE constexpr
    typename details::accessor_device_type_at<INDEX, VARTYPES...>::return_type
    accessor<INDEX, schema<VARTYPES...>>::get(
        device<schema<VARTYPES...>>& obj) {

    return details::accessor_device_get_at<INDEX, VARTYPES...>::get(
        obj.template get<INDEX>());
}

template <std::size_t INDEX, typename... VARTYPES>
VECMEM_HOST_AND_DEVICE constexpr
    typename details::accessor_device_type_at<INDEX,
                                              VARTYPES...>::const_return_type
    accessor<INDEX, schema<VARTYPES...>>::get(
        const device<schema<VARTYPES...>>& obj) {

    return details::accessor_device_get_at<INDEX, VARTYPES...>::get(
        obj.template get<INDEX>());
}

template <std::size_t INDEX, typename... VARTYPES>
VECMEM_HOST_AND_DEVICE constexpr typename details::accessor_device_type_at<
    INDEX, typename type::details::add_const<VARTYPES>::type...>::return_type
accessor<INDEX, schema<VARTYPES...>>::get(
    device<schema<typename type::details::add_const<VARTYPES>::type...>>& obj) {

    return details::accessor_device_get_at<
        INDEX, typename type::details::add_const<VARTYPES>::type...>::
        get(obj.template get<INDEX>());
}

template <std::size_t INDEX, typename... VARTYPES>
VECMEM_HOST_AND_DEVICE constexpr typename details::accessor_device_type_at<
    INDEX,
    typename type::details::add_const<VARTYPES>::type...>::const_return_type
accessor<INDEX, schema<VARTYPES...>>::get(
    const device<schema<typename type::details::add_const<VARTYPES>::type...>>&
        obj) {

    return details::accessor_device_get_at<
        INDEX, typename type::details::add_const<VARTYPES>::type...>::
        get(obj.template get<INDEX>());
}

}  // namespace edm
}  // namespace vecmem
