/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem {
namespace edm {

template <std::size_t INDEX, typename... VARTYPES>
VECMEM_HOST_AND_DEVICE constexpr auto accessor<INDEX, schema<VARTYPES...>>::get(
    device_type& obj) -> device_return_type {

    return obj.template get<INDEX>();
}

template <std::size_t INDEX, typename... VARTYPES>
VECMEM_HOST_AND_DEVICE constexpr auto accessor<INDEX, schema<VARTYPES...>>::get(
    const device_type& obj) -> device_const_return_type {

    return obj.template get<INDEX>();
}

template <std::size_t INDEX, typename... VARTYPES>
VECMEM_HOST_AND_DEVICE constexpr auto accessor<INDEX, schema<VARTYPES...>>::get(
    const_device_type& obj) -> const_device_return_type {

    return obj.template get<INDEX>();
}

template <std::size_t INDEX, typename... VARTYPES>
VECMEM_HOST_AND_DEVICE constexpr auto accessor<INDEX, schema<VARTYPES...>>::get(
    const const_device_type& obj) -> const_device_const_return_type {

    return obj.template get<INDEX>();
}

#if __cplusplus >= 201700L

template <std::size_t INDEX, typename... VARTYPES>
VECMEM_HOST constexpr auto accessor<INDEX, schema<VARTYPES...>>::get(
    host_type& obj) -> host_return_type {

    return obj.template get<INDEX>();
}

template <std::size_t INDEX, typename... VARTYPES>
VECMEM_HOST constexpr auto accessor<INDEX, schema<VARTYPES...>>::get(
    const host_type& obj) -> host_const_return_type {

    return obj.template get<INDEX>();
}

#endif  // __cplusplus >= 201700L

}  // namespace edm
}  // namespace vecmem
