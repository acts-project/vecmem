/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem {
namespace edm {

template <typename... VARTYPES, details::proxy_domain PDOMAIN,
          details::proxy_access PACCESS, details::proxy_type PTYPE>
template <typename PARENT>
VECMEM_HOST_AND_DEVICE proxy<schema<VARTYPES...>, PDOMAIN, PACCESS,
                             PTYPE>::proxy(PARENT& parent,
                                           typename PARENT::size_type index)
    : m_data{details::proxy_data_creator<schema<VARTYPES...>, PDOMAIN, PACCESS,
                                         PTYPE>::make(index, parent)} {

    static_assert(PACCESS == details::proxy_access::non_constant,
                  "This constructor is meant for non-const proxies.");
}

template <typename... VARTYPES, details::proxy_domain PDOMAIN,
          details::proxy_access PACCESS, details::proxy_type PTYPE>
template <typename PARENT>
VECMEM_HOST_AND_DEVICE proxy<schema<VARTYPES...>, PDOMAIN, PACCESS,
                             PTYPE>::proxy(const PARENT& parent,
                                           typename PARENT::size_type index)
    : m_data{details::proxy_data_creator<schema<VARTYPES...>, PDOMAIN, PACCESS,
                                         PTYPE>::make(index, parent)} {

    static_assert(PACCESS == details::proxy_access::constant,
                  "This constructor is meant for constant proxies.");
}

template <typename... VARTYPES, details::proxy_domain PDOMAIN,
          details::proxy_access PACCESS, details::proxy_type PTYPE>
template <std::size_t INDEX>
VECMEM_HOST_AND_DEVICE
    typename details::proxy_var_type_at<INDEX, PDOMAIN, PACCESS, PTYPE,
                                        VARTYPES...>::return_type
    proxy<schema<VARTYPES...>, PDOMAIN, PACCESS, PTYPE>::get() {

    return vecmem::get<INDEX>(m_data);
}

template <typename... VARTYPES, details::proxy_domain PDOMAIN,
          details::proxy_access PACCESS, details::proxy_type PTYPE>
template <std::size_t INDEX>
VECMEM_HOST_AND_DEVICE
    typename details::proxy_var_type_at<INDEX, PDOMAIN, PACCESS, PTYPE,
                                        VARTYPES...>::const_return_type
    proxy<schema<VARTYPES...>, PDOMAIN, PACCESS, PTYPE>::get() const {

    return vecmem::get<INDEX>(m_data);
}

}  // namespace edm
}  // namespace vecmem
