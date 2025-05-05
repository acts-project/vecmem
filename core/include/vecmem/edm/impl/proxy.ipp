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
template <typename... OVARTYPES, details::proxy_domain OPDOMAIN,
          details::proxy_access OPACCESS, details::proxy_type OPTYPE>
VECMEM_HOST_AND_DEVICE
proxy<schema<VARTYPES...>, PDOMAIN, PACCESS, PTYPE>::proxy(
    const proxy<schema<OVARTYPES...>, OPDOMAIN, OPACCESS, OPTYPE>& other)
    : m_data(other.variables()) {}

template <typename... VARTYPES, details::proxy_domain PDOMAIN,
          details::proxy_access PACCESS, details::proxy_type PTYPE>
VECMEM_HOST_AND_DEVICE
proxy<schema<VARTYPES...>, PDOMAIN, PACCESS, PTYPE>::proxy(
    typename details::proxy_var_type<VARTYPES, proxy_domain, access_type,
                                     proxy_type>::type... data)
    : m_data(data...) {}

template <typename... VARTYPES, details::proxy_domain PDOMAIN,
          details::proxy_access PACCESS, details::proxy_type PTYPE>
VECMEM_HOST_AND_DEVICE proxy<schema<VARTYPES...>, PDOMAIN, PACCESS, PTYPE>&
proxy<schema<VARTYPES...>, PDOMAIN, PACCESS, PTYPE>::operator=(
    const proxy& other) {

    if (this != &other) {
        details::proxy_tuple_copy(m_data, other.m_data);
    }
    return *this;
}

template <typename... VARTYPES, details::proxy_domain PDOMAIN,
          details::proxy_access PACCESS, details::proxy_type PTYPE>
template <typename... OVARTYPES, details::proxy_domain OPDOMAIN,
          details::proxy_access OPACCESS, details::proxy_type OPTYPE>
VECMEM_HOST_AND_DEVICE proxy<schema<VARTYPES...>, PDOMAIN, PACCESS, PTYPE>&
proxy<schema<VARTYPES...>, PDOMAIN, PACCESS, PTYPE>::operator=(
    const proxy<schema<OVARTYPES...>, OPDOMAIN, OPACCESS, OPTYPE>& other) {

    m_data = other.variables();
    return *this;
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

template <typename... VARTYPES, details::proxy_domain PDOMAIN,
          details::proxy_access PACCESS, details::proxy_type PTYPE>
VECMEM_HOST_AND_DEVICE auto
proxy<schema<VARTYPES...>, PDOMAIN, PACCESS, PTYPE>::variables() const
    -> const tuple_type& {

    return m_data;
}

template <typename... VARTYPES, details::proxy_domain PDOMAIN,
          details::proxy_access PACCESS, details::proxy_type PTYPE>
VECMEM_HOST_AND_DEVICE auto
proxy<schema<VARTYPES...>, PDOMAIN, PACCESS, PTYPE>::variables()
    -> tuple_type& {

    return m_data;
}

}  // namespace edm
}  // namespace vecmem
