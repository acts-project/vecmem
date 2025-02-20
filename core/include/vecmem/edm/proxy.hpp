/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/details/proxy_traits.hpp"
#include "vecmem/edm/schema.hpp"
#include "vecmem/utils/tuple.hpp"
#include "vecmem/utils/types.hpp"

namespace vecmem {
namespace edm {

/// Technical base type for @c proxy<schema<VARTYPES...>,PTYPE,CTYPE>
template <typename T, details::proxy_domain PDOMAIN,
          details::proxy_access PACCESS>
class proxy;

/// Structure-of-Arrays element proxy
///
/// This class implements a "view" of a single element in an SoA container.
///
/// @tparam ...VARTYPES The variable types to store in the proxy object
/// @tparam PTYPE       The type of the proxy (host or device)
/// @tparam CTYPE       The access mode of the proxy (const or non-const)
///
template <typename... VARTYPES, details::proxy_domain PDOMAIN,
          details::proxy_access PACCESS>
class proxy<schema<VARTYPES...>, PDOMAIN, PACCESS> {

public:
    /// The schema describing the host's payload
    using schema_type = schema<VARTYPES...>;
    /// The type of the proxy (host or device)
    static constexpr details::proxy_domain proxy_domain = PDOMAIN;
    /// The access mode of the proxy (const or non-const)
    static constexpr details::proxy_access access_type = PACCESS;
    /// The tuple type holding all of the the proxied variables
    using tuple_type =
        tuple<typename details::proxy_var_type<VARTYPES, proxy_domain,
                                               access_type>::type...>;

    /// @name Constructors and assignment operators
    /// @{

    /// Constructor of a non-const proxy on top of a parent container
    ///
    /// @tparam PARENT The type of the parent container
    /// @param  parent The parent container to create a proxy for
    /// @param  index  The index of the element to proxy
    ///
    template <typename PARENT>
    VECMEM_HOST_AND_DEVICE proxy(PARENT& parent,
                                 typename PARENT::size_type index);

    /// Constructor of a const proxy on top of a parent container
    ///
    /// @tparam PARENT The type of the parent container
    /// @param  parent The parent container to create a proxy for
    /// @param  index  The index of the element to proxy
    ///
    template <typename PARENT>
    VECMEM_HOST_AND_DEVICE proxy(const PARENT& parent,
                                 typename PARENT::size_type index);

    /// @}

    /// @name Variable accessor functions
    /// @{

    /// Get a specific variable (non-const)
    template <std::size_t INDEX>
    VECMEM_HOST_AND_DEVICE
        typename details::proxy_var_type_at<INDEX, PDOMAIN, PACCESS,
                                            VARTYPES...>::return_type
        get();
    /// Get a specific variable (const)
    template <std::size_t INDEX>
    VECMEM_HOST_AND_DEVICE
        typename details::proxy_var_type_at<INDEX, PDOMAIN, PACCESS,
                                            VARTYPES...>::const_return_type
        get() const;

    /// @}

private:
    /// The tuple holding all of the individual proxy objects
    tuple_type m_data;

};  // class proxy

}  // namespace edm
}  // namespace vecmem

// Include the implementation.
#include "vecmem/edm/impl/proxy.ipp"
