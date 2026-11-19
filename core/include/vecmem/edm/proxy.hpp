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

// System include(s).
#include <type_traits>

namespace vecmem {
namespace edm {

/// Technical base type for @c proxy<schema<VARTYPES...>,PDOMAIN,PACCESS,PTYPE>
template <typename T, details::proxy_domain PDOMAIN,
          details::proxy_access PACCESS, details::proxy_type PTYPE>
class proxy;

/// Structure-of-Arrays element proxy
///
/// This class implements a "view" of a single element in an SoA container.
///
/// @tparam ...VARTYPES The variable types to store in the proxy object
/// @tparam PDOMAIN     The "domain" of the proxy (host or device)
/// @tparam PACCESS     The access mode of the proxy (const or non-const)
/// @tparam PTYPE       The type of the proxy (reference or standalone)
///
template <typename... VARTYPES, details::proxy_domain PDOMAIN,
          details::proxy_access PACCESS, details::proxy_type PTYPE>
class proxy<schema<VARTYPES...>, PDOMAIN, PACCESS, PTYPE> {

public:
    /// The schema describing the host's payload
    using schema_type = schema<VARTYPES...>;
    /// The type of the proxy (host or device)
    static constexpr details::proxy_domain proxy_domain = PDOMAIN;
    /// The access mode of the proxy (const or non-const)
    static constexpr details::proxy_access access_type = PACCESS;
    /// The type of the proxy (reference or standalone)
    static constexpr details::proxy_type proxy_type = PTYPE;
    /// The tuple type holding all of the the proxied variables
    using tuple_type = tuple<typename details::proxy_var_type<
        VARTYPES, proxy_domain, access_type, proxy_type>::type...>;

    /// @name Constructors and assignment operators
    /// @{

    /// Default constructor (only for standalone proxies)
    template <details::proxy_type OPTYPE = proxy_type,
              std::enable_if_t<OPTYPE == details::proxy_type::standalone,
                               bool> = true>
    VECMEM_HOST_AND_DEVICE proxy();

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

    /// Default copy constructor
    proxy(const proxy&) = default;

    /// Default move constructor
    proxy(proxy&&) noexcept = default;

    /// Copy constructor
    ///
    /// @tparam OPDOMAIN The domain of the other proxy
    /// @tparam OPACCESS The access mode of the other proxy
    /// @tparam OPTYPE   The type of the other proxy
    ///
    /// @param other The proxy to copy
    ///
    template <typename... OVARTYPES, details::proxy_domain OPDOMAIN,
              details::proxy_access OPACCESS, details::proxy_type OPTYPE>
    VECMEM_HOST_AND_DEVICE proxy(
        const proxy<schema<OVARTYPES...>, OPDOMAIN, OPACCESS, OPTYPE>& other);

    /// Construct a proxy from a list of variables
    ///
    /// This is mainly meant for "standalone proxies", but technically would
    /// work for any type of a proxy.
    ///
    /// @param data The list of variables to proxy
    ///
    VECMEM_HOST_AND_DEVICE proxy(
        typename details::proxy_var_type<VARTYPES, proxy_domain, access_type,
                                         proxy_type>::type... data);

    /// Copy assignment operator from an identical type
    ///
    /// @param other The proxy to copy
    ///
    /// @return A reference to the proxy object
    ///
    VECMEM_HOST_AND_DEVICE
    proxy& operator=(const proxy& other);

    /// Copy assignment operator from a different type
    ///
    /// @tparam OVARTYPES The variable types of the other proxy
    /// @tparam OPDOMAIN  The domain of the other proxy
    /// @tparam OPACCESS  The access mode of the other proxy
    /// @tparam OPTYPE    The type of the other proxy
    ///
    /// @param other The proxy to copy
    ///
    /// @return A reference to the proxy object
    ///
    template <typename... OVARTYPES, details::proxy_domain OPDOMAIN,
              details::proxy_access OPACCESS, details::proxy_type OPTYPE>
    VECMEM_HOST_AND_DEVICE proxy<schema<VARTYPES...>, PDOMAIN, PACCESS, PTYPE>&
    operator=(
        const proxy<schema<OVARTYPES...>, OPDOMAIN, OPACCESS, OPTYPE>& other);

    /// @}

    /// @name Variable accessor functions
    /// @{

    /// Get a specific variable (non-const)
    template <std::size_t INDEX>
    VECMEM_HOST_AND_DEVICE
        typename details::proxy_var_type_at<INDEX, PDOMAIN, PACCESS, PTYPE,
                                            VARTYPES...>::return_type
        get();
    /// Get a specific variable (const)
    template <std::size_t INDEX>
    VECMEM_HOST_AND_DEVICE
        typename details::proxy_var_type_at<INDEX, PDOMAIN, PACCESS, PTYPE,
                                            VARTYPES...>::const_return_type
        get() const;

    /// @}

    /// @name Function(s) meant for internal use by other VecMem types
    /// @{

    /// Direct (non-const) access to the underlying tuple of variables
    VECMEM_HOST_AND_DEVICE
    tuple_type& variables();
    /// Direct (const) access to the underlying tuple of variables
    VECMEM_HOST_AND_DEVICE
    const tuple_type& variables() const;

    /// @}

private:
    /// The tuple holding all of the individual proxy objects
    tuple_type m_data;

};  // class proxy

#ifdef __cpp_concepts
namespace concepts {
/// Concept for a type being of a proxy type
template <typename T>
concept proxy =
    (std::is_same_v<typename T::proxy_domain, details::proxy_domain> &&
     std::is_same_v<typename T::access_type, details::proxy_access> &&
     std::is_same_v<typename T::proxy_type, details::proxy_type>);
}  // namespace concepts
#endif  // __cpp_concepts

}  // namespace edm
}  // namespace vecmem

// Include the implementation.
#include "vecmem/edm/impl/proxy.ipp"
