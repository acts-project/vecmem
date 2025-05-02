/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/jagged_device_vector.hpp"
#include "vecmem/edm/schema.hpp"
#include "vecmem/utils/tuple.hpp"
#if __cplusplus >= 201700L
#include "vecmem/containers/jagged_vector.hpp"
#include "vecmem/containers/vector.hpp"
#endif  // __cplusplus >= 201700L

// System include(s).
#include <tuple>
#include <type_traits>

namespace vecmem {
namespace edm {
namespace details {

/// @brief The "domain" of the proxy to use for a given container variable
enum class proxy_domain {
    /// Proxy for a host container element
    host,
    /// Proxy for a device container element
    device
};

/// @brief The "access type" of the proxy to use for a given container variable
enum class proxy_access {
    /// Proxy for a non-const container element
    non_constant,
    /// Proxy for a const container element
    constant
};

/// @brief The "type" of the proxy object
enum class proxy_type {
    /// The proxy references an SoA container's element
    reference,
    /// The "proxy" is standalone, not connected to a container
    standalone
};

/// @name Traits for the proxied variable types
/// @{

/// Technical base class for the @c proxy_var_type traits
template <typename VTYPE, proxy_domain PDOMAIN, proxy_access PACCESS,
          proxy_type PTYPE>
struct proxy_var_type;

/// Constant access to a scalar variable (both host and device)
template <typename VTYPE, proxy_domain PDOMAIN>
struct proxy_var_type<type::scalar<VTYPE>, PDOMAIN, proxy_access::constant,
                      proxy_type::reference> {

    /// The scalar is kept by constant lvalue reference in the proxy
    using type = std::add_lvalue_reference_t<std::add_const_t<VTYPE>>;
    /// It is returned as a const reference even on non-const access
    using return_type = type;
    /// It is returned as a const reference on const access
    using const_return_type = return_type;

    /// Helper function constructing a scalar proxy variable
    template <typename ITYPE>
    VECMEM_HOST_AND_DEVICE static type make(ITYPE, return_type variable) {
        return variable;
    }
};

/// Non-const access to a scalar variable (both host and device)
template <typename VTYPE, proxy_domain PDOMAIN>
struct proxy_var_type<type::scalar<VTYPE>, PDOMAIN, proxy_access::non_constant,
                      proxy_type::reference> {

    /// The scalar is kept by lvalue reference in the proxy
    using type = std::add_lvalue_reference_t<VTYPE>;
    /// It is returned as a non-const lvalue reference on non-const access
    using return_type = type;
    /// It is returned as a const reference on const access
    using const_return_type =
        std::add_lvalue_reference_t<std::add_const_t<VTYPE>>;

    /// Helper function constructing a scalar proxy variable
    template <typename ITYPE>
    VECMEM_HOST_AND_DEVICE static type make(ITYPE, return_type variable) {
        return variable;
    }
};

/// Standalone scalar variable (both host and device, const and non-const)
template <typename VTYPE, proxy_domain PDOMAIN, proxy_access PACCESS>
struct proxy_var_type<type::scalar<VTYPE>, PDOMAIN, PACCESS,
                      proxy_type::standalone> {

    /// The scalar is kept by value in the proxy
    using type = std::remove_cv_t<VTYPE>;
    /// It is returned as a const reference even on non-const access
    using return_type = std::add_lvalue_reference_t<type>;
    /// It is returned as a const reference on const access
    using const_return_type =
        std::add_lvalue_reference_t<std::add_const_t<type>>;

    /// Helper function constructing a scalar proxy variable
    template <typename ITYPE>
    VECMEM_HOST_AND_DEVICE static type make(ITYPE, const_return_type variable) {
        return variable;
    }
};

/// Constant access to a vector variable (both host and device)
template <typename VTYPE, proxy_domain PDOMAIN>
struct proxy_var_type<type::vector<VTYPE>, PDOMAIN, proxy_access::constant,
                      proxy_type::reference> {

    /// Vector elements are kept by value in the proxy
    using type = std::add_lvalue_reference_t<std::add_const_t<VTYPE>>;
    /// They are returned as a const reference even on non-const access
    using return_type = type;
    /// They are returned as a const reference on const access
    using const_return_type = return_type;

    /// Helper function constructing a vector proxy variable
    template <typename ITYPE, typename VECTYPE>
    VECMEM_HOST_AND_DEVICE static type make(ITYPE i, const VECTYPE& vec) {

        return vec.at(i);
    }
};

/// Non-const access to a vector variable (both host and device)
template <typename VTYPE, proxy_domain PDOMAIN>
struct proxy_var_type<type::vector<VTYPE>, PDOMAIN, proxy_access::non_constant,
                      proxy_type::reference> {

    /// Vector elements are kept by lvalue reference in the proxy
    using type = std::add_lvalue_reference_t<VTYPE>;
    /// They are returned as a non-const lvalue reference on non-const access
    using return_type = type;
    /// They are returned as a const reference on const access
    using const_return_type =
        std::add_lvalue_reference_t<std::add_const_t<VTYPE>>;

    /// Helper function constructing a vector proxy variable
    template <typename ITYPE, typename VECTYPE>
    VECMEM_HOST_AND_DEVICE static type make(ITYPE i, VECTYPE& vec) {

        return vec.at(i);
    }
};

/// Standalone vector variable (both host and device, const and non-const)
template <typename VTYPE, proxy_domain PDOMAIN, proxy_access PACCESS>
struct proxy_var_type<type::vector<VTYPE>, PDOMAIN, PACCESS,
                      proxy_type::standalone> {

    /// The scalar is kept by value in the proxy
    using type = std::remove_cv_t<VTYPE>;
    /// It is returned as a const reference even on non-const access
    using return_type = std::add_lvalue_reference_t<type>;
    /// It is returned as a const reference on const access
    using const_return_type =
        std::add_lvalue_reference_t<std::add_const_t<type>>;

    /// Helper function constructing a vector proxy variable
    template <typename ITYPE, typename VECTYPE>
    VECMEM_HOST_AND_DEVICE static type make(ITYPE i, const VECTYPE& vec) {

        return vec.at(i);
    }
};

/// Constant access to a jagged vector variable from a device container
template <typename VTYPE>
struct proxy_var_type<type::jagged_vector<VTYPE>, proxy_domain::device,
                      proxy_access::constant, proxy_type::reference> {

    /// Jagged vector elements are kept by constant device vectors in the proxy
    using type = device_vector<std::add_const_t<VTYPE>>;
    /// They are returned as a const reference to the device vector even in
    /// non-const access
    using return_type = std::add_lvalue_reference_t<std::add_const_t<type>>;
    /// They are returned as a const reference to the device vector on const
    /// access
    using const_return_type = return_type;

    /// Helper function constructing a vector proxy variable
    VECMEM_HOST_AND_DEVICE
    static type make(
        typename jagged_device_vector<std::add_const_t<VTYPE>>::size_type i,
        const jagged_device_vector<std::add_const_t<VTYPE>>& vec) {

        return vec.at(i);
    }
};

/// Non-const access to a jagged vector variable from a device container
template <typename VTYPE>
struct proxy_var_type<type::jagged_vector<VTYPE>, proxy_domain::device,
                      proxy_access::non_constant, proxy_type::reference> {

    /// Jagged vector elements are kept by non-const device vectors in the proxy
    using type = device_vector<VTYPE>;
    /// They are returned as non-const lvalue references to the non-const device
    /// vector in non-const access
    using return_type = std::add_lvalue_reference_t<type>;
    /// They are returned as const references to the non-const device vector in
    /// const access
    using const_return_type = std::add_lvalue_reference_t<
        std::add_const_t<device_vector<std::add_const_t<VTYPE>>>>;

    /// Helper function constructing a vector proxy variable
    VECMEM_HOST_AND_DEVICE
    static type make(typename jagged_device_vector<VTYPE>::size_type i,
                     jagged_device_vector<VTYPE>& vec) {

        return vec.at(i);
    }
};

#if __cplusplus >= 201700L

/// Constant access to a jagged vector variable from a host container
template <typename VTYPE>
struct proxy_var_type<type::jagged_vector<VTYPE>, proxy_domain::host,
                      proxy_access::constant, proxy_type::reference> {

    /// Jagged vector elements are kept by constant reference in the proxy
    using type = std::add_lvalue_reference_t<std::add_const_t<vector<VTYPE>>>;
    /// They are returned as a const reference even on non-const access
    using return_type = type;
    /// They are returned as a const reference on const access
    using const_return_type = type;

    /// Helper function constructing a vector proxy variable
    VECMEM_HOST
    static type make(typename jagged_vector<VTYPE>::size_type i,
                     const jagged_vector<VTYPE>& vec) {

        return vec.at(i);
    }
};

/// Non-const access to a jagged vector variable from a host container
template <typename VTYPE>
struct proxy_var_type<type::jagged_vector<VTYPE>, proxy_domain::host,
                      proxy_access::non_constant, proxy_type::reference> {

    /// Jagged vector elements are kept by non-const lvalue reference in the
    /// proxy
    using type = std::add_lvalue_reference_t<vector<VTYPE>>;
    /// They are returned as a non-const lvalue reference on non-const access
    using return_type = type;
    /// They are returned as a const reference on const access
    using const_return_type =
        std::add_lvalue_reference_t<std::add_const_t<vector<VTYPE>>>;

    /// Helper function constructing a vector proxy variable
    VECMEM_HOST
    static type make(typename jagged_vector<VTYPE>::size_type i,
                     jagged_vector<VTYPE>& vec) {

        return vec.at(i);
    }
};

/// Standalone host jagged vector variable (const and non-const)
template <typename VTYPE, proxy_access PACCESS>
struct proxy_var_type<type::jagged_vector<VTYPE>, proxy_domain::host, PACCESS,
                      proxy_type::standalone> {

    /// Jagged vector elements are kept by constant reference in the proxy
    using type = vector<VTYPE>;
    /// They are returned as a const reference even on non-const access
    using return_type = std::add_lvalue_reference_t<type>;
    /// They are returned as a const reference on const access
    using const_return_type =
        std::add_lvalue_reference_t<std::add_const_t<type>>;

    /// Helper function constructing a vector proxy variable
    VECMEM_HOST
    static type make(typename jagged_vector<VTYPE>::size_type i,
                     const jagged_vector<VTYPE>& vec) {

        return vec.at(i);
    }
};

#endif  // __cplusplus >= 201700L

/// Proxy types for one element of a type pack
template <std::size_t INDEX, proxy_domain PDOMAIN, proxy_access PACCESS,
          proxy_type PTYPE, typename... VARTYPES>
struct proxy_var_type_at {
    /// Type of the variable held by the proxy
    using type =
        typename proxy_var_type<tuple_element_t<INDEX, tuple<VARTYPES...>>,
                                PDOMAIN, PACCESS, PTYPE>::type;
    /// Return type on non-const access to the proxy
    using return_type =
        typename proxy_var_type<tuple_element_t<INDEX, tuple<VARTYPES...>>,
                                PDOMAIN, PACCESS, PTYPE>::return_type;
    /// Return type on const access to the proxy
    using const_return_type =
        typename proxy_var_type<tuple_element_t<INDEX, tuple<VARTYPES...>>,
                                PDOMAIN, PACCESS, PTYPE>::const_return_type;
};

/// @}

/// @name Traits for creating the proxy data tuples
/// @{

/// Technical base class for the @c proxy_data_creator traits
template <typename SCHEMA, proxy_domain PDOMAIN, proxy_access PACCESS,
          proxy_type PTYPE>
struct proxy_data_creator;

/// Helper class making the data tuple for a constant device proxy
template <typename VARTYPE, proxy_domain PDOMAIN>
struct proxy_data_creator<schema<VARTYPE>, PDOMAIN, proxy_access::constant,
                          proxy_type::reference> {

    /// Make all other instantiations of the struct friends
    template <typename, proxy_domain, proxy_access, proxy_type>
    friend struct proxy_data_creator;

    /// Proxy tuple type created by the helper
    using proxy_tuple_type =
        tuple<typename proxy_var_type<VARTYPE, PDOMAIN, proxy_access::constant,
                                      proxy_type::reference>::type>;

    /// Construct the tuple used by the proxy
    template <typename ITYPE, typename CONTAINER>
    VECMEM_HOST_AND_DEVICE static proxy_tuple_type make(ITYPE i,
                                                        const CONTAINER& c) {
        return make_impl<0>(i, c);
    }

private:
    template <std::size_t INDEX, typename ITYPE, typename CONTAINER>
    VECMEM_HOST_AND_DEVICE static proxy_tuple_type make_impl(
        ITYPE i, const CONTAINER& c) {

        return {proxy_var_type<
            VARTYPE, PDOMAIN, proxy_access::constant,
            proxy_type::reference>::make(i, c.template get<INDEX>())};
    }
};

/// Helper class making the data tuple for a non-const device proxy
template <typename VARTYPE, proxy_domain PDOMAIN>
struct proxy_data_creator<schema<VARTYPE>, PDOMAIN, proxy_access::non_constant,
                          proxy_type::reference> {

    /// Make all other instantiations of the struct friends
    template <typename, proxy_domain, proxy_access, proxy_type>
    friend struct proxy_data_creator;

    /// Proxy tuple type created by the helper
    using proxy_tuple_type = tuple<
        typename proxy_var_type<VARTYPE, PDOMAIN, proxy_access::non_constant,
                                proxy_type::reference>::type>;

    /// Construct the tuple used by the proxy
    template <typename ITYPE, typename CONTAINER>
    VECMEM_HOST_AND_DEVICE static proxy_tuple_type make(ITYPE i, CONTAINER& c) {
        return make_impl<0>(i, c);
    }

private:
    template <std::size_t INDEX, typename ITYPE, typename CONTAINER>
    VECMEM_HOST_AND_DEVICE static proxy_tuple_type make_impl(ITYPE i,
                                                             CONTAINER& c) {

        return {proxy_var_type<
            VARTYPE, PDOMAIN, proxy_access::non_constant,
            proxy_type::reference>::make(i, c.template get<INDEX>())};
    }
};

/// Helper class making the data tuple for a constant device proxy
template <typename VARTYPE, typename... VARTYPES, proxy_domain PDOMAIN>
struct proxy_data_creator<schema<VARTYPE, VARTYPES...>, PDOMAIN,
                          proxy_access::constant, proxy_type::reference> {

    /// Make all other instantiations of the struct friends
    template <typename, proxy_domain, proxy_access, proxy_type>
    friend struct proxy_data_creator;

    /// Proxy tuple type created by the helper
    using proxy_tuple_type =
        tuple<typename proxy_var_type<VARTYPE, PDOMAIN, proxy_access::constant,
                                      proxy_type::reference>::type,
              typename proxy_var_type<VARTYPES, PDOMAIN, proxy_access::constant,
                                      proxy_type::reference>::type...>;

    /// Construct the tuple used by the proxy
    template <typename ITYPE, typename CONTAINER>
    VECMEM_HOST_AND_DEVICE static proxy_tuple_type make(ITYPE i,
                                                        const CONTAINER& c) {
        return make_impl<0>(i, c);
    }

private:
    template <std::size_t INDEX, typename ITYPE, typename CONTAINER>
    VECMEM_HOST_AND_DEVICE static proxy_tuple_type make_impl(
        ITYPE i, const CONTAINER& c) {

        return proxy_tuple_type(
            proxy_var_type<
                VARTYPE, PDOMAIN, proxy_access::constant,
                proxy_type::reference>::make(i, c.template get<INDEX>()),
            proxy_data_creator<
                schema<VARTYPES...>, PDOMAIN, proxy_access::constant,
                proxy_type::reference>::template make_impl<INDEX + 1>(i, c));
    }
};

/// Helper class making the data tuple for a non-const device proxy
template <typename VARTYPE, typename... VARTYPES, proxy_domain PDOMAIN>
struct proxy_data_creator<schema<VARTYPE, VARTYPES...>, PDOMAIN,
                          proxy_access::non_constant, proxy_type::reference> {

    /// Make all other instantiations of the struct friends
    template <typename, proxy_domain, proxy_access, proxy_type>
    friend struct proxy_data_creator;

    /// Proxy tuple type created by the helper
    using proxy_tuple_type = tuple<
        typename proxy_var_type<VARTYPE, PDOMAIN, proxy_access::non_constant,
                                proxy_type::reference>::type,
        typename proxy_var_type<VARTYPES, PDOMAIN, proxy_access::non_constant,
                                proxy_type::reference>::type...>;

    /// Construct the tuple used by the proxy
    template <typename ITYPE, typename CONTAINER>
    VECMEM_HOST_AND_DEVICE static proxy_tuple_type make(ITYPE i, CONTAINER& c) {
        return make_impl<0>(i, c);
    }

private:
    template <std::size_t INDEX, typename ITYPE, typename CONTAINER>
    VECMEM_HOST_AND_DEVICE static proxy_tuple_type make_impl(ITYPE i,
                                                             CONTAINER& c) {

        return proxy_tuple_type(
            proxy_var_type<
                VARTYPE, PDOMAIN, proxy_access::non_constant,
                proxy_type::reference>::make(i, c.template get<INDEX>()),
            proxy_data_creator<
                schema<VARTYPES...>, PDOMAIN, proxy_access::non_constant,
                proxy_type::reference>::template make_impl<INDEX + 1>(i, c));
    }
};

/// @}

/// @name Code for copying the proxy tuples
/// @{

/// Copy a tuple that has just a single element
///
/// @tparam T The type of that single element
///
/// @param dst Destination tuple to copy into
/// @param src Source tuple to copy from
///
template <typename T>
void proxy_tuple_copy(tuple<T>& dst, const tuple<T>& src) {
    dst.m_head = src.m_head;
}

/// Copy a tuple that has more than one element
///
/// @tparam T The type of the first element
/// @tparam Ts The types of the rest of the elements
///
/// @param dst Destination tuple to copy into
/// @param src Source tuple to copy from
///
template <typename T, typename... Ts,
          std::enable_if_t<(sizeof...(Ts) > 0), bool> = true>
void proxy_tuple_copy(tuple<T, Ts...>& dst, const tuple<T, Ts...>& src) {
    dst.m_head = src.m_head;
    proxy_tuple_copy(dst.m_tail, src.m_tail);
}

/// @}

}  // namespace details
}  // namespace edm
}  // namespace vecmem
