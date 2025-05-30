/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/details/schema_traits.hpp"
#include "vecmem/utils/type_traits.hpp"

// System include(s).
#include <cassert>

namespace vecmem {
namespace edm {

template <typename... VARTYPES>
VECMEM_HOST_AND_DEVICE view<schema<VARTYPES...>>::view(
    size_type capacity, const memory_view_type& size)
    : m_capacity(capacity),
      m_views{},
      m_size{size},
      m_payload{0, nullptr},
      m_layout{0, nullptr},
      m_host_layout{0, nullptr} {}

template <typename... VARTYPES>
template <typename... OTHERTYPES,
          std::enable_if_t<
              vecmem::details::conjunction_v<std::is_constructible<
                  typename details::view_type<VARTYPES>::type,
                  typename details::view_type<OTHERTYPES>::type>...> &&
                  vecmem::details::disjunction_v<
                      vecmem::details::negation<std::is_same<
                          typename details::view_type<VARTYPES>::type,
                          typename details::view_type<OTHERTYPES>::type>>...>,
              bool>>
VECMEM_HOST_AND_DEVICE view<schema<VARTYPES...>>::view(
    const view<schema<OTHERTYPES...>>& other)
    : m_capacity{other.capacity()},
      m_views{other.variables()},
      m_size{other.size()},
      m_payload{other.payload()},
      m_layout{other.layout()},
      m_host_layout{other.host_layout()} {}

template <typename... VARTYPES>
template <typename... OTHERTYPES,
          std::enable_if_t<
              vecmem::details::conjunction_v<std::is_constructible<
                  typename details::view_type<VARTYPES>::type,
                  typename details::view_type<OTHERTYPES>::type>...> &&
                  vecmem::details::disjunction_v<
                      vecmem::details::negation<std::is_same<
                          typename details::view_type<VARTYPES>::type,
                          typename details::view_type<OTHERTYPES>::type>>...>,
              bool>>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::operator=(
    const view<schema<OTHERTYPES...>>& rhs) -> view& {

    // Note that self-assignment with this function should never be a thing.
    // So we don't need to check for it in production code.
    assert(static_cast<const void*>(this) != static_cast<const void*>(&rhs));

    // Copy the data from the other view.
    m_capacity = rhs.capacity();
    m_views = rhs.variables();
    m_size = rhs.size();
    m_payload = rhs.payload();
    m_layout = rhs.layout();
    m_host_layout = rhs.host_layout();

    // Return a reference to this object.
    return *this;
}

template <typename... VARTYPES>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::capacity() const
    -> size_type {

    return m_capacity;
}

template <typename... VARTYPES>
template <std::size_t INDEX>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::get()
    -> tuple_element_t<INDEX, tuple_type>& {

    return vecmem::get<INDEX>(m_views);
}

template <typename... VARTYPES>
template <std::size_t INDEX>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::get() const
    -> const tuple_element_t<INDEX, tuple_type>& {

    return vecmem::get<INDEX>(m_views);
}

template <typename... VARTYPES>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::variables()
    -> tuple_type& {

    return m_views;
}

template <typename... VARTYPES>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::variables() const
    -> const tuple_type& {

    return m_views;
}

template <typename... VARTYPES>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::size() const
    -> const memory_view_type& {

    return m_size;
}

template <typename... VARTYPES>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::payload() const
    -> const memory_view_type& {

    return m_payload;
}

template <typename... VARTYPES>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::layout() const
    -> const memory_view_type& {

    return m_layout;
}

template <typename... VARTYPES>
VECMEM_HOST_AND_DEVICE auto view<schema<VARTYPES...>>::host_layout() const
    -> const memory_view_type& {

    return m_host_layout;
}

namespace details {

template <std::size_t INDEX, typename... VARTYPES>
struct get_capacities_impl {

    VECMEM_HOST
    static std::vector<vecmem::data::vector_view<int>::size_type> get(
        const view<schema<VARTYPES...>>& soa) {

        return get_impl(soa, soa.template get<INDEX>());
    }

private:
    template <typename T>
    VECMEM_HOST static std::vector<vecmem::data::vector_view<int>::size_type>
    get_impl(const view<schema<VARTYPES...>>& soa, const T&) {

        return get_capacities_impl<INDEX - 1, VARTYPES...>::get(soa);
    }

    template <typename T>
    VECMEM_HOST static std::vector<vecmem::data::vector_view<int>::size_type>
    get_impl(const view<schema<VARTYPES...>>&,
             const vecmem::data::jagged_vector_view<T>& vec) {

        return vecmem::data::get_capacities(vec);
    }

};  // struct get_capacities_impl

template <typename... VARTYPES>
struct get_capacities_impl<0u, VARTYPES...> {

    VECMEM_HOST
    static std::vector<vecmem::data::vector_view<int>::size_type> get(
        const view<schema<VARTYPES...>>& soa) {

        // If we got this far, this *must* be a jagged vector.
        static_assert(
            type::details::is_jagged_vector<
                tuple_element_t<0u, tuple<VARTYPES...>>>::value,
            "The first variable in the schema must be a jagged vector");

        // Get the capacities with the helper function available for jagged
        // vectors.
        return vecmem::data::get_capacities(soa.template get<0>());
    }

};  // struct get_capacities_impl

}  // namespace details

template <typename... VARTYPES>
VECMEM_HOST std::vector<vecmem::data::vector_view<int>::size_type>
get_capacities(const view<schema<VARTYPES...>>& soa) {

    // Make sure that there's a jagged vector in here.
    static_assert(
        details::has_jagged_vector<schema<VARTYPES...>>::value,
        "Function can only be used on containers with jagged vectors");

    return details::get_capacities_impl<sizeof...(VARTYPES) - 1,
                                        VARTYPES...>::get(soa);
}

}  // namespace edm
}  // namespace vecmem
