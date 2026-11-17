/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/edm/details/data_traits.hpp"

namespace vecmem {
namespace edm {

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST edm::data<edm::schema<VARTYPES...>, INTERFACE>::data()
    : view_type(0u) {}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST edm::data<edm::schema<VARTYPES...>, INTERFACE>::data(
    size_type size, memory_resource& resource)
    : view_type(size),
      m_data{std::make_tuple(
          details::data_alloc<VARTYPES>::make(size, resource)...)} {

    // Set up the views for the jagged vector variables correctly.
    details::data_view_assign<VARTYPES...>(
        view_type::m_views, m_data, std::index_sequence_for<VARTYPES...>());
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST auto edm::data<edm::schema<VARTYPES...>, INTERFACE>::variables()
    -> tuple_type& {

    return m_data;
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST auto edm::data<edm::schema<VARTYPES...>, INTERFACE>::variables()
    const -> const tuple_type& {

    return m_data;
}

template <typename... VARTYPES, template <typename> class INTERFACE>
auto edm::data<edm::schema<VARTYPES...>, INTERFACE>::get_view()
    -> interface_type<view<schema_type, interface_type>> {

    interface_type<view<schema_type, interface_type>> result;
    result = *this;
    return result;
}

template <typename... VARTYPES, template <typename> class INTERFACE>
auto edm::data<edm::schema<VARTYPES...>, INTERFACE>::get_view() const
    -> interface_type<view<details::add_const_t<schema_type>, interface_type>> {

    interface_type<view<details::add_const_t<schema_type>, interface_type>>
        result;
    result = *this;
    return result;
}

}  // namespace edm

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST INTERFACE<edm::view<edm::schema<VARTYPES...>, INTERFACE>> get_data(
    edm::data<edm::schema<VARTYPES...>, INTERFACE>& data) {

    return data.get_view();
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST INTERFACE<
    edm::view<edm::details::add_const_t<edm::schema<VARTYPES...>>, INTERFACE>>
get_data(const edm::data<edm::schema<VARTYPES...>, INTERFACE>& data) {

    return data.get_view();
}

}  // namespace vecmem
