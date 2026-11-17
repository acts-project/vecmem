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
    : view<schema<VARTYPES...>, INTERFACE>(0u) {}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST edm::data<edm::schema<VARTYPES...>, INTERFACE>::data(
    size_type size, memory_resource& resource)
    : view<schema<VARTYPES...>, INTERFACE>(size),
      m_data{std::make_tuple(
          details::data_alloc<VARTYPES>::make(size, resource)...)} {

    // Set up the views for the jagged vector variables correctly.
    details::data_view_assign<VARTYPES...>(
        this->m_views, m_data, std::index_sequence_for<VARTYPES...>());
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
INTERFACE<view<schema<VARTYPES...>, INTERFACE>>
edm::data<edm::schema<VARTYPES...>, INTERFACE>::get_view() {

    INTERFACE<view<schema<VARTYPES...>, INTERFACE>> result;
    result = *this;
    return result;
}

template <typename... VARTYPES, template <typename> class INTERFACE>
INTERFACE<view<details::add_const_t<schema<VARTYPES...>>, INTERFACE>>
edm::data<edm::schema<VARTYPES...>, INTERFACE>::get_view() const {

    INTERFACE<view<details::add_const_t<schema<VARTYPES...>>, INTERFACE>>
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
