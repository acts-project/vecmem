/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/details/aligned_multiple_placement.hpp"
#include "vecmem/edm/details/buffer_traits.hpp"
#include "vecmem/edm/details/schema_traits.hpp"
#include "vecmem/edm/details/view_traits.hpp"

// System include(s).
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace vecmem {
namespace edm {

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST buffer<schema<VARTYPES...>, INTERFACE>::buffer()
    : INTERFACE<view<schema<VARTYPES...>, INTERFACE>>(0u) {}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST buffer<schema<VARTYPES...>, INTERFACE>::buffer(
    size_type capacity, memory_resource& mr, vecmem::data::buffer_type type)
    : INTERFACE<view<schema<VARTYPES...>, INTERFACE>>(capacity) {

    // Make sure that this constructor is not used for a container that has
    // jagged vectors in it.
    static_assert(
        std::disjunction_v<type::details::is_jagged_vector<VARTYPES>...> ==
            false,
        "Use the other buffer constructor with jagged vector variables!");

    // Perform the appropriate setup.
    switch (type) {
        case vecmem::data::buffer_type::fixed_size:
            setup_fixed(std::vector<std::size_t>(capacity), mr, nullptr,
                        std::index_sequence_for<VARTYPES...>{});
            break;
        case vecmem::data::buffer_type::resizable:
            setup_resizable(std::vector<std::size_t>(capacity), mr, nullptr,
                            std::index_sequence_for<VARTYPES...>{});
            break;
        default:
            throw std::invalid_argument("Unknown buffer type");
    }
}

template <typename... VARTYPES, template <typename> class INTERFACE>
template <
    typename SIZE_TYPE, typename SIZE_ALLOC,
    std::enable_if_t<
        std::is_integral_v<SIZE_TYPE> && std::is_unsigned_v<SIZE_TYPE>, bool>>
VECMEM_HOST buffer<schema<VARTYPES...>, INTERFACE>::buffer(
    const std::vector<SIZE_TYPE, SIZE_ALLOC>& capacities,
    memory_resource& main_mr, memory_resource* host_mr,
    vecmem::data::buffer_type type)
    : INTERFACE<view<schema<VARTYPES...>, INTERFACE>>(
          static_cast<size_type>(capacities.size())) {

    // Make sure that this constructor is only used for a container that has
    // jagged vectors in it.
    static_assert(
        std::disjunction_v<type::details::is_jagged_vector<VARTYPES>...>,
        "Use the other buffer constructor without jagged vector variables!");

    // Perform the appropriate setup.
    switch (type) {
        case vecmem::data::buffer_type::fixed_size:
            setup_fixed(capacities, main_mr, host_mr,
                        std::index_sequence_for<VARTYPES...>{});
            break;
        case vecmem::data::buffer_type::resizable:
            setup_resizable(capacities, main_mr, host_mr,
                            std::index_sequence_for<VARTYPES...>{});
            break;
        default:
            throw std::invalid_argument("Unknown buffer type");
    }
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST memory_resource* buffer<schema<VARTYPES...>, INTERFACE>::resource()
    const {

    return m_memory.get_deleter().resource();
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST memory_resource*
buffer<schema<VARTYPES...>, INTERFACE>::host_resource() const {

    return m_host_memory.get_deleter().resource();
}

template <typename... VARTYPES, template <typename> class INTERFACE>
template <typename SIZE_TYPE, typename SIZE_ALLOC, std::size_t... INDICES>
VECMEM_HOST void buffer<schema<VARTYPES...>, INTERFACE>::setup_fixed(
    const std::vector<SIZE_TYPE, SIZE_ALLOC>& capacities, memory_resource& mr,
    memory_resource* host_mr, std::index_sequence<INDICES...>) {

    // Sanity check.
    static_assert(sizeof...(VARTYPES) == sizeof...(INDICES),
                  "Invalid number of indices");

    // Tuple of pointers to the allocated "layout objects" and "payloads".
    std::tuple<typename details::view_type<VARTYPES>::layout_ptr...>
        layout_ptrs;
    std::tuple<typename details::view_type<VARTYPES>::layout_ptr...>
        host_layout_ptrs;
    std::tuple<typename details::view_type<VARTYPES>::payload_ptr...>
        payload_ptrs;

    // Allocate memory for fixed sized variables.
    std::tie(m_memory, std::get<INDICES>(layout_ptrs)...,
             std::get<INDICES>(payload_ptrs)...) =
        vecmem::details::aligned_multiple_placement<
            typename details::view_type<VARTYPES>::layout_type...,
            typename details::view_type<VARTYPES>::payload_type...>(
            mr, details::buffer_alloc<VARTYPES>::layout_size(capacities)...,
            details::buffer_alloc<VARTYPES>::payload_size(capacities)...);

    // Set the base class's memory views.
    this->m_layout = details::find_layout_view<INTERFACE, VARTYPES...>(
        layout_ptrs,
        {details::buffer_alloc<VARTYPES>::layout_size(capacities)...});
    this->m_payload = details::find_payload_view<INTERFACE, VARTYPES...>(
        payload_ptrs,
        {details::buffer_alloc<VARTYPES>::payload_size(capacities)...});
    this->m_size = {
        static_cast<typename base_type::memory_view_type::size_type>(0),
        nullptr};

    // If requested, allocate host memory for the layouts.
    if (host_mr != nullptr) {

        // Allocate memory for just the layout in host memory.
        std::tie(m_host_memory, std::get<INDICES>(host_layout_ptrs)...) =
            vecmem::details::aligned_multiple_placement<
                typename details::view_type<VARTYPES>::layout_type...>(
                *host_mr,
                details::buffer_alloc<VARTYPES>::layout_size(capacities)...);

        // Set the base class's memory view.
        this->m_host_layout = details::find_layout_view<INTERFACE, VARTYPES...>(
            host_layout_ptrs,
            {details::buffer_alloc<VARTYPES>::layout_size(capacities)...});
    } else {
        // The layout is apparently host accessible.
        this->m_host_layout = this->m_layout;
    }

    // Initialize the views from all the raw pointers.
    this->m_views =
        details::make_buffer_views<SIZE_TYPE, SIZE_ALLOC, VARTYPES...>(
            capacities, layout_ptrs, host_layout_ptrs, payload_ptrs,
            std::index_sequence_for<VARTYPES...>{});
}

template <typename... VARTYPES, template <typename> class INTERFACE>
template <typename SIZE_TYPE, typename SIZE_ALLOC, std::size_t... INDICES>
VECMEM_HOST void buffer<schema<VARTYPES...>, INTERFACE>::setup_resizable(
    const std::vector<SIZE_TYPE, SIZE_ALLOC>& capacities, memory_resource& mr,
    memory_resource* host_mr, std::index_sequence<INDICES...>) {

    // Sanity check(s).
    static_assert(sizeof...(VARTYPES) == sizeof...(INDICES),
                  "Invalid number of indices");
    static_assert(
        std::disjunction_v<type::details::is_vector<VARTYPES>...>,
        "Trying to create a resizable container without any vector variables!");

    // Does the container have jagged vectors in it?
    constexpr bool has_jagged_vectors =
        std::disjunction_v<type::details::is_jagged_vector<VARTYPES>...>;

    // Pointers to the allocated "size variables".
    std::tuple<typename details::view_type<VARTYPES>::size_ptr...> sizes_ptrs;

    // Tuple of pointers to the allocated "layout objects" and "payloads".
    std::tuple<typename details::view_type<VARTYPES>::layout_ptr...>
        layout_ptrs;
    std::tuple<typename details::view_type<VARTYPES>::layout_ptr...>
        host_layout_ptrs;
    std::tuple<typename details::view_type<VARTYPES>::payload_ptr...>
        payload_ptrs;

    // Allocate memory for fixed sized variables. A little differently for
    // containers that have some jagged vectors, versus ones that only have
    // 1D vectors.
    if constexpr (has_jagged_vectors) {
        // Perform the allocation.
        std::tie(m_memory, std::get<INDICES>(sizes_ptrs)...,
                 std::get<INDICES>(layout_ptrs)...,
                 std::get<INDICES>(payload_ptrs)...) =
            vecmem::details::aligned_multiple_placement<
                typename details::view_type<VARTYPES>::size_type...,
                typename details::view_type<VARTYPES>::layout_type...,
                typename details::view_type<VARTYPES>::payload_type...>(
                mr, details::buffer_alloc<VARTYPES>::layout_size(capacities)...,
                details::buffer_alloc<VARTYPES>::layout_size(capacities)...,
                details::buffer_alloc<VARTYPES>::payload_size(capacities)...);
        // Point the base class at the size array.
        this->m_size = {
            static_cast<typename base_type::memory_view_type::size_type>(
                (details::buffer_alloc<VARTYPES>::layout_size(capacities) +
                 ...) *
                sizeof(typename base_type::size_type)),
            reinterpret_cast<typename base_type::memory_view_type::pointer>(
                details::find_first_pointer(
                    sizes_ptrs, std::index_sequence_for<VARTYPES...>{}))};
    } else {
        // Perform the allocation.
        typename base_type::size_pointer size = nullptr;
        std::tie(m_memory, size, std::get<INDICES>(layout_ptrs)...,
                 std::get<INDICES>(payload_ptrs)...) =
            vecmem::details::aligned_multiple_placement<
                typename base_type::size_type,
                typename details::view_type<VARTYPES>::layout_type...,
                typename details::view_type<VARTYPES>::payload_type...>(
                mr, 1u,
                details::buffer_alloc<VARTYPES>::layout_size(capacities)...,
                details::buffer_alloc<VARTYPES>::payload_size(capacities)...);
        // Point the base class at the size variable.
        this->m_size = {
            static_cast<typename base_type::memory_view_type::size_type>(
                sizeof(typename base_type::size_type)),
            reinterpret_cast<typename base_type::memory_view_type::pointer>(
                size)};
        // Set all size pointers to point at the one allocated number.
        ((std::get<INDICES>(sizes_ptrs) = size), ...);
    }

    // Set the base class's memory views.
    this->m_layout = details::find_layout_view<INTERFACE, VARTYPES...>(
        layout_ptrs,
        {details::buffer_alloc<VARTYPES>::layout_size(capacities)...});
    this->m_payload = details::find_payload_view<INTERFACE, VARTYPES...>(
        payload_ptrs,
        {details::buffer_alloc<VARTYPES>::payload_size(capacities)...});

    // If requested, allocate host memory for the layouts.
    if (host_mr != nullptr) {

        // Allocate memory for just the layout in host memory.
        std::tie(m_host_memory, std::get<INDICES>(host_layout_ptrs)...) =
            vecmem::details::aligned_multiple_placement<
                typename details::view_type<VARTYPES>::layout_type...>(
                *host_mr,
                details::buffer_alloc<VARTYPES>::layout_size(capacities)...);

        // Set the base class's memory view.
        this->m_host_layout = details::find_layout_view<INTERFACE, VARTYPES...>(
            host_layout_ptrs,
            {details::buffer_alloc<VARTYPES>::layout_size(capacities)...});
    } else {
        // The layout is apparently host accessible.
        this->m_host_layout = this->m_layout;
    }

    // Initialize the views from all the raw pointers.
    this->m_views =
        details::make_buffer_views<SIZE_TYPE, SIZE_ALLOC, VARTYPES...>(
            capacities, sizes_ptrs, layout_ptrs, host_layout_ptrs, payload_ptrs,
            std::index_sequence_for<VARTYPES...>{});
}

}  // namespace edm

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST INTERFACE<edm::view<edm::schema<VARTYPES...>, INTERFACE>> get_data(
    edm::buffer<edm::schema<VARTYPES...>, INTERFACE>& buffer) {

    return buffer;
}

template <typename... VARTYPES, template <typename> class INTERFACE>
VECMEM_HOST INTERFACE<
    edm::view<edm::details::add_const_t<edm::schema<VARTYPES...>>, INTERFACE>>
get_data(const edm::buffer<edm::schema<VARTYPES...>, INTERFACE>& buffer) {

    return buffer;
}

}  // namespace vecmem
