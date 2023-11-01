/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/edm/schema.hpp"

// System include(s).
#include <tuple>

namespace vecmem::edm::details {

/// @name Traits for the view types for the individual variables
/// @{

template <typename TYPE>
struct view_type_base {
    using raw_type = TYPE;
    using pointer_type = raw_type*;
};  // struct view_type_base

template <typename TYPE>
struct view_type : public view_type_base<TYPE> {
    struct UNKNOWN_TYPE {};
    using type = UNKNOWN_TYPE;
};  // struct view_type

template <typename TYPE>
struct view_type<type::scalar<TYPE> > : public view_type_base<TYPE> {
    using type = TYPE*;
};  // struct view_type

template <typename TYPE>
struct view_type<type::vector<TYPE> > : public view_type_base<TYPE> {
    using type = data::vector_view<TYPE>;
};  // struct view_type

template <typename TYPE>
struct view_type<type::jagged_vector<TYPE> > : public view_type_base<TYPE> {
    using type = data::jagged_vector_view<TYPE>;
};  // struct view_type

/// @}

/// @name Traits for making allocations inside of buffers
/// @{

template <typename TYPE>
struct buffer_alloc {
    static std::size_t size(std::size_t size) { return size; }
    static typename view_type<TYPE>::type make_view(
        std::size_t size, typename view_type<TYPE>::pointer_type ptr) {
        return {size, ptr};
    }
    static typename view_type<TYPE>::type make_view(
        std::size_t capacity, unsigned int* size,
        typename view_type<TYPE>::pointer_type ptr) {
        return {capacity, size, ptr};
    }
};  // struct buffer_alloc

template <typename TYPE>
struct buffer_alloc<type::scalar<TYPE> > {
    static std::size_t size(std::size_t) { return 1u; }
    static typename view_type<type::scalar<TYPE> >::type make_view(
        std::size_t,
        typename view_type<type::scalar<TYPE> >::pointer_type ptr) {
        return ptr;
    }
    static typename view_type<type::scalar<TYPE> >::type make_view(
        std::size_t, unsigned int*,
        typename view_type<type::scalar<TYPE> >::pointer_type ptr) {
        return ptr;
    }
};  // struct buffer_alloc

template <typename TYPE>
struct buffer_alloc<type::vector<TYPE> > {
    static std::size_t size(std::size_t size) { return size; }
    static typename view_type<type::vector<TYPE> >::type make_view(
        std::size_t size,
        typename view_type<type::vector<TYPE> >::pointer_type ptr) {
        return {
            static_cast<
                typename view_type<type::vector<TYPE> >::type::size_type>(size),
            ptr};
    }
    static typename view_type<type::vector<TYPE> >::type make_view(
        std::size_t capacity, unsigned int* size,
        typename view_type<type::vector<TYPE> >::pointer_type ptr) {
        return {static_cast<
                    typename view_type<type::vector<TYPE> >::type::size_type>(
                    capacity),
                size, ptr};
    }
};  // struct buffer_alloc

template <typename TYPE>
struct buffer_alloc<type::jagged_vector<TYPE> > {};  // struct buffer_alloc

template <typename... TYPES, std::size_t... I>
auto make_buffer_views_impl(
    std::size_t size,
    std::tuple<typename view_type<TYPES>::pointer_type...>& ptrs,
    std::index_sequence<I...>) {

    return std::make_tuple(
        buffer_alloc<TYPES>::make_view(size, std::get<I>(ptrs))...);
}

template <typename... TYPES>
auto make_buffer_views(
    std::size_t size,
    std::tuple<typename view_type<TYPES>::pointer_type...>& ptrs) {

    return make_buffer_views_impl<TYPES...>(
        size, ptrs, std::index_sequence_for<TYPES...>());
}

template <typename... TYPES, std::size_t... I>
auto make_buffer_views_impl(
    std::size_t capacity, unsigned int* size,
    std::tuple<typename view_type<TYPES>::pointer_type...>& ptrs,
    std::index_sequence<I...>) {

    return std::make_tuple(
        buffer_alloc<TYPES>::make_view(capacity, size, std::get<I>(ptrs))...);
}

template <typename... TYPES>
auto make_buffer_views(
    std::size_t capacity, unsigned int* size,
    std::tuple<typename view_type<TYPES>::pointer_type...>& ptrs) {

    return make_buffer_views_impl<TYPES...>(
        capacity, size, ptrs, std::index_sequence_for<TYPES...>());
}

/// @}

}  // namespace vecmem::edm::details
