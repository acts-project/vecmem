/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// vecmem include(s).
#include "vecmem/containers/details/aligned_multiple_placement.hpp"

// System include(s).
#include <cassert>
#include <memory>

namespace vecmem {
namespace data {

/// A custom implementation for the default constructor is necessary because
/// @c vecmem::data::vector_view does not set its members to anything
/// explicitly in its default constructor. (In order to be trivially default
/// constructible.) So here we need to be explicit.
template <typename TYPE>
vector_buffer<TYPE>::vector_buffer()
    : base_type(static_cast<size_type>(0), nullptr) {}

template <typename TYPE>
vector_buffer<TYPE>::vector_buffer(size_type size, memory_resource& resource)
    : vector_buffer(size, size, resource) {}

template <typename TYPE>
vector_buffer<TYPE>::vector_buffer(size_type capacity, size_type size,
                                   memory_resource& resource)
    : base_type(capacity, nullptr, nullptr) {

    // A sanity check.
    assert(capacity >= size);

    // Exit early for null-capacity buffers.
    if (capacity == 0) {
        return;
    }

    std::tie(m_memory, base_type::m_size, base_type::m_ptr) =
        details::aligned_multiple_placement<std::remove_pointer_t<size_pointer>,
                                            std::remove_pointer_t<pointer>>(
            resource, size == capacity ? 0 : 1, capacity);
}

}  // namespace data

template <typename TYPE>
data::vector_view<TYPE>& get_data(data::vector_buffer<TYPE>& data) {

    return data;
}

template <typename TYPE>
const data::vector_view<TYPE>& get_data(const data::vector_buffer<TYPE>& data) {

    return data;
}

}  // namespace vecmem
