/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
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
vector_buffer<TYPE>::vector_buffer(size_type capacity,
                                   memory_resource& resource, buffer_type type)
    : base_type(capacity, nullptr, nullptr) {

    // Exit early for null-capacity buffers.
    if (capacity == 0) {
        return;
    }

    // Allocate the memory of the buffer.
    size_pointer size = nullptr;
    pointer ptr = nullptr;
    std::tie(m_memory, size, ptr) =
        details::aligned_multiple_placement<std::remove_pointer_t<size_pointer>,
                                            std::remove_pointer_t<pointer>>(
            resource, type == buffer_type::fixed_size ? 0 : 1, capacity);

    // Set up the base object.
    base_type::operator=(base_type{capacity, size, ptr});
}

template <typename TYPE>
memory_resource* vector_buffer<TYPE>::resource() const {

    return m_memory.get_deleter().resource();
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
