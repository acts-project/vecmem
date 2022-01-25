/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <cassert>
#include <memory>

namespace vecmem {
namespace data {

template <typename TYPE>
vector_buffer<TYPE>::vector_buffer(size_type size, memory_resource& resource)
    : vector_buffer(size, size, resource) {}

template <typename TYPE>
vector_buffer<TYPE>::vector_buffer(size_type capacity, size_type size,
                                   memory_resource& resource)
    : base_type(capacity, nullptr, nullptr), m_memory() {

    // A sanity check.
    assert(capacity >= size);

    // Exit early for null-capacity buffers.
    if (capacity == 0) {
        return;
    }

    // Alignment for the vector elements.
    static constexpr std::size_t alignment = alignof(TYPE);

    // Decide how many bytes we need to allocate.
    std::size_t byteSize = capacity * sizeof(TYPE);

    // Increase this size if the buffer describes a resizable vector.
    if (capacity != size) {
        byteSize +=
            sizeof(typename vecmem::data::vector_buffer<TYPE>::size_type);
        // Further increase this size so that we could for sure align the
        // payload data correctly.
        byteSize = ((byteSize + alignment - 1) / alignment) * alignment;
    }

    // Allocate the memory.
    m_memory = vecmem::make_unique_alloc<char[]>(resource, byteSize);

    // Set the base class's pointers correctly.
    if (size == capacity) {
        base_type::m_ptr = reinterpret_cast<pointer>(m_memory.get());
    } else {
        base_type::m_size = reinterpret_cast<size_pointer>(m_memory.get());
        void* ptr = m_memory.get() + sizeof(size_type);
        std::size_t space = byteSize - sizeof(size_type);
        base_type::m_ptr = reinterpret_cast<pointer>(
            std::align(alignof(TYPE), capacity * sizeof(TYPE), ptr, space));
    }
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
