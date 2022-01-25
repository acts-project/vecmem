/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <cassert>

namespace {

/// @name Buffer alignment helper(s)
/// @{

/// Trait for determining a possible padding between the vector size variable
/// and the vector payload.
template <typename TYPE, typename VALID = void>
struct buffer_alignment_padding;

/// Specialisation of the trait for "small" vector types
template <typename TYPE>
struct buffer_alignment_padding<
    TYPE,
    typename std::enable_if_t<(
        alignof(TYPE) <=
        alignof(typename vecmem::data::vector_buffer<TYPE>::size_type))> > {
    /// Alignment padding value
    static constexpr std::size_t value = 0;
};

/// Specialisation of the trait for "large" vector types
template <typename TYPE>
struct buffer_alignment_padding<
    TYPE,
    typename std::enable_if_t<(
        alignof(TYPE) >
        alignof(typename vecmem::data::vector_buffer<TYPE>::size_type))> > {
    /// Alignment padding value
    static constexpr std::size_t value =
        alignof(TYPE) -
        alignof(typename vecmem::data::vector_buffer<TYPE>::size_type);
};

/// @}

/// Function creating the smart pointer for @c vecmem::data::vector_buffer
template <typename TYPE>
vecmem::unique_alloc_ptr<char[]> allocate_buffer_memory(
    typename vecmem::data::vector_buffer<TYPE>::size_type capacity,
    typename vecmem::data::vector_buffer<TYPE>::size_type size,
    vecmem::memory_resource& resource) {

    // A sanity check.
    assert(capacity >= size);

    // Decide how many bytes to allocate.
    const std::size_t byteSize =
        ((capacity == size)
             ? (capacity * sizeof(TYPE))
             : (sizeof(typename vecmem::data::vector_buffer<TYPE>::size_type) +
                capacity * sizeof(TYPE) +
                buffer_alignment_padding<TYPE>::value));

    if (capacity == 0) {
        return nullptr;
    } else {
        return vecmem::make_unique_alloc<char[]>(resource, byteSize);
    }
}

}  // namespace

namespace vecmem {
namespace data {

template <typename TYPE>
vector_buffer<TYPE>::vector_buffer(size_type size, memory_resource& resource)
    : vector_buffer(size, size, resource) {}

template <typename TYPE>
vector_buffer<TYPE>::vector_buffer(size_type capacity, size_type size,
                                   memory_resource& resource)
    : base_type(capacity, nullptr, nullptr),
      m_memory(::allocate_buffer_memory<TYPE>(capacity, size, resource)) {

    // Set the base class's pointers correctly.
    if (capacity > 0) {
        if (size == capacity) {
            base_type::m_ptr = reinterpret_cast<pointer>(m_memory.get());
        } else {
            base_type::m_size = reinterpret_cast<size_pointer>(m_memory.get());
            base_type::m_ptr = reinterpret_cast<pointer>(
                m_memory.get() + sizeof(size_type) +
                buffer_alignment_padding<TYPE>::value);
        }
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
