/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <cassert>
#include <cstddef>
#include <memory>
#include <numeric>
#include <vector>

namespace {

/// Helper conversion function
template <typename TYPE>
std::vector<std::size_t> get_sizes(
    const vecmem::data::jagged_vector_view<TYPE>& jvv) {

    std::vector<std::size_t> result(jvv.m_size);
    for (std::size_t i = 0; i < jvv.m_size; ++i) {
        result[i] = jvv.m_ptr[i].size();
    }
    return result;
}

/// Function allocating memory for @c vecmem::data::jagged_vector_buffer
template <typename TYPE>
vecmem::unique_alloc_ptr<
    typename vecmem::data::jagged_vector_buffer<TYPE>::value_type[]>
allocate_jagged_buffer_outer_memory(
    typename vecmem::data::jagged_vector_buffer<TYPE>::size_type size,
    vecmem::memory_resource& resource) {

    if (size == 0) {
        return nullptr;
    } else {
        return vecmem::make_unique_alloc<
            typename vecmem::data::jagged_vector_buffer<TYPE>::value_type[]>(
            resource, size);
    }
}

/// Function allocating memory for @c vecmem::data::jagged_vector_buffer
template <typename TYPE>
vecmem::unique_alloc_ptr<char[]> allocate_jagged_buffer_inner_memory(
    const std::vector<std::size_t>& sizes, vecmem::memory_resource& resource,
    bool isResizable, std::size_t& bufferSize) {

    // Alignment for the vector elements.
    static constexpr std::size_t alignment = alignof(TYPE);

    // Determine the allocation size.
    bufferSize = std::accumulate(sizes.begin(), sizes.end(),
                                 static_cast<std::size_t>(0)) *
                 sizeof(TYPE);
    // Increase this size if the buffer describes a resizable vector.
    if (isResizable) {
        bufferSize +=
            sizes.size() * sizeof(typename vecmem::data::jagged_vector_buffer<
                                  TYPE>::value_type::size_type);
        // Further increase this size so that we could for sure align the
        // payload data correctly.
        bufferSize = ((bufferSize + alignment - 1) / alignment) * alignment;
    }

    // Return a smart pointer with the allocation.
    return vecmem::make_unique_alloc<char[]>(resource, bufferSize);
}

}  // namespace

namespace vecmem {
namespace data {

template <typename TYPE>
template <typename OTHERTYPE,
          std::enable_if_t<std::is_convertible<TYPE, OTHERTYPE>::value, bool> >
jagged_vector_buffer<TYPE>::jagged_vector_buffer(
    const jagged_vector_view<OTHERTYPE>& other, memory_resource& resource,
    memory_resource* host_access_resource)
    : jagged_vector_buffer(::get_sizes(other), resource, host_access_resource) {

}

template <typename TYPE>
jagged_vector_buffer<TYPE>::jagged_vector_buffer(
    const std::vector<std::size_t>& sizes, memory_resource& resource,
    memory_resource* host_access_resource)
    : base_type(sizes.size(), nullptr),
      m_outer_memory(::allocate_jagged_buffer_outer_memory<TYPE>(
          (host_access_resource == nullptr ? 0 : sizes.size()), resource)),
      m_outer_host_memory(::allocate_jagged_buffer_outer_memory<TYPE>(
          sizes.size(),
          (host_access_resource == nullptr ? resource
                                           : *host_access_resource))),
      m_inner_memory_size(0u),
      m_inner_memory(::allocate_jagged_buffer_inner_memory<TYPE>(
          sizes, resource, false, m_inner_memory_size)) {

    // Point the base class at the newly allocated memory.
    base_type::m_ptr =
        ((host_access_resource != nullptr) ? m_outer_memory.get()
                                           : m_outer_host_memory.get());

    // Set up the host accessible memory array.
    std::ptrdiff_t ptrdiff = 0;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        new (host_ptr() + i)
            value_type(static_cast<typename value_type::size_type>(sizes[i]),
                       reinterpret_cast<TYPE*>(m_inner_memory.get() + ptrdiff));
        ptrdiff += sizes[i] * sizeof(TYPE);
    }
}

template <typename TYPE>
jagged_vector_buffer<TYPE>::jagged_vector_buffer(
    const std::vector<std::size_t>& sizes,
    const std::vector<std::size_t>& capacities, memory_resource& resource,
    memory_resource* host_access_resource)
    : base_type(sizes.size(), nullptr),
      m_outer_memory(::allocate_jagged_buffer_outer_memory<TYPE>(
          (host_access_resource == nullptr ? 0 : sizes.size()), resource)),
      m_outer_host_memory(::allocate_jagged_buffer_outer_memory<TYPE>(
          sizes.size(),
          (host_access_resource == nullptr ? resource
                                           : *host_access_resource))),
      m_inner_memory_size(0u),
      m_inner_memory(::allocate_jagged_buffer_inner_memory<TYPE>(
          capacities, resource, true, m_inner_memory_size)) {

    // Some sanity check.
    assert(sizes.size() == capacities.size());

    // Point the base class at the newly allocated memory.
    base_type::m_ptr =
        ((host_access_resource != nullptr) ? m_outer_memory.get()
                                           : m_outer_host_memory.get());

    // Size of the data payload.
    const std::size_t dataSize =
        std::accumulate(capacities.begin(), capacities.end(),
                        static_cast<std::size_t>(0)) *
        sizeof(TYPE);

    //
    // Construct a correctly aligned "start pointer" for the data elements.
    //
    // Construct the unaligned pointer by simply jumping over the "size array".
    void* unaligned_start_ptr =
        m_inner_memory.get() +
        (capacities.size() * sizeof(typename value_type::size_type));
    // The remaining size of the buffer, with the "size array" size removed.
    std::size_t buffer_size =
        m_inner_memory_size -
        (capacities.size() * sizeof(typename value_type::size_type));
    // Construct the "start pointer" using std::align.
    char* start_ptr = static_cast<char*>(
        std::align(alignof(TYPE), dataSize, unaligned_start_ptr, buffer_size));

    // Set up the vecmem::vector_view objects in the host accessible memory.
    std::ptrdiff_t ptrdiff = 0;
    for (std::size_t i = 0; i < capacities.size(); ++i) {
        new (host_ptr() + i) value_type(
            static_cast<typename value_type::size_type>(capacities[i]),
            reinterpret_cast<typename value_type::size_pointer>(
                m_inner_memory.get() +
                i * sizeof(typename value_type::size_type)),
            reinterpret_cast<TYPE*>(start_ptr + ptrdiff));
        ptrdiff += capacities[i] * sizeof(TYPE);
    }
}

template <typename TYPE>
auto jagged_vector_buffer<TYPE>::host_ptr() const -> pointer {

    return m_outer_host_memory.get();
}

}  // namespace data

template <typename TYPE>
data::jagged_vector_view<TYPE>& get_data(
    data::jagged_vector_buffer<TYPE>& data) {

    return data;
}

template <typename TYPE>
const data::jagged_vector_view<TYPE>& get_data(
    const data::jagged_vector_buffer<TYPE>& data) {

    return data;
}

}  // namespace vecmem
