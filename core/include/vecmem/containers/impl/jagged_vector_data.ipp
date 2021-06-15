/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <cstddef>

namespace {

/// Function creating the smart pointer for @c vecmem::data::jagged_vector_data
template <typename TYPE>
std::unique_ptr<typename vecmem::data::jagged_vector_view<TYPE>::value_type,
                vecmem::details::deallocator>
allocate_jagged_memory(
    typename vecmem::data::jagged_vector_view<TYPE>::size_type size,
    vecmem::memory_resource& resource) {

    const std::size_t byteSize =
        size *
        sizeof(typename vecmem::data::jagged_vector_view<TYPE>::value_type);
    return {size == 0
                ? nullptr
                : static_cast<
                      typename vecmem::data::jagged_vector_view<TYPE>::pointer>(
                      resource.allocate(byteSize)),
            {byteSize, resource}};
}

}  // namespace

namespace vecmem {
namespace data {

template <typename T>
jagged_vector_data<T>::jagged_vector_data(size_type size, memory_resource& mem)
    : base_type(size, nullptr),
      m_memory(::allocate_jagged_memory<T>(size, mem)) {
    // Point the base class at the newly allocated memory.
    base_type::m_ptr = m_memory.get();

    /*
     * Construct vecmem::data::vector_view objects in the allocated area.
     * Simply with their default constructors, as they will need to be
     * filled "from the outside".
     */
    for (std::size_t i = 0; i < size; ++i) {
        /*
         * We use the memory allocated earlier and construct device vector
         * objects there.
         */
        new (base_type::m_ptr + i) vector_view<T>();
    }
}

}  // namespace data

template <typename TYPE>
data::jagged_vector_view<TYPE>& get_data(data::jagged_vector_data<TYPE>& data) {

    return data;
}

template <typename TYPE>
const data::jagged_vector_view<TYPE>& get_data(
    const data::jagged_vector_data<TYPE>& data) {

    return data;
}

}  // namespace vecmem
