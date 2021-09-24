/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// VecMem include(s).
#include "vecmem/containers/jagged_vector.hpp"
#include "vecmem/utils/debug.hpp"
#include "vecmem/utils/type_traits.hpp"

// System include(s).
#include <cassert>

namespace vecmem {

template <typename TYPE>
void copy::setup(data::vector_view<TYPE>& data) {

    // Check if anything needs to be done.
    if ((data.size_ptr() == nullptr) || (data.capacity() == 0)) {
        return;
    }

    // Initialize the "size variable" correctly on the buffer.
    do_memset(sizeof(typename data::vector_buffer<TYPE>::size_type),
              data.size_ptr(), 0);
    VECMEM_DEBUG_MSG(2,
                     "Prepared a device vector buffer of capacity %u "
                     "for use on a device (ptr: %p)",
                     data.capacity(), static_cast<void*>(data.size_ptr()));
}

template <typename TYPE>
data::vector_buffer<std::remove_cv_t<TYPE>> copy::to(
    const vecmem::data::vector_view<TYPE>& data, memory_resource& resource,
    type::copy_type cptype) {

    // Set up the result buffer.
    data::vector_buffer<std::remove_cv_t<TYPE>> result(
        data.capacity(), get_size(data), resource);
    setup(result);

    // Copy the payload of the vector.
    this->operator()(data, result, cptype);
    VECMEM_DEBUG_MSG(2,
                     "Created a vector buffer of type \"%s\" with "
                     "capacity %u",
                     typeid(TYPE).name(), data.capacity());
    return result;
}

template <typename TYPE1, typename TYPE2>
void copy::operator()(const data::vector_view<TYPE1>& from_view,
                      data::vector_view<TYPE2>& to_view,
                      type::copy_type cptype) {

    // The input and output types are allowed to be different, but only by
    // const-ness.
    static_assert(std::is_same<TYPE1, TYPE2>::value ||
                      details::is_same_nc<TYPE1, TYPE2>::value,
                  "Can only use compatible types in the copy");

    // Get the size of the source view.
    const typename data::vector_view<TYPE1>::size_type size =
        get_size(from_view);

    // Make sure that if the target view is resizable, that it would be set up
    // for the correct size.
    if (to_view.size_ptr() != 0) {
        assert(to_view.capacity() >= size);
        do_copy(sizeof(typename data::vector_view<TYPE2>::size_type), &size,
                to_view.size_ptr(), cptype);
    }

    // Copy the payload.
    assert(size == get_size(to_view));
    do_copy(size * sizeof(TYPE1), from_view.ptr(), to_view.ptr(), cptype);
}

template <typename TYPE1, typename TYPE2, typename ALLOC>
void copy::operator()(const data::vector_view<TYPE1>& from_view,
                      std::vector<TYPE2, ALLOC>& to_vec,
                      type::copy_type cptype) {

    // The input and output types are allowed to be different, but only by
    // const-ness.
    static_assert(std::is_same<TYPE1, TYPE2>::value ||
                      details::is_same_nc<TYPE1, TYPE2>::value,
                  "Can only use compatible types in the copy");

    // Figure out the size of the buffer.
    const typename data::vector_view<TYPE1>::size_type size =
        get_size(from_view);

    // Make the target vector the correct size.
    to_vec.resize(size);
    // Perform the memory copy.
    do_copy(size * sizeof(TYPE1), from_view.ptr(), to_vec.data(), cptype);
}

template <typename TYPE>
typename data::vector_view<TYPE>::size_type copy::get_size(
    const data::vector_view<TYPE>& data) {

    // Handle the simple case, when the view/buffer is not resizable.
    if (data.size_ptr() == nullptr) {
        return data.capacity();
    }

    // If it *is* resizable, don't assume that the size is host-accessible.
    // Explicitly copy it for access.
    typename data::vector_view<TYPE>::size_type result = 0;
    do_copy(sizeof(typename data::vector_view<TYPE>::size_type),
            data.size_ptr(), &result, type::unknown);

    // Return what we got.
    return result;
}

template <typename TYPE>
void copy::setup(data::jagged_vector_buffer<TYPE>& data) {

    // Check if anything needs to be done.
    if (data.m_size == 0) {
        return;
    }

    // "Set up" the inner vector descriptors, using the host-accessible data.
    for (typename data::jagged_vector_buffer<TYPE>::size_type i = 0;
         i < data.m_size; ++i) {
        // Find the first "inner vector" that has a non-zero capacity, and is
        // resizable.
        if ((data.host_ptr()[i].capacity() != 0) &&
            (data.host_ptr()[i].size_ptr() != nullptr)) {
            // Initialise the size values using this "inner vector's" pointer.
            do_memset(sizeof(typename data::vector_buffer<TYPE>::size_type) *
                          (data.m_size - i),
                      data.host_ptr()[i].size_ptr(), 0);
            // This would have initialised all of the "inner vectors" correctly,
            // we can end the loop.
            break;
        }
    }

    // Check if anything else needs to be done.
    if (data.m_ptr == data.host_ptr()) {
        return;
    }

    // Copy the description of the inner vectors of the buffer.
    do_copy(
        data.m_size *
            sizeof(
                typename vecmem::data::jagged_vector_buffer<TYPE>::value_type),
        data.host_ptr(), data.m_ptr, type::host_to_device);
    VECMEM_DEBUG_MSG(2,
                     "Prepared a jagged device vector buffer of size %lu "
                     "for use on a device",
                     data.m_size);
}

template <typename TYPE>
data::jagged_vector_buffer<std::remove_cv_t<TYPE>> copy::to(
    const data::jagged_vector_view<TYPE>& data, memory_resource& resource,
    memory_resource* host_access_resource, type::copy_type cptype) {

    // Create the result buffer object.
    data::jagged_vector_buffer<std::remove_cv_t<TYPE>> result(
        data, resource, host_access_resource);
    assert(result.m_size == data.m_size);

    // Copy the description of the "inner vectors" if necessary.
    setup(result);

    // Copy the payload of the inner vectors.
    copy_views(data.m_size, data.m_ptr, result.host_ptr(), cptype);

    // Return the newly created object.
    return result;
}

template <typename TYPE>
data::jagged_vector_buffer<std::remove_cv_t<TYPE>> copy::to(
    const data::jagged_vector_buffer<TYPE>& data, memory_resource& resource,
    memory_resource* host_access_resource, type::copy_type cptype) {

    // Create the result buffer object.
    data::jagged_vector_buffer<std::remove_cv_t<TYPE>> result(
        data, resource, host_access_resource);
    assert(result.m_size == data.m_size);

    // Copy the description of the "inner vectors" if necessary.
    setup(result);

    // Copy the payload of the inner vectors.
    copy_views(data.m_size, data.host_ptr(), result.host_ptr(), cptype);

    // Return the newly created object.
    return result;
}

template <typename TYPE1, typename TYPE2>
void copy::operator()(const data::jagged_vector_view<TYPE1>& from_view,
                      data::jagged_vector_view<TYPE2>& to_view,
                      type::copy_type cptype) {

    // The input and output types are allowed to be different, but only by
    // const-ness.
    static_assert(std::is_same<TYPE1, TYPE2>::value ||
                      details::is_same_nc<TYPE1, TYPE2>::value,
                  "Can only use compatible types in the copy");

    // A sanity check.
    assert(from_view.m_size == to_view.m_size);

    // Copy the payload of the inner vectors.
    copy_views(from_view.m_size, from_view.m_ptr, to_view.m_ptr, cptype);
}

template <typename TYPE1, typename TYPE2>
void copy::operator()(const data::jagged_vector_view<TYPE1>& from_view,
                      data::jagged_vector_buffer<TYPE2>& to_buffer,
                      type::copy_type cptype) {

    // The input and output types are allowed to be different, but only by
    // const-ness.
    static_assert(std::is_same<TYPE1, TYPE2>::value ||
                      details::is_same_nc<TYPE1, TYPE2>::value,
                  "Can only use compatible types in the copy");

    // A sanity check.
    assert(from_view.m_size == to_buffer.m_size);

    // Copy the payload of the inner vectors.
    copy_views(from_view.m_size, from_view.m_ptr, to_buffer.host_ptr(), cptype);
}

template <typename TYPE1, typename TYPE2>
void copy::operator()(const data::jagged_vector_buffer<TYPE1>& from_buffer,
                      data::jagged_vector_view<TYPE2>& to_view,
                      type::copy_type cptype) {

    // The input and output types are allowed to be different, but only by
    // const-ness.
    static_assert(std::is_same<TYPE1, TYPE2>::value ||
                      details::is_same_nc<TYPE1, TYPE2>::value,
                  "Can only use compatible types in the copy");

    // A sanity check.
    assert(from_buffer.m_size == to_view.m_size);

    // Copy the payload of the inner vectors.
    copy_views(from_buffer.m_size, from_buffer.host_ptr(), to_view.m_ptr,
               cptype);
}

template <typename TYPE1, typename TYPE2>
void copy::operator()(const data::jagged_vector_buffer<TYPE1>& from_buffer,
                      data::jagged_vector_buffer<TYPE2>& to_buffer,
                      type::copy_type cptype) {

    // The input and output types are allowed to be different, but only by
    // const-ness.
    static_assert(std::is_same<TYPE1, TYPE2>::value ||
                      details::is_same_nc<TYPE1, TYPE2>::value,
                  "Can only use compatible types in the copy");

    // A sanity check.
    assert(from_buffer.m_size == to_buffer.m_size);

    // Copy the payload of the inner vectors.
    copy_views(from_buffer.m_size, from_buffer.host_ptr(), to_buffer.host_ptr(),
               cptype);
}

template <typename TYPE1, typename TYPE2, typename ALLOC1, typename ALLOC2>
void copy::operator()(const data::jagged_vector_view<TYPE1>& from_view,
                      std::vector<std::vector<TYPE2, ALLOC2>, ALLOC1>& to_vec,
                      type::copy_type cptype) {

    // The input and output types are allowed to be different, but only by
    // const-ness.
    static_assert(std::is_same<TYPE1, TYPE2>::value ||
                      details::is_same_nc<TYPE1, TYPE2>::value,
                  "Can only use compatible types in the copy");

    // Resize the output object to the correct size.
    to_vec.resize(from_view.m_size);
    const auto sizes = get_sizes(from_view);
    for (typename data::jagged_vector_view<TYPE1>::size_type i = 0;
         i < from_view.m_size; ++i) {
        to_vec[i].resize(sizes[i]);
    }

    // Perform the memory copy.
    auto helper = vecmem::get_data(to_vec);
    this->operator()(from_view, helper, cptype);
}

template <typename TYPE1, typename TYPE2, typename ALLOC1, typename ALLOC2>
void copy::operator()(const data::jagged_vector_buffer<TYPE1>& from_buffer,
                      std::vector<std::vector<TYPE2, ALLOC2>, ALLOC1>& to_vec,
                      type::copy_type cptype) {

    // The input and output types are allowed to be different, but only by
    // const-ness.
    static_assert(std::is_same<TYPE1, TYPE2>::value ||
                      details::is_same_nc<TYPE1, TYPE2>::value,
                  "Can only use compatible types in the copy");

    // Resize the output object to the correct size.
    to_vec.resize(from_buffer.m_size);
    const auto sizes = get_sizes(from_buffer);
    for (typename data::jagged_vector_view<TYPE1>::size_type i = 0;
         i < from_buffer.m_size; ++i) {
        to_vec[i].resize(sizes[i]);
    }

    // Perform the memory copy.
    auto helper = vecmem::get_data(to_vec);
    this->operator()(from_buffer, helper, cptype);
}

template <typename TYPE>
std::vector<typename data::vector_view<TYPE>::size_type> copy::get_sizes(
    const data::jagged_vector_view<TYPE>& data) {

    // Perform the operation using the private function.
    return get_sizes(data.m_ptr, data.m_size);
}

template <typename TYPE>
std::vector<typename data::vector_view<TYPE>::size_type> copy::get_sizes(
    const data::jagged_vector_buffer<TYPE>& data) {

    // Perform the operation using the private function.
    return get_sizes(data.host_ptr(), data.m_size);
}

template <typename TYPE1, typename TYPE2>
void copy::copy_views(std::size_t size,
                      const data::vector_view<TYPE1>* from_view,
                      data::vector_view<TYPE2>* to_view,
                      type::copy_type cptype) {

    // The input and output types are allowed to be different, but only by
    // const-ness.
    static_assert(std::is_same<TYPE1, TYPE2>::value ||
                      details::is_same_nc<TYPE1, TYPE2>::value,
                  "Can only use compatible types in the copy");

    // Helper variables used in the copy.
    const std::remove_cv_t<TYPE1>* from_ptr = nullptr;
    TYPE2* to_ptr = nullptr;
    std::size_t copy_size = 0;
    [[maybe_unused]] std::size_t copy_ops = 0;

    // Get the sizes of the two views.
    const auto from_sizes = get_sizes(from_view, size);
    const auto to_sizes = get_sizes(to_view, size);

    // Helper lambda for figuring out if the next vector element is
    // connected to the currently processed one or not.
    auto next_is_connected = [size](const auto* array, const auto& sizes,
                                    std::size_t index) {
        // Check if the next non-empty vector element is connected to the
        // current one.
        std::size_t j = index + 1;
        while (j < size) {
            if (sizes[j] == 0) {
                ++j;
                continue;
            }
            return ((array[index].ptr() + sizes[index]) == array[j].ptr());
        }
        // If we got here, then the answer is no...
        return false;
    };

    // Perform the copy in multiple steps.
    for (std::size_t i = 0; i < size; ++i) {

        // Skip empty "inner vectors".
        if ((from_sizes[i] == 0) && (to_sizes[i] == 0)) {
            continue;
        }

        // Some sanity checks.
        assert(from_view[i].ptr() != nullptr);
        assert(to_view[i].ptr() != nullptr);
        assert(from_sizes[i] != 0);
        assert(from_sizes[i] == to_sizes[i]);

        // Set/update the helper variables.
        if ((from_ptr == nullptr) && (to_ptr == nullptr) && (copy_size == 0)) {
            from_ptr = from_view[i].ptr();
            to_ptr = to_view[i].ptr();
            copy_size = from_sizes[i] * sizeof(TYPE1);
        } else {
            assert(from_ptr != nullptr);
            assert(to_ptr != nullptr);
            assert(copy_size != 0);
            copy_size += from_sizes[i] * sizeof(TYPE1);
        }

        // Check if the next vector element connects to this one. If not,
        // perform the copy now.
        if ((!next_is_connected(from_view, from_sizes, i)) ||
            (!next_is_connected(to_view, to_sizes, i))) {

            // Perform the copy.
            do_copy(copy_size, from_ptr, to_ptr, cptype);

            // Reset/update the variables.
            from_ptr = nullptr;
            to_ptr = nullptr;
            copy_size = 0;
            copy_ops += 1;
        }
    }

    // Let the user know what happened.
    VECMEM_DEBUG_MSG(2,
                     "Copied the payload of a jagged vector of type "
                     "\"%s\" with %lu copy operation(s)",
                     typeid(TYPE2).name(), copy_ops);
}

template <typename TYPE>
std::vector<typename data::vector_view<TYPE>::size_type> copy::get_sizes(
    const data::vector_view<TYPE>* data, std::size_t size) {

    // Create the result vector.
    std::vector<typename data::vector_view<TYPE>::size_type> result(size, 0);

    // Try to get the "resizable sizes" first.
    for (std::size_t i = 0; i < size; ++i) {
        // Find the first "inner vector" that has a non-zero capacity, and is
        // resizable.
        if ((data[i].capacity() != 0) && (data[i].size_ptr() != nullptr)) {
            // Copy the sizes of the inner vectors into the result vector.
            do_copy(sizeof(typename data::vector_view<TYPE>::size_type) *
                        (size - i),
                    data[i].size_ptr(), result.data() + i, type::unknown);
            // At this point the result vector should have been set up
            // correctly.
            return result;
        }
    }

    // If we're still here, then the buffer is not resizable. So let's just
    // collect the capacity of each of the inner vectors.
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = data[i].capacity();
    }
    return result;
}

}  // namespace vecmem
