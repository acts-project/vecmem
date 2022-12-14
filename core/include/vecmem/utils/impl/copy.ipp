/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// VecMem include(s).
#include "vecmem/containers/jagged_vector.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/utils/debug.hpp"
#include "vecmem/utils/type_traits.hpp"

// System include(s).
#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace vecmem {

template <typename TYPE>
copy::event_type copy::setup(data::vector_view<TYPE> data) const {

    // Check if anything needs to be done.
    if ((data.size_ptr() == nullptr) || (data.capacity() == 0)) {
        return vecmem::copy::create_event();
    }

    // Initialize the "size variable" correctly on the buffer.
    do_memset(sizeof(typename data::vector_view<TYPE>::size_type),
              data.size_ptr(), 0);
    VECMEM_DEBUG_MSG(2,
                     "Prepared a device vector buffer of capacity %u "
                     "for use on a device (ptr: %p)",
                     data.capacity(), static_cast<void*>(data.size_ptr()));

    // Return a new event.
    return create_event();
}

template <typename TYPE>
copy::event_type copy::memset(data::vector_view<TYPE> data, int value) const {

    // Check if anything needs to be done.
    if (data.capacity() == 0) {
        return vecmem::copy::create_event();
    }

    // Call memset with the correct arguments.
    do_memset(data.capacity() * sizeof(TYPE), data.ptr(), value);
    VECMEM_DEBUG_MSG(2, "Set %u vector elements to %i at ptr: %p",
                     data.capacity(), value, static_cast<void*>(data.ptr()));

    // Return a new event.
    return create_event();
}

template <typename TYPE>
data::vector_buffer<std::remove_cv_t<TYPE>> copy::to(
    const vecmem::data::vector_view<TYPE>& data, memory_resource& resource,
    type::copy_type cptype) const {

    // Set up the result buffer.
    data::vector_buffer<std::remove_cv_t<TYPE>> result(
        data.capacity(), get_size(data), resource);
    setup(result)->wait();

    // Copy the payload of the vector. Explicitly waiting for the copy to finish
    // before returning the buffer.
    operator()(data, result, cptype)->wait();

    // Return the buffer.
    return result;
}

template <typename TYPE1, typename TYPE2>
copy::event_type copy::operator()(const data::vector_view<TYPE1>& from_view,
                                  data::vector_view<TYPE2> to_view,
                                  type::copy_type cptype) const {

    // The input and output types are allowed to be different, but only by
    // const-ness.
    static_assert(std::is_same<TYPE1, TYPE2>::value ||
                      details::is_same_nc<TYPE1, TYPE2>::value,
                  "Can only use compatible types in the copy");

    // Get the size of the source view.
    const typename data::vector_view<TYPE1>::size_type size =
        get_size(from_view);

    // Make sure that the copy can happen.
    if (to_view.capacity() < size) {
        std::ostringstream msg;
        msg << "Target capacity (" << to_view.capacity() << ") < source size ("
            << size << ")";
        throw std::length_error(msg.str());
    }

    // Make sure that if the target view is resizable, that it would be set up
    // for the correct size.
    if (to_view.size_ptr() != nullptr) {
        do_copy(sizeof(typename data::vector_view<TYPE2>::size_type), &size,
                to_view.size_ptr(), cptype);
    }

    // Copy the payload.
    assert(size == get_size(to_view));
    do_copy(size * sizeof(TYPE1), from_view.ptr(), to_view.ptr(), cptype);

    // Return a new event.
    return create_event();
}

template <typename TYPE1, typename TYPE2, typename ALLOC>
copy::event_type copy::operator()(const data::vector_view<TYPE1>& from_view,
                                  std::vector<TYPE2, ALLOC>& to_vec,
                                  type::copy_type cptype) const {

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

    // Return a new event.
    return create_event();
}

template <typename TYPE>
typename data::vector_view<TYPE>::size_type copy::get_size(
    const data::vector_view<TYPE>& data) const {

    // Handle the simple case, when the view/buffer is not resizable.
    if (data.size_ptr() == nullptr) {
        return data.capacity();
    }

    // If it *is* resizable, don't assume that the size is host-accessible.
    // Explicitly copy it for access.
    typename data::vector_view<TYPE>::size_type result = 0;
    do_copy(sizeof(typename data::vector_view<TYPE>::size_type),
            data.size_ptr(), &result, type::unknown);

    // Wait for the copy operation to finish. With some backends
    // (khm... SYCL... khm...) copies can be asynchronous even into
    // non-pinned host memory.
    create_event()->wait();

    // Return what we got.
    return result;
}

template <typename TYPE>
copy::event_type copy::setup(data::jagged_vector_view<TYPE> data) const {

    // Check if anything needs to be done.
    if (data.size() == 0) {
        return vecmem::copy::create_event();
    }

    // "Set up" the inner vector descriptors, using the host-accessible data.
    // But only if the jagged vector buffer is resizable.
    if (data.host_ptr()[0].size_ptr() != nullptr) {
        do_memset(
            sizeof(typename data::vector_buffer<TYPE>::size_type) * data.size(),
            data.host_ptr()[0].size_ptr(), 0);
    }

    // Check if anything else needs to be done.
    if (data.ptr() == data.host_ptr()) {
        return create_event();
    }

    // Copy the description of the inner vectors of the buffer.
    do_copy(
        data.size() *
            sizeof(
                typename vecmem::data::jagged_vector_buffer<TYPE>::value_type),
        data.host_ptr(), data.ptr(), type::host_to_device);
    VECMEM_DEBUG_MSG(2,
                     "Prepared a jagged device vector buffer of size %lu "
                     "for use on a device",
                     data.size());

    // Return a new event.
    return create_event();
}

template <typename TYPE>
copy::event_type copy::memset(data::jagged_vector_view<TYPE> data,
                              int value) const {

    // Use a very naive/expensive implementation.
    for (std::size_t i = 0; i < data.size(); ++i) {
        this->memset(data.host_ptr()[i], value);
    }

    // Return a new event.
    return create_event();
}

template <typename TYPE>
data::jagged_vector_buffer<std::remove_cv_t<TYPE>> copy::to(
    const data::jagged_vector_view<TYPE>& data, memory_resource& resource,
    memory_resource* host_access_resource, type::copy_type cptype) const {

    // Create the result buffer object.
    data::jagged_vector_buffer<std::remove_cv_t<TYPE>> result(
        data, resource, host_access_resource);
    assert(result.size() == data.size());

    // Copy the description of the "inner vectors" if necessary.
    setup(result)->wait();

    // Copy the payload of the inner vectors. Explicitly waiting for the copy to
    // finish before returning the buffer.
    operator()(data, result, cptype)->wait();

    // Return the newly created object.
    return result;
}

template <typename TYPE1, typename TYPE2>
copy::event_type copy::operator()(
    const data::jagged_vector_view<TYPE1>& from_view,
    data::jagged_vector_view<TYPE2> to_view, type::copy_type cptype) const {

    // The input and output types are allowed to be different, but only by
    // const-ness.
    static_assert(std::is_same<TYPE1, TYPE2>::value ||
                      details::is_same_nc<TYPE1, TYPE2>::value,
                  "Can only use compatible types in the copy");

    // A sanity check.
    if (from_view.size() > to_view.size()) {
        std::ostringstream msg;
        msg << "from_view.size() (" << from_view.size()
            << ") > to_view.size() (" << to_view.size() << ")";
        throw std::length_error(msg.str());
    }

    // Check if anything needs to be done.
    const std::size_t size = from_view.size();
    if (size == 0) {
        return vecmem::copy::create_event();
    }

    // Calculate the contiguous-ness of the memory allocations.
    const bool from_is_contiguous = is_contiguous(from_view.host_ptr(), size);
    const bool to_is_contiguous = is_contiguous(to_view.host_ptr(), size);
    VECMEM_DEBUG_MSG(3, "from_is_contiguous = %d, to_is_contiguous = %d",
                     from_is_contiguous, to_is_contiguous);

    // Get the sizes of the source jagged vector.
    const auto sizes = get_sizes(from_view);

    // Before even attempting the copy, make sure that the target view either
    // has the correct sizes, or can be resized correctly.
    set_sizes(sizes, to_view);

    // Check whether the source and target capacities match up. We can only
    // perform the "optimised copy" if they do.
    std::vector<typename data::vector_view<TYPE1>::size_type> capacities(size);
    bool capacities_match = true;
    for (std::size_t i = 0; i < size; ++i) {
        if (from_view.host_ptr()[i].capacity() !=
            to_view.host_ptr()[i].capacity()) {
            capacities_match = false;
            break;
        }
        capacities[i] = from_view.host_ptr()[i].capacity();
    }

    // Perform the copy as best as we can.
    if (from_is_contiguous && to_is_contiguous && capacities_match) {
        // Perform the copy in one go.
        copy_views_contiguous_impl(capacities, from_view.host_ptr(),
                                   to_view.host_ptr(), cptype);
    } else {
        // Do the copy as best as we can. Note that since they are not
        // contiguous anyway, we use the sizes of the vectors here, not their
        // capcities.
        copy_views_impl(sizes, from_view.host_ptr(), to_view.host_ptr(),
                        cptype);
    }

    // Return a new event.
    return create_event();
}

template <typename TYPE1, typename TYPE2, typename ALLOC1, typename ALLOC2>
copy::event_type copy::operator()(
    const data::jagged_vector_view<TYPE1>& from_view,
    std::vector<std::vector<TYPE2, ALLOC2>, ALLOC1>& to_vec,
    type::copy_type cptype) const {

    // The input and output types are allowed to be different, but only by
    // const-ness.
    static_assert(std::is_same<TYPE1, TYPE2>::value ||
                      details::is_same_nc<TYPE1, TYPE2>::value,
                  "Can only use compatible types in the copy");

    // Resize the output object to the correct size.
    to_vec.resize(from_view.size());
    const auto sizes = get_sizes(from_view);
    assert(sizes.size() == to_vec.size());
    for (typename data::jagged_vector_view<TYPE1>::size_type i = 0;
         i < from_view.size(); ++i) {
        to_vec[i].resize(sizes[i]);
    }

    // Perform the memory copy.
    return operator()(from_view, vecmem::get_data(to_vec), cptype);
}

template <typename TYPE>
std::vector<typename data::vector_view<TYPE>::size_type> copy::get_sizes(
    const data::jagged_vector_view<TYPE>& data) const {

    // Perform the operation using the private function.
    return get_sizes_impl(data.host_ptr(), data.size());
}

template <typename TYPE>
copy::event_type copy::set_sizes(
    const std::vector<typename data::vector_view<TYPE>::size_type>& sizes,
    data::jagged_vector_view<TYPE> data) const {

    // Finish early if possible.
    if ((sizes.size() == 0) && (data.size() == 0)) {
        return vecmem::copy::create_event();
    }
    // Make sure that the sizes match up.
    if (sizes.size() != data.size()) {
        throw std::runtime_error(
            "Incorrect size vector received for target jagged vector sizes");
    }
    // Make sure that the target jagged vector is either resizable, or it has
    // the correct sizes/capacities already.
    bool perform_copy = true;
    for (typename data::jagged_vector_view<TYPE>::size_type i = 0;
         i < data.size(); ++i) {
        if (data.host_ptr()[i].size_ptr() == nullptr) {
            perform_copy = false;
        } else if (perform_copy == true) {
            throw std::runtime_error(
                "Inconsistent target jagged vector view received for resizing");
        } else if (data.host_ptr()[i].capacity() != sizes[i]) {
            throw std::runtime_error(
                "Non-resizable jaggged vector does not match the requested "
                "size");
        }
    }
    // If no copy is necessary, we're done.
    if (perform_copy == false) {
        return vecmem::copy::create_event();
    }
    // Perform the copy with some internal knowledge of how resizable jagged
    // vector buffers work.
    do_copy(sizeof(typename data::vector_view<TYPE>::size_type) * sizes.size(),
            sizes.data(), data.host_ptr()->size_ptr(), type::unknown);

    // Return a new event.
    return create_event();
}

template <typename TYPE1, typename TYPE2>
void copy::copy_views_impl(
    const std::vector<typename data::vector_view<TYPE1>::size_type>& sizes,
    const data::vector_view<TYPE1>* from_view,
    data::vector_view<TYPE2>* to_view, type::copy_type cptype) const {

    // The input and output types are allowed to be different, but only by
    // const-ness.
    static_assert(std::is_same<TYPE1, TYPE2>::value ||
                      details::is_same_nc<TYPE1, TYPE2>::value,
                  "Can only use compatible types in the copy");

    // Some security checks.
    assert(from_view != nullptr);
    assert(to_view != nullptr);

    // Helper variable(s) used in the copy.
    const std::size_t size = sizes.size();
    [[maybe_unused]] std::size_t copy_ops = 0;

    // Perform the copy in multiple steps.
    for (std::size_t i = 0; i < size; ++i) {

        // Skip empty "inner vectors".
        if (sizes[i] == 0) {
            continue;
        }

        // Some sanity checks.
        assert(from_view[i].ptr() != nullptr);
        assert(to_view[i].ptr() != nullptr);
        assert(sizes[i] <= from_view[i].capacity());
        assert(sizes[i] <= to_view[i].capacity());

        // Perform the copy.
        do_copy(sizes[i] * sizeof(TYPE1), from_view[i].ptr(), to_view[i].ptr(),
                cptype);
        ++copy_ops;
    }

    // Let the user know what happened.
    VECMEM_DEBUG_MSG(2,
                     "Copied the payload of a jagged vector of type "
                     "\"%s\" with %lu copy operation(s)",
                     typeid(TYPE2).name(), copy_ops);
}

template <typename TYPE1, typename TYPE2>
void copy::copy_views_contiguous_impl(
    const std::vector<typename data::vector_view<TYPE1>::size_type>& sizes,
    const data::vector_view<TYPE1>* from_view,
    data::vector_view<TYPE2>* to_view, type::copy_type cptype) const {

    // The input and output types are allowed to be different, but only by
    // const-ness.
    static_assert(std::is_same<TYPE1, TYPE2>::value ||
                      details::is_same_nc<TYPE1, TYPE2>::value,
                  "Can only use compatible types in the copy");

    // Some security checks.
    assert(from_view != nullptr);
    assert(to_view != nullptr);
    assert(is_contiguous(from_view, sizes.size()));
    assert(is_contiguous(to_view, sizes.size()));

    // Helper variable(s) used in the copy.
    const std::size_t size = sizes.size();
    const std::size_t total_size =
        std::accumulate(sizes.begin(), sizes.end(),
                        static_cast<std::size_t>(0)) *
        sizeof(TYPE1);

    // Find the first non-empty element.
    for (std::size_t i = 0; i < size; ++i) {

        // Jump over empty elements.
        if (sizes[i] == 0) {
            continue;
        }

        // Some sanity checks.
        assert(from_view[i].ptr() != nullptr);
        assert(to_view[i].ptr() != nullptr);

        // Perform the copy.
        do_copy(total_size, from_view[i].ptr(), to_view[i].ptr(), cptype);
        break;
    }

    // Let the user know what happened.
    VECMEM_DEBUG_MSG(2,
                     "Copied the payload of a jagged vector of type "
                     "\"%s\" with 1 copy operation(s)",
                     typeid(TYPE2).name());
}

template <typename TYPE>
std::vector<typename data::vector_view<TYPE>::size_type> copy::get_sizes_impl(
    const data::vector_view<TYPE>* data, std::size_t size) const {

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
            // Wait for the copy operation to finish. With some backends
            // (khm... SYCL... khm...) copies can be asynchronous even into
            // non-pinned host memory.
            create_event()->wait();
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

template <typename TYPE>
bool copy::is_contiguous(const data::vector_view<TYPE>* data,
                         std::size_t size) {

    // We should never call this function for an empty jagged vector.
    assert(size > 0);

    // Check whether all memory blocks are contiguous.
    auto ptr = data[0].ptr();
    for (std::size_t i = 1; i < size; ++i) {
        if ((ptr + data[i - 1].capacity()) != data[i].ptr()) {
            return false;
        }
        ptr = data[i].ptr();
    }
    return true;
}

}  // namespace vecmem
