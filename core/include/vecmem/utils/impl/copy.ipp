/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// VecMem include(s).
#include "vecmem/containers/details/resize_jagged_vector.hpp"
#include "vecmem/containers/jagged_vector.hpp"
#include "vecmem/edm/details/schema_traits.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/utils/debug.hpp"
#include "vecmem/utils/type_traits.hpp"

// System include(s).
#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <type_traits>

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
    data::vector_buffer<std::remove_cv_t<TYPE>> result(get_size(data),
                                                       resource);
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
        // Select what type of copy this should be. Keeping in mind that we copy
        // from a variable on the host stack. So the question is just whether
        // the target is the host, or a device.
        type::copy_type size_cptype = type::unknown;
        switch (cptype) {
            case type::host_to_device:
            case type::device_to_device:
                size_cptype = type::host_to_device;
                break;
            case type::device_to_host:
            case type::host_to_host:
                size_cptype = type::host_to_host;
                break;
            default:
                break;
        }
        // Perform the copy.
        do_copy(sizeof(typename data::vector_view<TYPE2>::size_type), &size,
                to_view.size_ptr(), size_cptype);
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
    details::resize_jagged_vector(to_vec, from_view.size());
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
        std::ostringstream msg;
        msg << "sizes.size() (" << sizes.size() << ") != data.size() ("
            << data.size() << ")";
        throw std::length_error(msg.str());
    }
    // Make sure that the target jagged vector is either resizable, or it has
    // the correct sizes/capacities already.
    bool perform_copy = true;
    for (typename data::jagged_vector_view<TYPE>::size_type i = 0;
         i < data.size(); ++i) {
        if (data.host_ptr()[i].size_ptr() == nullptr) {
            perform_copy = false;
            if (data.host_ptr()[i].capacity() != sizes[i]) {
                std::ostringstream msg;
                msg << "data.host_ptr()[" << i << "].capacity() ("
                    << data.host_ptr()[i].capacity() << ") != sizes[" << i
                    << "] (" << sizes[i] << ")";
                throw std::length_error(msg.str());
            }
        } else if (perform_copy == false) {
            throw std::runtime_error(
                "Inconsistent target jagged vector view received for resizing");
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

template <typename SCHEMA>
copy::event_type copy::setup(edm::view<SCHEMA> data) const {

    // For empty containers nothing needs to be done.
    if (data.capacity() == 0) {
        return vecmem::copy::create_event();
    }

    // Copy the data layout to the device, if needed.
    if (data.layout().ptr() != data.host_layout().ptr()) {
        operator()(data.host_layout(), data.layout(), type::unknown);
    }

    // Initialize the "size variable(s)" correctly on the buffer.
    if (data.size().ptr() != nullptr) {
        memset(data.size(), 0);
    }
    VECMEM_DEBUG_MSG(3,
                     "Prepared an SoA container of capacity %u "
                     "for use on a device (layout: {%u, %p}, size: {%u, %p})",
                     data.capacity(), data.layout().size(),
                     static_cast<void*>(data.layout().ptr()),
                     data.size().size(), static_cast<void*>(data.size().ptr()));

    // Return a new event.
    return create_event();
}

template <typename... VARTYPES>
copy::event_type copy::memset(edm::view<edm::schema<VARTYPES...>> data,
                              int value) const {

    // For buffers, we can do this in one go.
    if (data.payload().ptr() != nullptr) {
        memset(data.payload(), value);
    } else {
        // Do the operation using the helper function, recursively.
        memset_impl<0>(data, value);
    }

    // Return a new event.
    return create_event();
}

template <typename... VARTYPES1, typename... VARTYPES2>
copy::event_type copy::operator()(
    const edm::view<edm::schema<VARTYPES1...>>& from_view,
    edm::view<edm::schema<VARTYPES2...>> to_view,
    type::copy_type cptype) const {

    // The input and output types are allowed to be different, but only by
    // const-ness.
    static_assert(sizeof...(VARTYPES1) == sizeof...(VARTYPES2),
                  "Can only use compatible types in the copy");
    static_assert(
        std::conjunction_v<std::is_same<VARTYPES1, VARTYPES2>...> ||
            std::conjunction_v<
                edm::type::details::is_same_nc<VARTYPES1, VARTYPES2>...>,
        "Can only use compatible types in the copy");

    // First, handle the simple case, when both views have a contiguous memory
    // layout.
    if ((from_view.payload().ptr() != nullptr) &&
        (to_view.payload().ptr() != nullptr) &&
        (from_view.payload().size() == to_view.payload().size())) {

        // If the "common size" is zero, we're done.
        if (from_view.payload().size() == 0) {
            return vecmem::copy::create_event();
        }

        // Let the user know what's happening.
        VECMEM_DEBUG_MSG(2, "Performing simple SoA copy of %u bytes",
                         from_view.payload().size());

        // Copy the payload with a single copy operation.
        operator()(from_view.payload(), to_view.payload(), cptype);

        // If the target view is resizable, set its size.
        if (to_view.size().ptr() != nullptr) {
            // If the source is also resizable, the situation should be simple.
            if (from_view.size().ptr() != nullptr) {
                // Check that the sizes are the same.
                if (from_view.size().size() != to_view.size().size()) {
                    std::ostringstream msg;
                    msg << "from_view.size().size() ("
                        << from_view.size().size()
                        << ") != to_view.size().size() ("
                        << to_view.size().size() << ")";
                    throw std::length_error(msg.str());
                }
                // Perform a dumb copy.
                operator()(from_view.size(), to_view.size(), cptype);
            } else {
                // If not, then copy the size(s) recursively.
                copy_sizes_impl<0>(from_view, to_view, cptype);
            }
        }

        // Create a synchronization event.
        return create_event();
    }

    // If not, then do an un-optimized copy, variable-by-variable.
    copy_payload_impl<0>(from_view, to_view, cptype);

    // Return a new event.
    return create_event();
}

template <typename... VARTYPES1, typename... VARTYPES2>
copy::event_type copy::operator()(
    const edm::view<edm::schema<VARTYPES1...>>& from_view,
    edm::host<edm::schema<VARTYPES2...>>& to_vec,
    type::copy_type cptype) const {

    // The input and output types are allowed to be different, but only by
    // const-ness.
    static_assert(sizeof...(VARTYPES1) == sizeof...(VARTYPES2),
                  "Can only use compatible types in the copy");
    static_assert(std::is_same<edm::schema<VARTYPES1...>,
                               edm::schema<VARTYPES2...>>::value ||
                      details::is_same_nc<edm::schema<VARTYPES1...>,
                                          edm::schema<VARTYPES2...>>::value,
                  "Can only use compatible types in the copy");

    // Resize the output object to the correct size(s).
    resize_impl<0>(from_view, to_vec, cptype);

    // Perform the memory copy.
    return operator()(from_view, vecmem::get_data(to_vec), cptype);
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

template <std::size_t INDEX, typename... VARTYPES>
void copy::memset_impl(edm::view<edm::schema<VARTYPES...>> data,
                       int value) const {

    // Scalars do not have their own dedicated @c memset functions.
    if constexpr (edm::type::details::is_scalar<typename std::tuple_element<
                      INDEX, std::tuple<VARTYPES...>>::type>::value) {
        do_memset(sizeof(typename std::tuple_element<
                         INDEX, std::tuple<VARTYPES...>>::type::type),
                  data.template get<INDEX>(), value);
    } else {
        // But vectors and jagged vectors do.
        memset(data.template get<INDEX>(), value);
    }
    // Recurse into the next variable.
    if constexpr (sizeof...(VARTYPES) > (INDEX + 1)) {
        memset_impl<INDEX + 1>(data, value);
    }
}

template <std::size_t INDEX, typename... VARTYPES1, typename... VARTYPES2>
void copy::resize_impl(const edm::view<edm::schema<VARTYPES1...>>& from_view,
                       edm::host<edm::schema<VARTYPES2...>>& to_vec,
                       [[maybe_unused]] type::copy_type cptype) const {

    // The input and output types are allowed to be different, but only by
    // const-ness.
    static_assert(sizeof...(VARTYPES1) == sizeof...(VARTYPES2),
                  "Can only use compatible types in the copy");
    static_assert(std::is_same<edm::schema<VARTYPES1...>,
                               edm::schema<VARTYPES2...>>::value ||
                      details::is_same_nc<edm::schema<VARTYPES1...>,
                                          edm::schema<VARTYPES2...>>::value,
                  "Can only use compatible types in the copy");
    // The target is a host container, so the copy type can't be anything
    // targeting a device.
    assert((cptype == type::device_to_host) || (cptype == type::host_to_host) ||
           (cptype == type::unknown));

    // First, handle containers with no jagged vectors in them.
    if constexpr ((std::disjunction_v<
                       edm::type::details::is_jagged_vector<VARTYPES1>...> ==
                   false) &&
                  (std::disjunction_v<
                       edm::type::details::is_jagged_vector<VARTYPES2>...> ==
                   false)) {
        // The single size of the source container.
        typename edm::view<edm::schema<VARTYPES1...>>::size_type size =
            from_view.capacity();
        // If the container is resizable, take its size.
        if (from_view.size().ptr() != nullptr) {
            assert(
                from_view.size().size() ==
                sizeof(
                    typename edm::view<edm::schema<VARTYPES1...>>::size_type));
            do_copy(
                sizeof(
                    typename edm::view<edm::schema<VARTYPES1...>>::size_type),
                from_view.size().ptr(), &size, cptype);
            create_event()->wait();
        }
        // Resize the target container.
        VECMEM_DEBUG_MSG(4, "Resizing a (non-jagged) container to size %u",
                         size);
        to_vec.resize(size);
    } else {
        // Resize vector and jagged vector variables one by one. Note that
        // evaluation order matters here. Since all jagged vectors are vectors,
        // but not all vectors are jagged vectors. ;-)
        if constexpr (edm::type::details::is_jagged_vector<
                          typename std::tuple_element<
                              INDEX, std::tuple<VARTYPES1...>>::type>::value) {
            // Get the sizes of this jagged vector.
            auto sizes = get_sizes(from_view.template get<INDEX>());
            // Set the "outer size" of the jagged vector.
            VECMEM_DEBUG_MSG(
                4, "Resizing jagged vector variable at index %lu to size %lu",
                INDEX, sizes.size());
            details::resize_jagged_vector(to_vec.template get<INDEX>(),
                                          sizes.size());
            // Set the "inner sizes" of the jagged vector.
            for (std::size_t i = 0; i < sizes.size(); ++i) {
                to_vec.template get<INDEX>()[i].resize(sizes[i]);
            }
        } else if constexpr (edm::type::details::is_vector<
                                 typename std::tuple_element<
                                     INDEX,
                                     std::tuple<VARTYPES1...>>::type>::value) {
            // Get the size of this vector.
            auto size = get_size(from_view.template get<INDEX>());
            // Resize the target vector.
            VECMEM_DEBUG_MSG(4,
                             "Resizing vector variable at index %lu to size %u",
                             INDEX, size);
            to_vec.template get<INDEX>().resize(size);
        }
        // Call this function recursively.
        if constexpr (sizeof...(VARTYPES1) > (INDEX + 1)) {
            resize_impl<INDEX + 1>(from_view, to_vec, cptype);
        }
    }
}

template <std::size_t INDEX, typename... VARTYPES1, typename... VARTYPES2>
void copy::copy_sizes_impl(
    [[maybe_unused]] const edm::view<edm::schema<VARTYPES1...>>& from_view,
    [[maybe_unused]] edm::view<edm::schema<VARTYPES2...>> to_view,
    [[maybe_unused]] type::copy_type cptype) const {

    // The input and output types are allowed to be different, but only by
    // const-ness.
    static_assert(sizeof...(VARTYPES1) == sizeof...(VARTYPES2),
                  "Can only use compatible types in the copy");
    static_assert(std::is_same<edm::schema<VARTYPES1...>,
                               edm::schema<VARTYPES2...>>::value ||
                      details::is_same_nc<edm::schema<VARTYPES1...>,
                                          edm::schema<VARTYPES2...>>::value,
                  "Can only use compatible types in the copy");

    // This should only be called for a resizable target container, with
    // a non-resizable source container.
    assert(to_view.size().ptr() != nullptr);
    assert(from_view.size().ptr() == nullptr);

    // First, handle containers with no jagged vectors in them.
    if constexpr ((std::disjunction_v<
                       edm::type::details::is_jagged_vector<VARTYPES1>...> ==
                   false) &&
                  (std::disjunction_v<
                       edm::type::details::is_jagged_vector<VARTYPES2>...> ==
                   false)) {
        // Size of the source container.
        typename edm::view<edm::schema<VARTYPES1...>>::size_type size =
            from_view.capacity();
        // Choose the copy type.
        type::copy_type size_cptype = type::unknown;
        switch (cptype) {
            case type::host_to_device:
            case type::device_to_device:
                size_cptype = type::host_to_device;
                break;
            case type::host_to_host:
            case type::device_to_host:
                size_cptype = type::host_to_host;
                break;
            default:
                break;
        }
        // Set the size of the target container.
        do_copy(
            sizeof(typename edm::view<edm::schema<VARTYPES2...>>::size_type),
            &size, to_view.size().ptr(), size_cptype);
    } else {
        // For the jagged vector case we recursively copy the sizes of every
        // jagged vector variable. The rest of the variables are not resizable
        // in such a container, so they are ignored here.
        if constexpr (edm::type::details::is_jagged_vector<
                          typename std::tuple_element<
                              INDEX, std::tuple<VARTYPES1...>>::type>::value) {
            // Copy the sizes for this variable.
            const auto sizes = get_sizes(from_view.template get<INDEX>());
            set_sizes(sizes, to_view.template get<INDEX>());
        }
        // Call this function recursively.
        if constexpr (sizeof...(VARTYPES1) > (INDEX + 1)) {
            copy_sizes_impl<INDEX + 1>(from_view, to_view, cptype);
        }
    }
}

template <std::size_t INDEX, typename... VARTYPES1, typename... VARTYPES2>
void copy::copy_payload_impl(
    const edm::view<edm::schema<VARTYPES1...>>& from_view,
    edm::view<edm::schema<VARTYPES2...>> to_view,
    type::copy_type cptype) const {

    // The input and output types are allowed to be different, but only by
    // const-ness.
    static_assert(sizeof...(VARTYPES1) == sizeof...(VARTYPES2),
                  "Can only use compatible types in the copy");
    static_assert(std::is_same<edm::schema<VARTYPES1...>,
                               edm::schema<VARTYPES2...>>::value ||
                      details::is_same_nc<edm::schema<VARTYPES1...>,
                                          edm::schema<VARTYPES2...>>::value,
                  "Can only use compatible types in the copy");

    // Scalars do not have their own dedicated @c copy functions.
    if constexpr (edm::type::details::is_scalar<typename std::tuple_element<
                      INDEX, std::tuple<VARTYPES1...>>::type>::value) {
        do_copy(sizeof(typename std::tuple_element<
                       INDEX, std::tuple<VARTYPES1...>>::type::type),
                from_view.template get<INDEX>(), to_view.template get<INDEX>(),
                cptype);
    } else {
        // But vectors and jagged vectors do.
        operator()(from_view.template get<INDEX>(),
                   to_view.template get<INDEX>(), cptype);
    }
    // Recurse into the next variable.
    if constexpr (sizeof...(VARTYPES1) > (INDEX + 1)) {
        copy_payload_impl<INDEX + 1>(from_view, to_view, cptype);
    }
}

}  // namespace vecmem
