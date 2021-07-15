/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// VecMem include(s).
#include "vecmem/containers/data/jagged_vector_buffer.hpp"
#include "vecmem/containers/data/jagged_vector_view.hpp"
#include "vecmem/containers/data/vector_buffer.hpp"
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/memory/memory_resource.hpp"

// System include(s).
#include <cstddef>
#include <vector>

namespace vecmem {

/// Class implementing (synchronous) host <-> device memory copies
///
/// Since most of the logic of explicitly copying the payload of vecmem
/// containers between the host and device is independent of the exact GPU
/// language used, this common base class is used for implementing most of
/// that logic.
///
/// Language specific @c copy classes should only need to re-implement the
/// @c do_copy function, everything else should be provided by this class.
///
class copy {

public:
    /// Wrapper struct around the @c copy_type enumeration
    ///
    /// The code does not use "enum struct type" to declare the copy type, as
    /// that unnecessarily makes it hard to use these values as array indices
    /// in client code.
    ///
    struct type {
        /// Types of memory copies to handle
        enum copy_type {
            /// Copy operation between the host and a device
            host_to_device = 0,
            /// Copy operation between a device and the host
            device_to_host = 1,
            /// Copy operation on the host
            host_to_host = 2,
            /// Copy operation between two devices
            device_to_device = 3,
            /// Unknown copy type, determined at runtime
            unknown = 4,
            /// The number of copy types, useful for technical reasons
            count = 5
        };  // enum copy_type
    };      // struct type

    /// @name 1-dimensional vector data handling functions
    /// @{

    /// Set up the internal state of a vector buffer correctly on a device
    template <typename TYPE>
    void setup(data::vector_view<TYPE>& data);

    /// Copy a 1-dimensional vector to the specified memory resource
    template <typename TYPE>
    data::vector_buffer<TYPE> to(const data::vector_view<TYPE>& data,
                                 memory_resource& resource,
                                 type::copy_type cptype = type::unknown);

    /// Copy a 1-dimensional vector's data between two existing memory blocks
    template <typename TYPE>
    void operator()(const data::vector_view<TYPE>& from,
                    data::vector_view<TYPE>& to,
                    type::copy_type cptype = type::unknown);

    /// Copy a 1-dimensional vector's data into a vector object
    template <typename TYPE1, typename TYPE2, typename ALLOC>
    void operator()(const data::vector_view<TYPE1>& from,
                    std::vector<TYPE2, ALLOC>& to,
                    type::copy_type cptype = type::unknown);

    /// @}

    /// @name Jagged vector data handling functions
    /// @{

    /// Copy the internal state of a jagged vector buffer to the target device
    template <typename TYPE>
    void setup(data::jagged_vector_buffer<TYPE>& data);

    /// Copy a jagged vector to the specified memory resource
    template <typename TYPE>
    data::jagged_vector_buffer<TYPE> to(
        const data::jagged_vector_view<TYPE>& data, memory_resource& resource,
        memory_resource* host_access_resource = nullptr,
        type::copy_type cptype = type::unknown);

    /// Copy a jagged vector to the specified memory resource
    template <typename TYPE>
    data::jagged_vector_buffer<TYPE> to(
        const data::jagged_vector_buffer<TYPE>& data, memory_resource& resource,
        memory_resource* host_access_resource = nullptr,
        type::copy_type cptype = type::unknown);

    /// Copy a jagged vector's data between two existing allocations
    template <typename TYPE>
    void operator()(const data::jagged_vector_view<TYPE>& from,
                    data::jagged_vector_view<TYPE>& to,
                    type::copy_type cptype = type::unknown);

    /// Copy a jagged vector's data between two existing allocations
    template <typename TYPE>
    void operator()(const data::jagged_vector_view<TYPE>& from,
                    data::jagged_vector_buffer<TYPE>& to,
                    type::copy_type cptype = type::unknown);

    /// Copy a jagged vector's data between two existing allocations
    template <typename TYPE>
    void operator()(const data::jagged_vector_buffer<TYPE>& from,
                    data::jagged_vector_view<TYPE>& to,
                    type::copy_type cptype = type::unknown);

    /// Copy a jagged vector's data between two existing allocations
    template <typename TYPE>
    void operator()(const data::jagged_vector_buffer<TYPE>& from,
                    data::jagged_vector_buffer<TYPE>& to,
                    type::copy_type cptype = type::unknown);

    /// Copy a jagged vector's data into a vector object
    template <typename TYPE1, typename TYPE2, typename ALLOC1, typename ALLOC2>
    void operator()(const data::jagged_vector_view<TYPE1>& from,
                    std::vector<std::vector<TYPE2, ALLOC2>, ALLOC1>& to,
                    type::copy_type cptype = type::unknown);

    /// Copy a jagged vector's data into a vector object
    template <typename TYPE1, typename TYPE2, typename ALLOC1, typename ALLOC2>
    void operator()(const data::jagged_vector_buffer<TYPE1>& from,
                    std::vector<std::vector<TYPE2, ALLOC2>, ALLOC1>& to,
                    type::copy_type cptype = type::unknown);

    /// @}

protected:
    /// Perform a "low level" memory copy
    virtual void do_copy(std::size_t size, const void* from, void* to,
                         type::copy_type cptype);
    /// Perform a "low level" memory filling operation
    virtual void do_memset(std::size_t size, void* ptr, int value);

private:
    /// Helper function performing the copy of a jagged array/vector
    template <typename TYPE>
    void copy_views(std::size_t size, const data::vector_view<TYPE>* from,
                    data::vector_view<TYPE>* to, type::copy_type cptype);
    /// Helper function for getting the size of a resizable 1D buffer
    template <typename TYPE>
    typename data::vector_view<TYPE>::size_type get_size(
        const data::vector_view<TYPE>& data);

};  // class copy

}  // namespace vecmem

// Include the implementation.
#include "vecmem/utils/impl/copy.ipp"
