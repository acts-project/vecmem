/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include "vecmem/utils/cuda/async_copy.hpp"

#include "../cuda_error_handling.hpp"
#include "../cuda_wrappers.hpp"
#include "vecmem/utils/debug.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

// System include(s).
#include <cassert>
#include <string>

namespace vecmem::cuda {

/// Helper array for translating between the vecmem and CUDA copy type
/// definitions
static constexpr cudaMemcpyKind copy_type_translator[copy::type::count] = {
    cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyHostToHost,
    cudaMemcpyDeviceToDevice, cudaMemcpyDefault};

/// Helper array for providing a printable name for the copy type definitions
static const std::string copy_type_printer[copy::type::count] = {
    "host to device", "device to host", "host to host", "device to device",
    "unknown"};

async_copy::async_copy(const stream_wrapper& stream) : m_stream(stream) {}

void async_copy::do_copy(std::size_t size, const void* from_ptr, void* to_ptr,
                         type::copy_type cptype) {

    // Check if anything needs to be done.
    if (size == 0) {
        VECMEM_DEBUG_MSG(5, "Skipping unnecessary memory copy");
        return;
    }

    // Some sanity checks.
    assert(from_ptr != nullptr);
    assert(to_ptr != nullptr);
    assert(static_cast<int>(cptype) >= 0);
    assert(static_cast<int>(cptype) < static_cast<int>(copy::type::count));

    // Perform the copy.
    VECMEM_CUDA_ERROR_CHECK(cudaMemcpyAsync(to_ptr, from_ptr, size,
                                            copy_type_translator[cptype],
                                            details::get_stream(m_stream)));

    // Let the user know what happened.
    VECMEM_DEBUG_MSG(4,
                     "Initiated asynchronous %s memory copy of %lu bytes "
                     "from %p to %p",
                     copy_type_printer[cptype].c_str(), size, from_ptr, to_ptr);
}

void async_copy::do_memset(std::size_t size, void* ptr, int value) {

    // Check if anything needs to be done.
    if (size == 0) {
        VECMEM_DEBUG_MSG(5, "Skipping unnecessary memory filling");
        return;
    }

    // Some sanity checks.
    assert(ptr != nullptr);

    // Perform the operation.
    VECMEM_CUDA_ERROR_CHECK(
        cudaMemsetAsync(ptr, value, size, details::get_stream(m_stream)));

    // Let the user know what happened.
    VECMEM_DEBUG_MSG(
        4, "Initiated setting %lu bytes to %i at %p asynchronously with CUDA",
        size, value, ptr);
}

}  // namespace vecmem::cuda
