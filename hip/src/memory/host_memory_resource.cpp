/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/hip/host_memory_resource.hpp"

#include "../utils/hip_error_handling.hpp"
#include "vecmem/utils/debug.hpp"

// HIP include(s).
#include <hip/hip_runtime_api.h>

// System include(s).
#include <cassert>
#include <stdexcept>

namespace vecmem::hip {

host_memory_resource::host_memory_resource() = default;

host_memory_resource::~host_memory_resource() noexcept = default;

void* host_memory_resource::do_allocate(std::size_t nbytes, std::size_t) {

    if (nbytes == 0) {
        throw std::bad_alloc();
    }

    // Allocate the memory.
    void* result = nullptr;
    VECMEM_HIP_ERROR_CHECK(hipHostMalloc(&result, nbytes));
    VECMEM_DEBUG_MSG(2, "Allocated %ld bytes at %p", nbytes, result);
    return result;
}

void host_memory_resource::do_deallocate(void* ptr, std::size_t bytes,
                                         std::size_t) {

    assert(ptr != nullptr);
    if (bytes == 0u) {
        return;
    }

    // Free the memory.
    VECMEM_DEBUG_MSG(2, "De-allocating memory at %p", ptr);
    VECMEM_HIP_ERROR_CHECK(hipHostFree(ptr));
}

bool host_memory_resource::do_is_equal(
    const memory_resource& other) const noexcept {

    // The two are equal if they are of the same type.
    return (dynamic_cast<const host_memory_resource*>(&other) != nullptr);
}

}  // namespace vecmem::hip
