/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/hip/managed_memory_resource.hpp"

#include "../utils/hip_error_handling.hpp"
#include "vecmem/utils/debug.hpp"

// HIP include(s).
#include <hip/hip_runtime_api.h>

// System include(s).
#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace vecmem::hip {

managed_memory_resource::managed_memory_resource() = default;

managed_memory_resource::~managed_memory_resource() noexcept = default;

void *managed_memory_resource::do_allocate(std::size_t bytes,
                                           std::size_t alignment) {

    if (bytes == 0) {
        throw std::bad_alloc();
    }

    // Allocate the memory.
    void *res = nullptr;
    VECMEM_HIP_ERROR_CHECK(hipMallocManaged(&res, bytes));

    // HIP cannot be asked for an alignment, so check the one that we got.
    if (reinterpret_cast<std::uintptr_t>(res) % alignment != 0) {
        VECMEM_DEBUG_MSG(1, "Allocation at %p is not aligned to %ld bytes", res,
                         alignment);
        VECMEM_HIP_ERROR_CHECK(hipFree(res));
        throw std::bad_alloc();
    }
    VECMEM_DEBUG_MSG(2, "Allocated %ld bytes at %p", bytes, res);
    return res;
}

void managed_memory_resource::do_deallocate(void *p, std::size_t bytes,
                                            std::size_t) {

    assert(p != nullptr);
    if (bytes == 0u) {
        return;
    }

    // Free the memory.
    VECMEM_DEBUG_MSG(2, "De-allocating memory at %p", p);
    VECMEM_HIP_ERROR_CHECK(hipFree(p));
}

bool managed_memory_resource::do_is_equal(
    const memory_resource &other) const noexcept {

    // The two are equal if they are of the same type.
    return (dynamic_cast<const managed_memory_resource *>(&other) != nullptr);
}

}  // namespace vecmem::hip
