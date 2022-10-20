/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/cuda/host_memory_resource.hpp"

#include "../../utils/cuda_error_handling.hpp"
#include "vecmem/utils/debug.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

namespace vecmem::cuda {

void *host_memory_resource::do_allocate(std::size_t bytes, std::size_t) {

    // Allocate the memory.
    void *res = nullptr;
    VECMEM_CUDA_ERROR_CHECK(cudaMallocHost(&res, bytes));
    VECMEM_DEBUG_MSG(2, "Allocated %ld bytes at %p", bytes, res);
    return res;
}

void host_memory_resource::do_deallocate(void *p, std::size_t, std::size_t) {

    // Free the memory.
    VECMEM_DEBUG_MSG(2, "De-allocating memory at %p", p);
    VECMEM_CUDA_ERROR_CHECK(cudaFreeHost(p));
}

bool host_memory_resource::do_is_equal(
    const memory_resource &other) const noexcept {
    const host_memory_resource *c;
    c = dynamic_cast<const host_memory_resource *>(&other);

    return c != nullptr;
}

}  // namespace vecmem::cuda
