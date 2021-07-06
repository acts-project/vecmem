/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "vecmem/memory/cuda/host_memory_resource.hpp"

#include <cuda_runtime_api.h>

#include "../../utils/cuda_error_handling.hpp"
#include "vecmem/memory/memory_resource.hpp"

namespace vecmem::cuda {
void *host_memory_resource::do_allocate(std::size_t bytes, std::size_t) {
    void *res;
    VECMEM_CUDA_ERROR_CHECK(cudaMallocHost(&res, bytes));
    return res;
}

void host_memory_resource::do_deallocate(void *p, std::size_t, std::size_t) {
    VECMEM_CUDA_ERROR_CHECK(cudaFreeHost(p));
}

bool host_memory_resource::do_is_equal(
    const memory_resource &other) const noexcept {
    const host_memory_resource *c;
    c = dynamic_cast<const host_memory_resource *>(&other);

    return c != nullptr;
}
}  // namespace vecmem::cuda
