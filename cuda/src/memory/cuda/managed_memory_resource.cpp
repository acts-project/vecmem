/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "../../utils/cuda_error_handling.hpp"

#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/memory/cuda/managed_memory_resource.hpp"

#include <cuda_runtime_api.h>

namespace vecmem::cuda {
    void * managed_memory_resource::do_allocate(
        std::size_t bytes,
        std::size_t
    ) {
        void * res;
        VECMEM_CUDA_ERROR_CHECK(cudaMallocManaged(&res, bytes));
        return res;
    }

    void managed_memory_resource::do_deallocate(
        void * p,
        std::size_t,
        std::size_t
    ) {
        VECMEM_CUDA_ERROR_CHECK(cudaFree(p));
    }

    bool managed_memory_resource::do_is_equal(
        const memory_resource & other
    ) const noexcept {
        const managed_memory_resource * c;
        c = dynamic_cast<const managed_memory_resource *>(&other);

        return c != nullptr;
    }
}
