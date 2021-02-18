/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "../../utils/cuda_error_handling.hpp"
#include "../../utils/cuda_wrappers.hpp"

#include "vecmem/memory/resources/memory_resource.hpp"
#include "vecmem/memory/cuda/device_memory_resource.hpp"

#include <cuda_runtime_api.h>

namespace vecmem::cuda {
    device_memory_resource::device_memory_resource(
        int device
    ) :
        m_device(device >= 0 ? device : details::get_device())
    {
    }

    void * device_memory_resource::do_allocate(
        std::size_t bytes,
        std::size_t
    ) {
        int current_device = 0;

        VECMEM_CUDA_ERROR_CHECK(cudaGetDevice(&current_device));
        VECMEM_CUDA_ERROR_CHECK(cudaSetDevice(m_device));

        void * res;
        VECMEM_CUDA_ERROR_CHECK(cudaMalloc(&res, bytes));

        VECMEM_CUDA_ERROR_CHECK(cudaSetDevice(current_device));

        return res;
    }

    void device_memory_resource::do_deallocate(
        void * p,
        std::size_t,
        std::size_t
    ) {
        int current_device = 0;

        VECMEM_CUDA_ERROR_CHECK(cudaGetDevice(&current_device));
        VECMEM_CUDA_ERROR_CHECK(cudaSetDevice(m_device));

        VECMEM_CUDA_ERROR_CHECK(cudaFree(p));

        VECMEM_CUDA_ERROR_CHECK(cudaSetDevice(current_device));
    }

    bool device_memory_resource::do_is_equal(
        const memory_resource & other
    ) const noexcept {
        const device_memory_resource * c;
        c = dynamic_cast<const device_memory_resource *>(&other);

        /*
         * The equality check here is ever so slightly more difficult. Not only
         * does the other object need to be a device memory resource, it must
         * also target the same device.
         */
        return c != nullptr && c->m_device == m_device;
    }
}
