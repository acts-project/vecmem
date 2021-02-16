#include <memory>

#include <cuda_runtime_api.h>

#include "vecmem/memory/resources/memory_resource.hpp"
#include "vecmem/memory/cuda/resources/terminal/device_resource.hpp"
#include "vecmem/utils/cuda_error_handling.hpp"

namespace vecmem::memory::resources::terminal {
    void * cuda_device_resource::do_allocate(
        std::size_t bytes,
        std::size_t
    ) {
        void * res;
        VECMEM_CUDA_ERROR_CHECK(cudaMalloc(&res, bytes));
        return res;
    }

    void cuda_device_resource::do_deallocate(
        void * p,
        std::size_t,
        std::size_t
    ) {
        VECMEM_CUDA_ERROR_CHECK(cudaFree(p));
    }

    bool cuda_device_resource::do_is_equal(
        const memory_resource &
    ) const noexcept {
        return false;
    }

    bool cuda_device_resource::is_host_accessible() const {
        return false;
    }
}