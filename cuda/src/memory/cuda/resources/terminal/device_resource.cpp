#include <memory>

#include <cuda_runtime.h>

#include "vecmem/memory/resources/memory_resource.hpp"
#include "vecmem/memory/cuda/resources/terminal/device_resource.hpp"

namespace vecmem::memory::resources::terminal {
    void * cuda_device_resource::do_allocate(
        std::size_t bytes,
        std::size_t
    ) {
        void * res;
        cudaMalloc(&res, bytes);
        return res;
    }

    void cuda_device_resource::do_deallocate(
        void * p,
        std::size_t,
        std::size_t
    ) {
        cudaFree(p);
    }

    bool cuda_device_resource::do_is_equal(
        const memory_resource & other
    ) const noexcept {
        return false;
    }

    bool cuda_device_resource::is_host_accessible() const {
        return false;
    }
}