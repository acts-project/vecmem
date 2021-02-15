#include <memory>
#include <memory_resource>

#include <cuda_runtime.h>

#include "vecmem/memory/cuda/resources/terminal/host_resource.hpp"

namespace vecmem::memory::resources::terminal {
    void * cuda_host_resource::do_allocate(
        std::size_t bytes,
        std::size_t
    ) {
        void * res;
        cudaMallocHost(&res, bytes);
        return res;
    }

    void cuda_host_resource::do_deallocate(
        void * p,
        std::size_t,
        std::size_t
    ) {
        cudaFreeHost(p);
    }

    bool cuda_host_resource::do_is_equal(
        const std::pmr::memory_resource & other
    ) const noexcept {
        return false;
    }

    bool cuda_host_resource::is_host_accessible() const {
        return true;
    }
}