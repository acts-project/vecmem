#include "vecmem/memory/resources/memory_resource.hpp"
#include "vecmem/memory/resources/base_resource.hpp"
#include "vecmem/memory/cuda/resources/terminal/device_resource.hpp"
#include "vecmem/memory/cuda/resources/terminal/host_resource.hpp"
#include "vecmem/memory/cuda/resources/terminal/managed_resource.hpp"

namespace vecmem::memory::resources {
    base_resource * get_terminal_cuda_device_resource() {
        static terminal::cuda_device_resource res;
        return &res;
    }

    base_resource * get_terminal_cuda_host_resource() {
        static terminal::cuda_host_resource res;
        return &res;
    }

    base_resource * get_terminal_cuda_managed_resource() {
        static terminal::cuda_managed_resource res;
        return &res;
    }

}