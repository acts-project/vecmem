#include <memory_resource>

#include "vecmem/memory/resources/base_resource.hpp"

namespace vecmem::memory::resources {
    base_resource * get_terminal_cuda_device_resource() noexcept;
    base_resource * get_terminal_cuda_host_resource() noexcept;
    base_resource * get_terminal_cuda_managed_resource() noexcept;
}