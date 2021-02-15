#pragma once

#include "vecmem/memory/resources/base_resource.hpp"

namespace vecmem::memory::resources {
    base_resource * get_terminal_malloc_resource() noexcept;
}