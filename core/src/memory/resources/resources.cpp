#include "vecmem/memory/resources/terminal/malloc_resource.hpp"
#include "vecmem/memory/resources/base_resource.hpp"

namespace vecmem::memory::resources {
    base_resource * get_terminal_malloc_resource() {
        static terminal::malloc_resource res;
        return &res;
    }
}