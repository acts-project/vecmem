#include <memory>

#include "vecmem/memory/resources/memory_resource.hpp"
#include "vecmem/memory/resources/terminal/malloc_resource.hpp"

namespace vecmem::memory::resources::terminal {
    void * malloc_resource::do_allocate(
        std::size_t bytes,
        std::size_t
    ) {
        return malloc(bytes);
    }

    void malloc_resource::do_deallocate(
        void * p,
        std::size_t,
        std::size_t
    ) {
        free(p);
    }

    bool malloc_resource::do_is_equal(
        const memory_resource & other
    ) const noexcept {
        return false;
    }

    bool malloc_resource::is_host_accessible() const {
        return true;
    }
}