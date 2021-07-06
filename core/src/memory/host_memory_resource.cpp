/**
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "vecmem/memory/host_memory_resource.hpp"

#include <memory>

#include "vecmem/memory/memory_resource.hpp"

namespace vecmem {
void *host_memory_resource::do_allocate(std::size_t bytes, std::size_t) {
    return malloc(bytes);
}

void host_memory_resource::do_deallocate(void *p, std::size_t, std::size_t) {
    free(p);
}

bool host_memory_resource::do_is_equal(
    const memory_resource &other) const noexcept {
    /*
     * All malloc resources are equal to each other, because they have no
     * internal state. Of course they have a shared underlying state in the
     * form of the underlying C library memory manager, but that is not
     * relevant for us.
     */
    const host_memory_resource *c;
    c = dynamic_cast<const host_memory_resource *>(&other);

    return c != nullptr;
}
}  // namespace vecmem
