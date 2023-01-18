/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "vecmem/memory/coalescing_memory_resource.hpp"

#include <cassert>
#include <cstddef>
#include <initializer_list>

#include "vecmem/memory/memory_resource.hpp"

namespace vecmem {
coalescing_memory_resource::coalescing_memory_resource(
    std::vector<std::reference_wrapper<memory_resource>> &&upstreams)
    : m_upstreams(upstreams) {}

void *coalescing_memory_resource::do_allocate(std::size_t size,
                                              std::size_t align) {
    /*
     * Try to allocate with each of the upstream resources.
     */
    for (memory_resource &res : m_upstreams) {
        try {
            /*
             * Try to allocate the memory, and store the result with the
             * allocator reference in the allocation map.
             */
            void *ptr = res.allocate(size, align);

            m_allocations.emplace(ptr, res);

            return ptr;
        } catch (std::bad_alloc &) {
            /*
             * If we cannot allocate with this resource, try the next one.
             */
            continue;
        }
    }

    /*
     * If all resources fail to allocate, then we do as well.
     */
    throw std::bad_alloc();
}

void coalescing_memory_resource::do_deallocate(void *ptr, std::size_t size,
                                               std::size_t align) {
    /*
     * Fetch the resource used to allocate this pointer.
     */
    auto nh = m_allocations.extract(ptr);

    /*
     * For debug builds, throw an assertion error if we do not know this
     * allocation.
     */
    assert(nh);

    /*
     * If we know who allocated this memory, forward the deallocation request
     * to them.
     */
    memory_resource &res = nh.mapped();

    res.deallocate(nh.key(), size, align);
}
}  // namespace vecmem
