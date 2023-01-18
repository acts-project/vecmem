/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "vecmem/memory/choice_memory_resource.hpp"

#include <cassert>
#include <cstddef>
#include <vector>

#include "vecmem/memory/memory_resource.hpp"

namespace vecmem {
choice_memory_resource::choice_memory_resource(
    std::function<memory_resource &(std::size_t, std::size_t)> decision)
    : m_decision(decision) {}

void *choice_memory_resource::do_allocate(std::size_t size, std::size_t align) {
    /*
     * We cannot blindly allocate, because we need to keep track of which
     * upstream allocator allocated this memory. Thus, we must also store
     * the allocation result in a map.
     */
    memory_resource &res = m_decision(size, align);

    void *ptr = res.allocate(size, align);

    m_allocations.emplace(ptr, res);

    return ptr;
}

void choice_memory_resource::do_deallocate(void *ptr, std::size_t size,
                                           std::size_t align) {
    /*
     * Extract the record of which upstream resource was used to allocate the
     * given pointer.
     */
    auto nh = m_allocations.extract(ptr);

    /*
     * For debug builds, throw an assertion error if we do not know this
     * allocation.
     */
    assert(nh);

    /*
     * Retrieve the correct resource and deallocate the memory.
     */
    memory_resource &res = nh.mapped();

    res.deallocate(nh.key(), size, align);
}
}  // namespace vecmem
