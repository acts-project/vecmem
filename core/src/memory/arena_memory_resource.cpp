/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/arena_memory_resource.hpp"

#include "alignment.hpp"
#include "arena.hpp"
#include "vecmem/utils/debug.hpp"

namespace vecmem {

arena_memory_resource::arena_memory_resource(memory_resource& upstream,
                                             std::size_t initial_size,
                                             std::size_t maximum_size)
    : m_arena(std::make_unique<details::arena>(initial_size, maximum_size,
                                               upstream)) {}

arena_memory_resource::~arena_memory_resource() {}

void* arena_memory_resource::do_allocate(std::size_t bytes, std::size_t) {

    void* ptr = m_arena->allocate(alignment::align_up(bytes, 8));
    VECMEM_DEBUG_MSG(4, "Allocated %lu bytes at %p", bytes, ptr);
    return ptr;
}

void arena_memory_resource::do_deallocate(void* p, std::size_t bytes,
                                          std::size_t) {

    VECMEM_DEBUG_MSG(4, "De-allocating memory at %p", p);
    m_arena->deallocate(p, alignment::align_up(bytes, 8));
}

bool arena_memory_resource::do_is_equal(
    const memory_resource& other) const noexcept {
    const arena_memory_resource* c;
    c = dynamic_cast<const arena_memory_resource*>(&other);

    return c != nullptr;
}

}  // namespace vecmem
