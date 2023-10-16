/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/details/arena_memory_resource.hpp"

#include "arena_memory_resource_impl.hpp"
#include "vecmem/utils/debug.hpp"

namespace vecmem::details {

arena_memory_resource::arena_memory_resource(memory_resource& upstream,
                                             std::size_t initial_size,
                                             std::size_t maximum_size)
    : m_impl(new arena_memory_resource_impl{initial_size, maximum_size,
                                            upstream}) {}

arena_memory_resource::arena_memory_resource(arena_memory_resource&& parent)
    : m_impl(parent.m_impl) {

    parent.m_impl = nullptr;
}

arena_memory_resource::~arena_memory_resource() {

    delete m_impl;
}

arena_memory_resource& arena_memory_resource::operator=(
    arena_memory_resource&& rhs) {

    delete m_impl;
    m_impl = rhs.m_impl;
    rhs.m_impl = nullptr;
    return *this;
}

void* arena_memory_resource::mr_allocate(std::size_t bytes, std::size_t) {

    if (bytes == 0) {
        return nullptr;
    }

    void* ptr = m_impl->allocate(bytes);
    VECMEM_DEBUG_MSG(2, "Allocated %lu bytes at %p", bytes, ptr);
    return ptr;
}

void arena_memory_resource::mr_deallocate(void* p, std::size_t bytes,
                                          std::size_t) {

    if (p == nullptr) {
        return;
    }

    VECMEM_DEBUG_MSG(2, "De-allocating memory at %p", p);
    m_impl->deallocate(p, bytes);
}

}  // namespace vecmem::details
