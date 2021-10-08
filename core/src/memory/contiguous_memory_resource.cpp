/**
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/contiguous_memory_resource.hpp"

#include "vecmem/utils/debug.hpp"

// System include(s).
#include <stdexcept>

namespace vecmem {

contiguous_memory_resource::contiguous_memory_resource(
    memory_resource &upstream, std::size_t size)
    : m_upstream(upstream),
      m_size(size),
      m_begin(m_upstream.allocate(m_size)),
      m_next(m_begin) {

    VECMEM_DEBUG_MSG(
        2, "Allocated %lu bytes at %p from the upstream memory resource",
        m_size, m_begin);
}

contiguous_memory_resource::~contiguous_memory_resource() {
    /*
     * Deallocate our memory arena upstream.
     */
    m_upstream.deallocate(m_begin, m_size);
    VECMEM_DEBUG_MSG(
        2, "De-allocated %lu bytes at %p using the upstream memory resource",
        m_size, m_begin);
}

void *contiguous_memory_resource::do_allocate(std::size_t size, std::size_t) {
    /*
     * Calculate where the end of this current allocation would be.
     */
    void *next = static_cast<void *>(static_cast<char *>(m_next) + size);

    /*
     * The start of this allocation, save it in a temporary variable as we
     * will override it later before returning.
     */
    void *curr = m_next;

    /*
     * If the end of this allocation is past the end of our memory space,
     * we can't allocate, and should throw an error.
     */
    if (next >= (static_cast<char *>(m_begin) + m_size)) {
        throw std::bad_alloc();
    }

    /*
     * Update the start of the next allocation and return.
     */
    m_next = next;

    VECMEM_DEBUG_MSG(4, "Allocated %lu bytes at %p", size, curr);
    return curr;
}

void contiguous_memory_resource::do_deallocate(void *, std::size_t,
                                               std::size_t) {
    /*
     * Deallocation is a no-op for this memory resource, so we do nothing.
     */
    return;
}

}  // namespace vecmem
