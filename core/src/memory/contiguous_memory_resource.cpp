/**
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "vecmem/memory/contiguous_memory_resource.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>

#include "vecmem/memory/memory_resource.hpp"

namespace vecmem {
contiguous_memory_resource::contiguous_memory_resource(
    memory_resource &upstream, std::size_t size)
    : m_upstream(upstream),
      m_size(size),
      m_begin(m_upstream.allocate(m_size)),
      m_next(m_begin) {}

contiguous_memory_resource::~contiguous_memory_resource() {
    /*
     * Deallocate our memory arena upstream.
     */
    m_upstream.deallocate(m_begin, m_size);
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

    return curr;
}

void contiguous_memory_resource::do_deallocate(void *, std::size_t,
                                               std::size_t) {
    /*
     * Deallocation is a no-op for this memory resource, so we do nothing.
     */
    return;
}

bool contiguous_memory_resource::do_is_equal(
    const memory_resource &other) const noexcept {
    /*
     * These memory resources are equal if and only if they are the same
     * object.
     */
    return this == &other;
}

}  // namespace vecmem
