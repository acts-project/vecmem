/**
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/binary_page_memory_resource.hpp"

#include "binary_page_memory_resource_impl.hpp"

namespace vecmem {

binary_page_memory_resource::binary_page_memory_resource(
    memory_resource &upstream)
    : m_impl(std::make_unique<details::binary_page_memory_resource_impl>(
          upstream)) {}

binary_page_memory_resource::~binary_page_memory_resource() {}

void *binary_page_memory_resource::do_allocate(std::size_t size,
                                               std::size_t align) {

    return m_impl->do_allocate(size, align);
}

void binary_page_memory_resource::do_deallocate(void *p, std::size_t size,
                                                std::size_t align) {

    m_impl->do_deallocate(p, size, align);
}

bool binary_page_memory_resource::do_is_equal(
    const memory_resource &other) const noexcept {
    /*
     * These memory resources are only equal if they are actually the same
     * object.
     */
    return this == &other;
}

}  // namespace vecmem
