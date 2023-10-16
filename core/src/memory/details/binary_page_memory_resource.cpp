/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/details/binary_page_memory_resource.hpp"

#include "binary_page_memory_resource_impl.hpp"
#include "vecmem/utils/debug.hpp"

namespace vecmem::details {

binary_page_memory_resource::binary_page_memory_resource(
    memory_resource& upstream)
    : m_impl(new binary_page_memory_resource_impl{upstream}) {}

binary_page_memory_resource::binary_page_memory_resource(
    binary_page_memory_resource&& parent)
    : m_impl(parent.m_impl) {

    parent.m_impl = nullptr;
}

binary_page_memory_resource::~binary_page_memory_resource() {

    delete m_impl;
}

binary_page_memory_resource& binary_page_memory_resource::operator=(
    binary_page_memory_resource&& rhs) {

    delete m_impl;
    m_impl = rhs.m_impl;
    rhs.m_impl = nullptr;
    return *this;
}

void* binary_page_memory_resource::mr_allocate(std::size_t size,
                                               std::size_t align) {

    if (size == 0) {
        return nullptr;
    }

    void* ptr = m_impl->do_allocate(size, align);
    VECMEM_DEBUG_MSG(2, "Allocated %lu bytes at %p", size, ptr);
    return ptr;
}

void binary_page_memory_resource::mr_deallocate(void* p, std::size_t size,
                                                std::size_t align) {

    if (p == nullptr) {
        return;
    }

    VECMEM_DEBUG_MSG(2, "De-allocating memory at %p", p);
    m_impl->do_deallocate(p, size, align);
}

}  // namespace vecmem::details
