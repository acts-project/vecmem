/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/details/choice_memory_resource.hpp"

#include "choice_memory_resource_impl.hpp"
#include "vecmem/utils/debug.hpp"

namespace vecmem::details {

choice_memory_resource::choice_memory_resource(
    std::function<memory_resource&(std::size_t, std::size_t)> decision)
    : m_impl(new choice_memory_resource_impl{decision}) {}

choice_memory_resource::choice_memory_resource(choice_memory_resource&& parent)
    : m_impl(parent.m_impl) {

    parent.m_impl = nullptr;
}

choice_memory_resource::~choice_memory_resource() {

    delete m_impl;
}

choice_memory_resource& choice_memory_resource::operator=(
    choice_memory_resource&& rhs) {

    delete m_impl;
    m_impl = rhs.m_impl;
    rhs.m_impl = nullptr;
    return *this;
}

void* choice_memory_resource::mr_allocate(std::size_t size, std::size_t align) {

    if (size == 0) {
        return nullptr;
    }

    void* ptr = m_impl->allocate(size, align);
    VECMEM_DEBUG_MSG(3, "Allocated %lu bytes at %p", size, ptr);
    return ptr;
}

void choice_memory_resource::mr_deallocate(void* p, std::size_t size,
                                           std::size_t align) {

    if (p == nullptr) {
        return;
    }

    VECMEM_DEBUG_MSG(3, "De-allocating memory at %p", p);
    m_impl->deallocate(p, size, align);
}

}  // namespace vecmem::details
