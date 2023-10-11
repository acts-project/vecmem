/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/details/instrumenting_memory_resource.hpp"

#include "instrumenting_memory_resource_impl.hpp"
#include "vecmem/utils/debug.hpp"

namespace vecmem::details {

instrumenting_memory_resource::memory_event::memory_event(type t, std::size_t s,
                                                          std::size_t a,
                                                          void* p,
                                                          std::size_t ns)
    : m_type(t), m_size(s), m_align(a), m_ptr(p), m_time(ns) {}

instrumenting_memory_resource::instrumenting_memory_resource(
    memory_resource& upstream)
    : m_impl(new instrumenting_memory_resource_impl{upstream}) {}

instrumenting_memory_resource::instrumenting_memory_resource(
    instrumenting_memory_resource&& parent)
    : m_impl(parent.m_impl) {

    parent.m_impl = nullptr;
}

instrumenting_memory_resource::~instrumenting_memory_resource() {

    delete m_impl;
}

instrumenting_memory_resource& instrumenting_memory_resource::operator=(
    instrumenting_memory_resource&& rhs) {

    delete m_impl;
    m_impl = rhs.m_impl;
    rhs.m_impl = nullptr;
    return *this;
}

const std::vector<instrumenting_memory_resource::memory_event>&
instrumenting_memory_resource::get_events(void) const {

    return m_impl->get_events();
}

void instrumenting_memory_resource::add_pre_allocate_hook(
    std::function<void(std::size_t, std::size_t)> f) {

    m_impl->add_pre_allocate_hook(f);
}

void instrumenting_memory_resource::add_post_allocate_hook(
    std::function<void(std::size_t, std::size_t, void*)> f) {

    m_impl->add_post_allocate_hook(f);
}

void instrumenting_memory_resource::add_pre_deallocate_hook(
    std::function<void(void*, std::size_t, std::size_t)> f) {

    m_impl->add_pre_deallocate_hook(f);
}

void* instrumenting_memory_resource::mr_allocate(std::size_t size,
                                                 std::size_t align) {

    if (size == 0) {
        return nullptr;
    }

    void* ptr = m_impl->allocate(size, align);
    VECMEM_DEBUG_MSG(3, "Allocated %lu bytes at %p", size, ptr);
    return ptr;
}

void instrumenting_memory_resource::mr_deallocate(void* p, std::size_t size,
                                                  std::size_t align) {

    if (p == nullptr) {
        return;
    }

    VECMEM_DEBUG_MSG(3, "De-allocating memory at %p", p);
    m_impl->deallocate(p, size, align);
}

}  // namespace vecmem::details
