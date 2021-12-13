/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cassert>

#include "vecmem/memory/unique_ptr.hpp"

namespace vecmem::details {
unique_alloc_deleter_impl::unique_alloc_deleter_impl(void) = default;

unique_alloc_deleter_impl::unique_alloc_deleter_impl(memory_resource& mr,
                                                     std::size_t s,
                                                     std::size_t a)
    : m_mr(&mr), m_size(s), m_align(a) {}

unique_alloc_deleter_impl::unique_alloc_deleter_impl(
    const unique_alloc_deleter_impl& i) = default;

unique_alloc_deleter_impl::unique_alloc_deleter_impl(
    unique_alloc_deleter_impl&& i) = default;

unique_alloc_deleter_impl& unique_alloc_deleter_impl::operator=(
    const unique_alloc_deleter_impl& i) = default;

unique_alloc_deleter_impl& unique_alloc_deleter_impl::operator=(
    unique_alloc_deleter_impl&& i) = default;

void unique_alloc_deleter_impl::operator()(void* p) const {
    assert(m_mr != nullptr);

    /*
     * As before, if this happens... Something has gone VERY wrong.
     */
    if (m_mr == nullptr) {
        return;
    }

    /*
     * Deallocate the memory that we were using.
     */
    if (m_align > 0) {
        m_mr->deallocate(p, m_size, m_align);
    } else {
        m_mr->deallocate(p, m_size);
    }
}
}  // namespace vecmem::details
