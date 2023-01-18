/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cstddef>

#include "vecmem/memory/details/memory_resource_base.hpp"
#include "vecmem/memory/memory_resource.hpp"

namespace vecmem {
/**
 * @brief This memory resource forwards allocation and deallocation requests to
 * the upstream resource.
 *
 * This allocator is here to act as the unit in the monoid of memory resources.
 * It serves only niche practical purpose.
 */
class VECMEM_CORE_EXPORT identity_memory_resource final
    : public details::memory_resource_base {
public:
    /**
     * @brief Constructs the identity memory resource.
     *
     * @param[in] upstream The upstream memory resource to use.
     */
    identity_memory_resource(memory_resource& upstream);

private:
    virtual void* do_allocate(std::size_t, std::size_t) override;

    virtual void do_deallocate(void* p, std::size_t, std::size_t) override;

    virtual bool do_is_equal(const memory_resource&) const noexcept override;

    memory_resource& m_upstream;
};
}  // namespace vecmem
