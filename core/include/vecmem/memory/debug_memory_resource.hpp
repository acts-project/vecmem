/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cstddef>
#include <unordered_map>

#include "vecmem/memory/details/memory_resource_base.hpp"
#include "vecmem/memory/memory_resource.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4251)
#endif

namespace vecmem {
/**
 * @brief This memory resource forwards allocation and deallocation requests to
 * the upstream resource, but alerts the user of potential problems.
 *
 * For example, this memory resource can be used to catch overlapping
 * allocations, double frees, invalid frees, and other memory integrity issues.
 */
class VECMEM_CORE_EXPORT debug_memory_resource final
    : public details::memory_resource_base {
public:
    /**
     * @brief Constructs the debug memory resource.
     *
     * @param[in] upstream The upstream memory resource to use.
     */
    debug_memory_resource(memory_resource& upstream);

private:
    virtual void* do_allocate(std::size_t, std::size_t) override;

    virtual void do_deallocate(void* p, std::size_t, std::size_t) override;

    memory_resource& m_upstream;

    std::unordered_map<void*, std::pair<std::size_t, std::size_t>>
        m_allocations;
};
}  // namespace vecmem

#ifdef _MSC_VER
#pragma warning(pop)
#endif
