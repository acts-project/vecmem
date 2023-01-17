/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <cstddef>
#include <functional>
#include <unordered_map>
#include <vector>

#include "vecmem/memory/details/memory_resource_base.hpp"
#include "vecmem/memory/memory_resource.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4251)
#endif

namespace vecmem {
/**
 * @brief This memory resource tries to allocate with several upstream resources
 * and returns the first succesful one.
 */
class VECMEM_CORE_EXPORT coalescing_memory_resource final
    : public details::memory_resource_base {
public:
    /**
     * @brief Constructs the coalescing memory resource.
     *
     * @note The memory resources passed to this constructed are given in order
     * of decreasing priority. That is to say, the first one is tried first,
     * then the second, etc.
     *
     * @param[in] upstreams The upstream memory resources to use.
     */
    coalescing_memory_resource(
        std::vector<std::reference_wrapper<memory_resource>>&& upstreams);

private:
    virtual void* do_allocate(std::size_t, std::size_t) override;

    virtual void do_deallocate(void* p, std::size_t, std::size_t) override;

    const std::vector<std::reference_wrapper<memory_resource>> m_upstreams;

    std::unordered_map<void*, std::reference_wrapper<memory_resource>>
        m_allocations;
};
}  // namespace vecmem

#ifdef _MSC_VER
#pragma warning(pop)
#endif
