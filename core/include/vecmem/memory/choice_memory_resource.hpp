/**
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

#include "vecmem/memory/details/memory_resource_base.hpp"
#include "vecmem/memory/memory_resource.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4251)
#endif

namespace vecmem {
/**
 * @brief This memory resource conditionally allocates memory. It is
 * constructed with a function that determines which upstream resource to use.
 *
 * This resource can be used to construct complex conditional allocation
 * schemes.
 */
class VECMEM_CORE_EXPORT choice_memory_resource final
    : public details::memory_resource_base {
public:
    /**
     * @brief Constructs the choice memory resource.
     *
     * @param[in] upstreams The upstream memory resources to use.
     * @param[in] decision The function which picks the upstream memory
     * resource to use by index.
     */
    choice_memory_resource(
        std::function<memory_resource&(std::size_t, std::size_t)> decision);

private:
    virtual void* do_allocate(std::size_t, std::size_t) override;

    virtual void do_deallocate(void* p, std::size_t, std::size_t) override;

    std::unordered_map<void*, std::reference_wrapper<memory_resource>>
        m_allocations;

    std::function<memory_resource&(std::size_t, std::size_t)> m_decision;
};
}  // namespace vecmem

#ifdef _MSC_VER
#pragma warning(pop)
#endif
