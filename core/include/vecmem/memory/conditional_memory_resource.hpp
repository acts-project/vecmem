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

#include "vecmem/memory/details/memory_resource_base.hpp"
#include "vecmem/memory/memory_resource.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4251)
#endif

namespace vecmem {
/**
 * @brief This memory resource conditionally allocates memory. It is
 * constructed with a predicate function that determines whether an allocation
 * should succeed or not.
 *
 * This resource can be used to construct complex conditional allocation
 * schemes.
 */
class VECMEM_CORE_EXPORT conditional_memory_resource final
    : public details::memory_resource_base {
public:
    /**
     * @brief Constructs the conditional memory resource.
     *
     * @param[in] upstream The upstream memory resource to use.
     * @param[in] pred The predicate function that determines whether the
     * allocation should succeed.
     */
    conditional_memory_resource(
        memory_resource& upstream,
        std::function<bool(std::size_t, std::size_t)> pred);

private:
    virtual void* do_allocate(std::size_t, std::size_t) override;

    virtual void do_deallocate(void* p, std::size_t, std::size_t) override;

    memory_resource& m_upstream;

    std::function<bool(std::size_t, std::size_t)> m_pred;
};
}  // namespace vecmem

#ifdef _MSC_VER
#pragma warning(pop)
#endif
