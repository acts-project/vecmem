/**
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/vecmem_core_export.hpp"

namespace vecmem {

/**
 * @brief Memory resource which wraps the malloc standard library call.
 *
 * This is probably the simplest memory resource you can possibly write. It
 * is a terminal resource which does nothing but wrap malloc and free. It
 * is state-free (on the relevant levels of abstraction).
 */
class VECMEM_CORE_EXPORT host_memory_resource : public vecmem::memory_resource {

private:
    virtual void* do_allocate(std::size_t, std::size_t) override;

    virtual void do_deallocate(void* p, std::size_t, std::size_t) override;

    virtual bool do_is_equal(const memory_resource&) const noexcept override;

};  // class host_memory_resource

}  // namespace vecmem
