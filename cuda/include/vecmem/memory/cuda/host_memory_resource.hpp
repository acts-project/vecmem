/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/vecmem_cuda_export.hpp"

namespace vecmem::cuda {

/**
 * @brief Memory resource that wraps page-locked CUDA host allocation.
 *
 * This is an allocator-type memory resource that allocates CUDA host
 * memory, which is page-locked by default to allow faster transfer to the
 * CUDA devices.
 */
class VECMEM_CUDA_EXPORT host_memory_resource : public memory_resource {
private:
    virtual void* do_allocate(std::size_t, std::size_t) override;

    virtual void do_deallocate(void* p, std::size_t, std::size_t) override;

    virtual bool do_is_equal(const memory_resource&) const noexcept override;

};  // class host_memory_resource

}  // namespace vecmem::cuda
