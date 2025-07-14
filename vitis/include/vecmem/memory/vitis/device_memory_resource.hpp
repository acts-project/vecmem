/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/vecmem_vitis_export.hpp"

#include "xrt/xrt_bo.h"

#include <typeinfo>
// TODO: Find a better way to handle the opencl version
#define CL_TARGET_OPENCL_VERSION 120
#include "vecmem/utils/xcl2.hpp"

/// @brief Namespace holding types that work on/with CUDA
namespace vecmem::vitis {

/**
 * @brief Memory resource that wraps direct allocations on a CUDA device.
 *
 * This is an allocator-type memory resource (that is to say, it only
 * allocates, it does not try to manage memory in a smart way) that works
 * for CUDA device memory. Each instance is bound to a specific device.
 */
class device_memory_resource final : public memory_resource {

public:
    VECMEM_VITIS_EXPORT device_memory_resource(xrt::bo bo);
//    /// Destructor
    VECMEM_VITIS_EXPORT
    ~device_memory_resource();

private:
    /// @name Function(s) implementing @c vecmem::memory_resource
    /// @{

    /// Allocate memory on the selected device
    VECMEM_VITIS_EXPORT
    virtual void* do_allocate(std::size_t, std::size_t) override final;
    // De-allocate a previously allocated memory block on the selected device
    VECMEM_VITIS_EXPORT
    virtual void do_deallocate(void* p, std::size_t,
                               std::size_t) override final;
    /// Compares @c *this for equality with @c other
    VECMEM_VITIS_EXPORT
    virtual bool do_is_equal(
        const memory_resource& other) const noexcept override final;

    /// @}
//
    const xrt::bo buffer_object;
    std::size_t curr_ptr = 0;

};  // class device_memory_resource

}  // namespace vecmem::cuda
