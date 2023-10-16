/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/details/memory_resource_base.hpp"
#include "vecmem/vecmem_cuda_export.hpp"

/// @brief Namespace holding implementation details for the CUDA code
namespace vecmem::cuda::details {

/**
 * @brief Memory resource that wraps direct allocations on a CUDA device.
 *
 * This is an allocator-type memory resource (that is to say, it only
 * allocates, it does not try to manage memory in a smart way) that works
 * for CUDA device memory. Each instance is bound to a specific device.
 */
class VECMEM_CUDA_EXPORT device_memory_resource
    : public vecmem::details::memory_resource_base {

public:
    /// Invalid/default device identifier
    static constexpr int INVALID_DEVICE = -1;

    /**
     * @brief Construct a CUDA device resource for a specific device.
     *
     * This constructor takes a device identifier argument which works in
     * the same way as in standard CUDA code. If the device number is
     * positive, that device is selected. If the device number is negative,
     * the currently selected device is used.
     *
     * @note The default device is resolved at resource construction time,
     * not at allocation time.
     *
     * @param[in] device CUDA identifier for the device to use
     */
    device_memory_resource(int device = INVALID_DEVICE);

protected:
    /// @name Function(s) implementing @c vecmem::details::memory_resource_base
    /// @{

    /// Allocate memory on the selected device
    virtual void* mr_allocate(std::size_t, std::size_t) override;
    /// De-allocate a previously allocated memory block on the selected device
    virtual void mr_deallocate(void* p, std::size_t, std::size_t) override;
    /// Compares @c *this for equality with @c other
    virtual bool mr_is_equal(
        const memory_resource& other) const noexcept override;

    /// @}

private:
    /// CUDA device identifier to use for the (de-)allocations
    const int m_device;

};  // class device_memory_resource

}  // namespace vecmem::cuda::details
