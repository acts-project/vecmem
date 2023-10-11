/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/details/memory_resource_base.hpp"
#include "vecmem/vecmem_hip_export.hpp"

/// @brief Namespace holding implementation details for the ROCm/HIP code
namespace vecmem::hip::details {

/// Memory resource for a specific HIP device
class VECMEM_HIP_EXPORT device_memory_resource
    : public vecmem::details::memory_resource_base {

public:
    /// Invalid/default device identifier
    static constexpr int INVALID_DEVICE = -1;

    /// Constructor, allowing the specification of the device to use
    device_memory_resource(int device = INVALID_DEVICE);

protected:
    /// @name Function(s) implementing @c vecmem::details::memory_resource_base
    /// @{

    /// Function performing the memory allocation
    void* mr_allocate(std::size_t nbytes, std::size_t alignment) override final;

    /// Function performing the memory de-allocation
    void mr_deallocate(void* ptr, std::size_t nbytes,
                       std::size_t alignment) override final;

    /// Function comparing two memory resource instances
    bool mr_is_equal(
        const memory_resource& other) const noexcept override final;

    /// @}

private:
    /// The HIP device used by this resource
    const int m_device;

};  // class device_memory_resource

}  // namespace vecmem::hip::details
