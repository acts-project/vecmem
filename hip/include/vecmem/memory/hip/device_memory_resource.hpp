/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/vecmem_hip_export.hpp"

/// @brief Namespace holding types that work on/with ROCm/HIP
namespace vecmem::hip {

/// Memory resource for a specific HIP device
class device_memory_resource final : public memory_resource {

public:
    /// Invalid/default device identifier
    static constexpr int INVALID_DEVICE = -1;

    /// Constructor, allowing the specification of the device to use
    VECMEM_HIP_EXPORT
    device_memory_resource(int device = INVALID_DEVICE);
    /// Destructor
    VECMEM_HIP_EXPORT
    ~device_memory_resource() noexcept override;

private:
    /// @name Function(s) implementing @c vecmem::memory_resource
    /// @{

    /// Function performing the memory allocation
    VECMEM_HIP_EXPORT
    void* do_allocate(std::size_t nbytes, std::size_t alignment) override;

    /// Function performing the memory de-allocation
    VECMEM_HIP_EXPORT
    void do_deallocate(void* ptr, std::size_t nbytes,
                       std::size_t alignment) override;

    /// Function comparing two memory resource instances
    VECMEM_HIP_EXPORT
    bool do_is_equal(const memory_resource& other) const noexcept override;

    /// @}

    /// The HIP device used by this resource
    const int m_device;

};  // class device_memory_resource

}  // namespace vecmem::hip
