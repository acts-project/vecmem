/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/details/memory_resource_base.hpp"
#include "vecmem/vecmem_hip_export.hpp"

namespace vecmem::hip {

/// Memory resource for a specific HIP device
class VECMEM_HIP_EXPORT device_memory_resource final
    : public vecmem::details::memory_resource_base {

public:
    /// Invalid/default device identifier
    static constexpr int INVALID_DEVICE = -1;

    /// Constructor, allowing the specification of the device to use
    device_memory_resource(int device = INVALID_DEVICE);

private:
    /// @name Function(s) implemented from @c vecmem::memory_resource
    /// @{

    /// Function performing the memory allocation
    void* do_allocate(std::size_t nbytes, std::size_t alignment) override final;

    /// Function performing the memory de-allocation
    void do_deallocate(void* ptr, std::size_t nbytes,
                       std::size_t alignment) override final;

    /// Function comparing two memory resource instances
    bool do_is_equal(
        const memory_resource& other) const noexcept override final;

    /// @}

    /// The HIP device used by this resource
    const int m_device;

};  // class device_memory_resource

}  // namespace vecmem::hip
