/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/utils/metal/device_wrapper.hpp"
#include "vecmem/vecmem_metal_export.hpp"

// System include(s).
#include <memory>

/// @brief Namespace holding types that work with Metal
namespace vecmem::metal {

// Forward declaration(s).
namespace details {
struct shared_memory_resource_data;
}

/**
 * @brief Memory resource that exposes Metal shared memory allocations.
 *
 * This is an allocator-type memory resource (that is to say, it only
 * allocates, it does not try to manage memory in a smart way) that works
 * for Metal shared memory. Each instance is bound to a specific device.
 */
class shared_memory_resource final : public memory_resource {

public:
    /// Default constructor
    VECMEM_METAL_EXPORT
    shared_memory_resource(const device_wrapper& device = {});
    /// Destructor
    VECMEM_METAL_EXPORT
    ~shared_memory_resource();

private:
    /// @name Function(s) implementing @c vecmem::memory_resource
    /// @{

    /// Allocate memory on the selected device
    VECMEM_METAL_EXPORT
    virtual void* do_allocate(std::size_t, std::size_t) override final;
    /// De-allocate a previously allocated memory block on the selected device
    VECMEM_METAL_EXPORT
    virtual void do_deallocate(void* p, std::size_t,
                               std::size_t) override final;
    /// Compares @c *this for equality with @c other
    VECMEM_METAL_EXPORT
    virtual bool do_is_equal(
        const memory_resource& other) const noexcept override final;

    /// @}

    /// Internal state of the memory resource
    std::unique_ptr<details::shared_memory_resource_data> m_data;

};  // class shared_memory_resource

}  // namespace vecmem::metal
