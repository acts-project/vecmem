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

#include <typeinfo>

/// @brief Namespace holding types that work on/with Vitis
namespace vecmem::vitis {

/**
 * @brief Memory resource that wraps memory allocation to a local buffer
 *
 * This is an allocator-type memory resource (that is to say, it only
 * allocates, it does not try to manage memory in a smart way) that 
 * wraps a given buffer and allocates memory in the buffer.
 * Deallocation is not supported.
 */
class device_memory_resource final : public memory_resource {

public:
    VECMEM_VITIS_EXPORT device_memory_resource(uint8_t* buffer, std::size_t size);
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
    std::size_t curr_ptr = 2;
    const uint8_t* buffer;
    const std::size_t size ;

};  // class device_memory_resource

}  // namespace vecmem::vitis
