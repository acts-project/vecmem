/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/vitis/device_memory_resource.hpp"


#include "vecmem/utils/debug.hpp"

#include <iostream>

namespace vecmem::vitis {

device_memory_resource::device_memory_resource(int s) : state(s) {
    std::cout << "Constructing device memory resource for device " << s;
}

device_memory_resource::~device_memory_resource() = default;

// returns the pointer to the allocated memory
void *device_memory_resource::do_allocate(std::size_t bytes, std::size_t) {

    if (bytes == 0) {
        return nullptr;
    }

//    // Make sure that we would use the appropriate device.
//    details::select_device dev(m_device);
//
//    // Allocate the memory.
//    void *res = nullptr;
//    VECMEM_CUDA_ERROR_CHECK(cudaMalloc(&res, bytes));
    VECMEM_DEBUG_MSG(2, "Vitis: allocated %ld bytes at %p", bytes, res);
    return nullptr; // Placeholder for actual allocation logic
//    return res;
}

void device_memory_resource::do_deallocate(void *p, std::size_t, std::size_t) {

    if (p == nullptr) {
        return;
    }

    // Make sure that we would use the appropriate device.
//    details::select_device dev(m_device);

    // Free the memory.
    VECMEM_DEBUG_MSG(2, "De-allocating memory at %p on device %i", p, m_device);
}

bool device_memory_resource::do_is_equal(
    const memory_resource &other) const noexcept {
    const device_memory_resource *c;
    c = dynamic_cast<const device_memory_resource *>(&other);

    /*
     * The equality check here is ever so slightly more difficult. Not only
     * does the other object need to be a device memory resource, it must
     * also target the same device.
     */
    return c != nullptr ;
}
} // namespace vecmem::vitis

