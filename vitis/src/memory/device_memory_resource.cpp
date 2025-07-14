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

device_memory_resource::device_memory_resource(
        cl_context c, 
        cl_device_id d, 
        cl_kernel k,
        cl_mem_flags f) : context(c), device_id(d), kernel(k), flags(f) {
    std::cout << "Constructing device memory resource for device" << std::endl;
}

device_memory_resource::~device_memory_resource() = default;

// returns the pointer to the allocated memory
void *device_memory_resource::do_allocate(std::size_t bytes, std::size_t) {
    if (bytes == 0) {
        return nullptr;
    }
    cl_int err;
    // allocate memory on host device
    printf("Vitis: allocating %ld bytes", bytes);
//    OCL_CHECK(err, cl_mem buffer = clCreateBuffer(context, flags,
//                                      bytes, nullptr, &err));
//    OCL_CHECK(err, err = clSetKernelArg(kernel, argc++, sizeof(cl_mem), &buffer));

    printf("Vitis: allocated %ld bytes at %p", bytes);
    return buffer;
}

void device_memory_resource::do_deallocate(void *p, std::size_t, std::size_t) {

    if (p == nullptr) {
        return;
    }

    VECMEM_DEBUG_MSG(1, "WARNING: Memory deallocation is not implemented yet");
//    VECMEM_DEBUG_MSG(2, "De-allocating memory at %p on device %i", p, m_device);
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
    return c != nullptr && c->context == context &&
           c->device_id == device_id;
}
} // namespace vecmem::vitis

