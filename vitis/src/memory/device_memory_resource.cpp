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
#include <memory>

namespace vecmem::vitis {

device_memory_resource::device_memory_resource(uint8_t* b, std::size_t s): buffer(b), size(s) {
    std::cout << "Constructing device memory resource for device" << std::endl;
}

device_memory_resource::~device_memory_resource() = default;

// returns the pointer to the allocated memory
void *device_memory_resource::do_allocate(std::size_t bytes, std::size_t) {
    if (bytes == 0) {
        return nullptr;
    }
    // Allocate the memory.
    if (curr_ptr + bytes > size) {
        VECMEM_DEBUG_MSG(1, "WARNING: Not enough memory in the buffer object");
        return nullptr;
    }

    void* old_ptr = reinterpret_cast<void*>(curr_ptr);

    curr_ptr += bytes;
    return old_ptr;
}

void device_memory_resource::do_deallocate(void *p, std::size_t, std::size_t) {

    if (p == nullptr) {
        return;
    }

    VECMEM_DEBUG_MSG(1, "WARNING: Memory deallocation is not supported");
//    VECMEM_DEBUG_MSG(2, "De-allocating memory at %p on device %i", p, m_device);
}

bool device_memory_resource::do_is_equal(
    const memory_resource &other) const noexcept {
    const device_memory_resource *c;
    c = dynamic_cast<const device_memory_resource *>(&other);

    return c != nullptr && c->buffer == buffer;
}
} // namespace vecmem::vitis

