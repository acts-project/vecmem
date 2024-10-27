/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/memory/metal/shared_memory_resource.hpp"
#include "../utils/get_device.h"

// System import(s).
#import <Metal/Metal.h>

// System include(s).
#include <cassert>
#include <map>
#include <stdexcept>

namespace vecmem::metal {
namespace details {

/// Internal state for the shared memory resource.
struct shared_memory_resource_data {
    /// The Metal device.
    device_wrapper m_device;
    /// Memory buffers allocated on the device.
    std::map<void*, id<MTLBuffer> > m_buffers;
};

}  // namespace details

shared_memory_resource::shared_memory_resource(const device_wrapper& device)
    : m_data(std::make_unique<details::shared_memory_resource_data>()) {

    // Save the device wrapper object.
    m_data->m_device = device;
}

shared_memory_resource::~shared_memory_resource() {}

void* shared_memory_resource::do_allocate(std::size_t bytes, std::size_t) {

    // Create a new shared buffer.
    id<MTLBuffer> buffer = [details::get_device(m_data->m_device)
        newBufferWithLength:bytes
                    options:MTLResourceStorageModeShared];
    // If the buffer was created successfully, store it in the map, and return
    // its pointer.
    if (buffer != nil) {
        void* ptr = buffer.contents;
        assert(m_data);
        m_data->m_buffers[ptr] = buffer;
        return ptr;
    }

    // If the buffer could not be created, throw an exception.
    throw std::bad_alloc();
}

void shared_memory_resource::do_deallocate(void* p, std::size_t, std::size_t) {

    // Find the buffer, and release it. It's undefined behaviour to deallocate
    // a buffer that was not allocated by this memory resource. So we might as
    // well crash when that happens.
    assert(m_data);
    auto it = m_data->m_buffers.find(p);
    assert(it != m_data->m_buffers.end());
    m_data->m_buffers.erase(it);
}

bool shared_memory_resource::do_is_equal(
    const memory_resource& other) const noexcept {

    // The memory resource manages buffers itself. It is only equal to itself.
    return this == &other;
}

}  // namespace vecmem::metal
