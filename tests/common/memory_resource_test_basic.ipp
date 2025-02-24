/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// System include(s).
#include <stdexcept>

/// Perform some very basic tests that do not need host accessibility
TEST_P(memory_resource_test_basic, allocations) {

    vecmem::memory_resource* resource = GetParam();
    EXPECT_THROW(static_cast<void>(resource->allocate(0)), std::bad_alloc);
    for (std::size_t size = 1; size < 100000; size += 1000) {
        void* ptr = resource->allocate(size);
        EXPECT_NE(ptr, nullptr);
        resource->deallocate(ptr, size);
    }
}
