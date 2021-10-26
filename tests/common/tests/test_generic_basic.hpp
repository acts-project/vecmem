/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

#include <gtest/gtest.h>

#include "vecmem/memory/memory_resource.hpp"

/// Base test case for the CUDA memory resources
///
/// This just makes sure that the memory resources defined in the
/// @c vecmem::cuda library are more-or-less functional.
///
class memory_resource_test_basic
    : public testing::TestWithParam<vecmem::memory_resource*> {};

/// Perform some very basic tests that do not need host accessibility
TEST_P(memory_resource_test_basic, allocations) {

    vecmem::memory_resource* resource = GetParam();
    for (std::size_t size = 1000; size < 100000; size += 1000) {
        void* ptr = resource->allocate(size);
        resource->deallocate(ptr, size);
    }
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(memory_resource_test_basic);
