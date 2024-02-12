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
#include "vecmem/utils/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <tuple>

/// Test case for @c vecmem::copy specializations
///
/// It tests the copy of different data types, using the provided copy objects
/// and memory resources.
///
class copy_tests
    : public testing::TestWithParam<std::tuple<
          vecmem::copy&, vecmem::memory_resource&, vecmem::memory_resource&>> {
};

// Include the implementation.
#include "copy_tests.ipp"
