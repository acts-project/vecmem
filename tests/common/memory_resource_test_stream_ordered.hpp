/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2026 CERN for the benefit of the ACTS project
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

/// Test case for stream-ordered memory resources
///
/// It uses the same configurations as @c copy_tests.
///
class memory_resource_test_stream_ordered
    : public testing::TestWithParam<
          std::tuple<vecmem::copy*&, vecmem::copy*&, vecmem::memory_resource*&,
                     vecmem::memory_resource*&>> {};

// Include the implementation.
#include "memory_resource_test_stream_ordered.ipp"
