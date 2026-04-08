/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Project include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/utils/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <tuple>

/// Test fixture for the SoA copy tests
class soa_copy_tests_hybrid_jagged
    : public ::testing::TestWithParam<
          std::tuple<vecmem::copy*&, vecmem::copy*&, vecmem::memory_resource*&,
                     vecmem::memory_resource*&>> {

protected:
    /// Access the "main" copy object
    inline vecmem::copy& main_copy();
    /// Access the "host" copy object
    inline vecmem::copy& host_copy();
    /// Access the "main" / "device" memory resource
    inline vecmem::memory_resource& main_mr();
    /// Access the "host" memory resource
    inline vecmem::memory_resource& host_mr();

};  // class soa_copy_tests_hybrid_jagged

// Include the implementation.
#include "soa_copy_tests_hybrid_jagged.ipp"
