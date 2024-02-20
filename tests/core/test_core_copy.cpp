/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Test include(s).
#include "../common/copy_tests.hpp"

// VecMem include(s).
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/utils/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <tuple>
#include <vector>

// Objects used in the test(s).
static vecmem::host_memory_resource core_host_resource;
static vecmem::copy core_copy;

// Instantiate the test suite(s).
INSTANTIATE_TEST_SUITE_P(core_copy_tests, copy_tests,
                         testing::Values(std::tie(core_copy, core_copy,
                                                  core_host_resource,
                                                  core_host_resource)));
