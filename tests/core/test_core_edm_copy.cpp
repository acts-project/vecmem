/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/soa_copy_tests.hpp"

// Project include(s).
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/utils/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

/// Host memory resource to use in the tests.
static vecmem::host_memory_resource host_mr;

/// Host copy object to use in the tests.
static vecmem::copy host_copy;

// Instantiate the test suites.
INSTANTIATE_TEST_SUITE_P(core_soa_copy_tests_simple, soa_copy_tests_simple,
                         testing::Values(std::tie(host_mr, host_mr, host_copy,
                                                  host_copy)));
INSTANTIATE_TEST_SUITE_P(core_soa_copy_tests_jagged, soa_copy_tests_jagged,
                         testing::Values(std::tie(host_mr, host_mr, host_copy,
                                                  host_copy)));
