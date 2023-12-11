/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/soa_copy_tests.hpp"

// Project include(s).
#include "vecmem/memory/sycl/device_memory_resource.hpp"
#include "vecmem/memory/sycl/host_memory_resource.hpp"
#include "vecmem/utils/copy.hpp"
#include "vecmem/utils/sycl/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

/// Host memory resource to use in the tests.
static vecmem::sycl::host_memory_resource host_mr;
/// Device memory resource to use in the tests.
static vecmem::sycl::device_memory_resource device_mr;

/// Host copy object to use in the tests.
static vecmem::copy host_copy;
/// Synchronous device copy object to use in the tests.
static vecmem::sycl::copy device_copy{{}};

// Instantiate the test suites.
INSTANTIATE_TEST_SUITE_P(sycl_soa_copy_tests_simple, soa_copy_tests_simple,
                         testing::Values(std::tie(host_mr, device_mr, host_copy,
                                                  device_copy)));
INSTANTIATE_TEST_SUITE_P(sycl_soa_copy_tests_jagged, soa_copy_tests_jagged,
                         testing::Values(std::tie(host_mr, device_mr, host_copy,
                                                  device_copy)));
