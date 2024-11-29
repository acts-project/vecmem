/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/memory_resource_name_gen.hpp"
#include "../common/memory_resource_test_basic.hpp"
#include "../common/memory_resource_test_host_accessible.hpp"
#include "vecmem/memory/metal/shared_memory_resource.hpp"
#include "vecmem/utils/metal/device_wrapper.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// Memory resources.
static vecmem::metal::device_wrapper metal_device;
static vecmem::metal::shared_memory_resource shared_resource{metal_device};

// Instantiate the allocation tests on all of the resources.
INSTANTIATE_TEST_SUITE_P(metal_basic_memory_resource_tests,
                         memory_resource_test_basic,
                         testing::Values(&shared_resource),
                         vecmem::testing::memory_resource_name_gen(
                             {{&shared_resource, "shared_resource"}}));

// Instantiate the full test suite on the host-accessible memory resources.
INSTANTIATE_TEST_SUITE_P(metal_host_accessible_memory_resource_tests,
                         memory_resource_test_host_accessible,
                         testing::Values(&shared_resource),
                         vecmem::testing::memory_resource_name_gen(
                             {{&shared_resource, "shared_resource"}}));
