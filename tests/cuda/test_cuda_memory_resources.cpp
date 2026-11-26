/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/memory_resource_name_gen.hpp"
#include "../common/memory_resource_test_basic.hpp"
#include "../common/memory_resource_test_host_accessible.hpp"
#include "vecmem/memory/cuda/async_device_memory_resource.hpp"
#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "vecmem/memory/cuda/host_memory_resource.hpp"
#include "vecmem/memory/cuda/managed_memory_resource.hpp"
#include "vecmem/utils/cuda/stream_wrapper.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

// GoogleTest include(s).
#include <gtest/gtest.h>

// Non-owning stream, to avoid destroying it after the runtime shut down.
static vecmem::cuda::stream_wrapper stream(cudaStreamPerThread);

// Memory resources.
static vecmem::cuda::device_memory_resource device_resource;
static vecmem::cuda::host_memory_resource host_resource;
static vecmem::cuda::managed_memory_resource managed_resource;
static vecmem::cuda::async_device_memory_resource async_device_resource(stream);

// Instantiate the allocation tests on all of the resources.
INSTANTIATE_TEST_SUITE_P(
    cuda_memory_resource_tests_basic, memory_resource_test_basic,
    testing::Values(&device_resource, &host_resource, &managed_resource,
                    &async_device_resource),
    vecmem::testing::memory_resource_name_gen(
        {{&device_resource, "device_resource"},
         {&host_resource, "host_resource"},
         {&managed_resource, "managed_resource"},
         {&async_device_resource, "async_device_resource"}}));

// Instantiate the full test suite on the host-accessible memory resources.
INSTANTIATE_TEST_SUITE_P(cuda_host_accessible_memory_resource_tests,
                         memory_resource_test_host_accessible,
                         testing::Values(&host_resource, &managed_resource),
                         vecmem::testing::memory_resource_name_gen(
                             {{&host_resource, "host_resource"},
                              {&managed_resource, "managed_resource"}}));
