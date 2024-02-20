/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Test include(s).
#include "../common/copy_tests.hpp"

// VecMem include(s).
#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "vecmem/memory/cuda/host_memory_resource.hpp"
#include "vecmem/memory/cuda/managed_memory_resource.hpp"
#include "vecmem/utils/copy.hpp"
#include "vecmem/utils/cuda/async_copy.hpp"
#include "vecmem/utils/cuda/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <tuple>
#include <vector>

// Objects used in the test(s).
static vecmem::cuda::host_memory_resource cuda_host_resource;
static vecmem::cuda::device_memory_resource cuda_device_resource;
static vecmem::cuda::managed_memory_resource cuda_managed_resource;
static vecmem::copy cuda_host_copy;
static vecmem::cuda::copy cuda_device_copy;
static vecmem::cuda::stream_wrapper cuda_stream;
static vecmem::cuda::async_copy cuda_async_device_copy{cuda_stream};

// Instantiate the test suite(s).
INSTANTIATE_TEST_SUITE_P(
    cuda_copy_tests, copy_tests,
    testing::Values(std::tie(cuda_device_copy, cuda_host_copy,
                             cuda_device_resource, cuda_host_resource),
                  //   std::tie(cuda_async_device_copy, cuda_host_copy,
                  //            cuda_device_resource, cuda_host_resource),
                    std::tie(cuda_device_copy, cuda_host_copy,
                             cuda_managed_resource, cuda_host_resource),
                  //   std::tie(cuda_async_device_copy, cuda_host_copy,
                  //            cuda_managed_resource, cuda_host_resource),
                    std::tie(cuda_device_copy, cuda_host_copy,
                             cuda_managed_resource, cuda_managed_resource),
                  //   std::tie(cuda_async_device_copy, cuda_host_copy,
                  //            cuda_managed_resource, cuda_managed_resource),
                    std::tie(cuda_device_copy, cuda_host_copy,
                             cuda_device_resource, cuda_managed_resource)/*,
                    std::tie(cuda_async_device_copy, cuda_host_copy,
                             cuda_device_resource, cuda_managed_resource)*/));
