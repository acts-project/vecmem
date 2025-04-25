/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Test include(s).
#include "../common/copy_tests.hpp"
#include "../common/soa_copy_tests.hpp"

// VecMem include(s).
#include "vecmem/memory/hip/device_memory_resource.hpp"
#include "vecmem/memory/hip/host_memory_resource.hpp"
#include "vecmem/memory/hip/managed_memory_resource.hpp"
#include "vecmem/utils/copy.hpp"
#include "vecmem/utils/hip/async_copy.hpp"
#include "vecmem/utils/hip/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <memory>
#include <tuple>

// Objects used in the test(s).
static vecmem::hip::host_memory_resource hip_host_resource;
static vecmem::memory_resource* hip_host_resource_ptr = &hip_host_resource;
static vecmem::hip::device_memory_resource hip_device_resource;
static vecmem::memory_resource* hip_device_resource_ptr = &hip_device_resource;
static vecmem::hip::managed_memory_resource hip_managed_resource;
static vecmem::memory_resource* hip_managed_resource_ptr =
    &hip_managed_resource;
static vecmem::copy hip_host_copy;
static vecmem::copy* hip_host_copy_ptr = &hip_host_copy;
static vecmem::hip::copy hip_device_copy;
static vecmem::copy* hip_device_copy_ptr = &hip_device_copy;
static std::unique_ptr<vecmem::hip::stream_wrapper> hip_stream;
static std::unique_ptr<vecmem::hip::async_copy> hip_async_device_copy;
static vecmem::copy* hip_async_device_copy_ptr = nullptr;

/// Environment taking care of setting up and tearing down the async test types
class AsyncHIPCopyEnvironment : public ::testing::Environment {
public:
    /// Set up the environment
    void SetUp() override {
        hip_stream = std::make_unique<vecmem::hip::stream_wrapper>();
        hip_async_device_copy =
            std::make_unique<vecmem::hip::async_copy>(*hip_stream);
        hip_async_device_copy_ptr = hip_async_device_copy.get();
    }
    /// Tear down the environment
    void TearDown() override {
        hip_async_device_copy.reset();
        hip_stream.reset();
        hip_async_device_copy_ptr = nullptr;
    }
};

/// Register the environment
static ::testing::Environment* const async_hip_copy_env =
    ::testing::AddGlobalTestEnvironment(new AsyncHIPCopyEnvironment{});

/// The configurations to run the tests with. Skip tests with asynchronous
/// copies on "managed host memory" on Windows. (Using managed memory as
/// "device memory" does seem to be fine.)
static const auto hip_copy_configs = testing::Values(
    std::tie(hip_device_copy_ptr, hip_host_copy_ptr, hip_device_resource_ptr,
             hip_host_resource_ptr),
    std::tie(hip_async_device_copy_ptr, hip_host_copy_ptr,
             hip_device_resource_ptr, hip_host_resource_ptr),
    std::tie(hip_device_copy_ptr, hip_host_copy_ptr, hip_managed_resource_ptr,
             hip_host_resource_ptr),
    std::tie(hip_async_device_copy_ptr, hip_host_copy_ptr,
             hip_managed_resource_ptr, hip_host_resource_ptr),
    std::tie(hip_device_copy_ptr, hip_host_copy_ptr, hip_managed_resource_ptr,
             hip_managed_resource_ptr),
    std::tie(hip_async_device_copy_ptr, hip_host_copy_ptr,
             hip_managed_resource_ptr, hip_managed_resource_ptr),
    std::tie(hip_device_copy_ptr, hip_host_copy_ptr, hip_device_resource_ptr,
             hip_managed_resource_ptr),
    std::tie(hip_async_device_copy_ptr, hip_host_copy_ptr,
             hip_device_resource_ptr, hip_managed_resource_ptr));

// Instantiate the test suite(s).
INSTANTIATE_TEST_SUITE_P(hip_copy_tests, copy_tests, hip_copy_configs);
INSTANTIATE_TEST_SUITE_P(hip_soa_copy_tests_simple, soa_copy_tests_simple,
                         hip_copy_configs);
INSTANTIATE_TEST_SUITE_P(hip_soa_copy_tests_jagged, soa_copy_tests_jagged,
                         hip_copy_configs);
