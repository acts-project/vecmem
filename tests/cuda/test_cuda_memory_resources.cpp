/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/memory_resource_name_gen.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "vecmem/memory/cuda/host_memory_resource.hpp"
#include "vecmem/memory/cuda/managed_memory_resource.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <algorithm>
#include <cmath>
#include <vector>

// Memory resources.
static vecmem::cuda::device_memory_resource device_resource;
static vecmem::cuda::host_memory_resource host_resource;
static vecmem::cuda::managed_memory_resource managed_resource;

/// Base test case for the CUDA memory resources
///
/// This just makes sure that the memory resources defined in the
/// @c vecmem::cuda library are more-or-less functional.
///
class cuda_memory_resource_test
    : public testing::TestWithParam<vecmem::memory_resource*> {};

/// Perform some very basic tests that do not need host accessibility
TEST_P(cuda_memory_resource_test, allocations) {

    vecmem::memory_resource* resource = GetParam();
    for (std::size_t size = 1000; size < 100000; size += 1000) {
        void* ptr = resource->allocate(size);
        resource->deallocate(ptr, size);
    }
}

// Instantiate the allocation tests on all of the resources.
INSTANTIATE_TEST_SUITE_P(cuda_memory_resource_tests, cuda_memory_resource_test,
                         testing::Values(&device_resource, &host_resource,
                                         &managed_resource),
                         vecmem::testing::memory_resource_name_gen(
                             {{&device_resource, "device_resource"},
                              {&host_resource, "host_resource"},
                              {&managed_resource, "managed_resource"}}));

namespace {

/// Custom non-trivial type used in the tests.
struct TestType {
    TestType(int a, long b = 123) : m_a(a), m_b(b) {}
    int m_a;
    long m_b;
};
/// Helper operator for @c TestType
bool operator==(const TestType& value1, const TestType& value2) {
    return ((value1.m_a == value2.m_a) && (value1.m_b == value2.m_b));
}

/// Comparison operator for fundamental types
template <typename T>
bool almost_equal(const T& value1, const T& value2) {
    return (std::abs(value1 - value2) < 0.001);
}

/// Comparison operator for the custom test type
template <>
bool almost_equal<TestType>(const TestType& value1, const TestType& value2) {
    return (value1 == value2);
}

}  // namespace

/// Test case for host-accessible CUDA memory resources
///
/// Providing a slightly more elaborate test for memory resources that can be
/// read/written from host code.
///
class cuda_host_accessible_memory_resource_test
    : public cuda_memory_resource_test {

    protected:
    /// Function performing some basic tests using @c vecmem::vector
    template <typename T>
    void test_host_accessible_resource(vecmem::vector<T>& test_vector) {

        // Set up the test vector, and create a reference vector.
        std::vector<T> reference_vector;
        reference_vector.reserve(100);
        test_vector.reserve(100);

        // Fill them up with some dummy content.
        for (int i = 0; i < 20; ++i) {
            reference_vector.push_back(i * 2);
            test_vector.push_back(i * 2);
        }
        // Make sure that they are the same.
        EXPECT_EQ(reference_vector.size(), test_vector.size());
        EXPECT_TRUE(std::equal(reference_vector.begin(), reference_vector.end(),
                               test_vector.begin()));

        // Remove a couple of elements from the vectors.
        for (int i : {26, 38, 25}) {
            std::remove(reference_vector.begin(), reference_vector.end(), i);
            std::remove(test_vector.begin(), test_vector.end(), i);
        }
        // Make sure that they are still the same.
        EXPECT_EQ(reference_vector.size(), test_vector.size());
        EXPECT_TRUE(std::equal(reference_vector.begin(), reference_vector.end(),
                               test_vector.begin(), ::almost_equal<T>));
    }

};  // class cuda_host_accessible_memory_resource_test

/// Test the host accessible memory resource with an integer type.
TEST_P(cuda_host_accessible_memory_resource_test, int_value) {

    vecmem::vector<int> test_vector(GetParam());
    test_host_accessible_resource(test_vector);
}

/// Test the host accessible memory resource with a floating point type.
TEST_P(cuda_host_accessible_memory_resource_test, double_value) {

    vecmem::vector<double> test_vector(GetParam());
    test_host_accessible_resource(test_vector);
}

/// Test the host accessible memory resource with a custom type.
TEST_P(cuda_host_accessible_memory_resource_test, custom_value) {

    vecmem::vector< ::TestType> test_vector(GetParam());
    test_host_accessible_resource(test_vector);
}

// Instantiate the full test suite on the host-accessible memory resources.
INSTANTIATE_TEST_SUITE_P(cuda_host_accessible_memory_resource_tests,
                         cuda_host_accessible_memory_resource_test,
                         testing::Values(&host_resource, &managed_resource),
                         vecmem::testing::memory_resource_name_gen(
                             {{&host_resource, "host_resource"},
                              {&managed_resource, "managed_resource"}}));
