/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/memory_resource_name_gen.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/arena_memory_resource.hpp"
#include "vecmem/memory/binary_page_memory_resource.hpp"
#include "vecmem/memory/contiguous_memory_resource.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/memory/instrumenting_memory_resource.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

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

/// Test case for the core memory resources
///
/// This just makes sure that the memory resources defined in the
/// @c vecmem::core library are more-or-less functional. Detailed tests of the
/// different memory resources are implemented in other test cases.
///
class core_memory_resource_test
    : public testing::TestWithParam<vecmem::memory_resource*> {

protected:
    /// Function performing some basic tests using @c vecmem::vector
    template <typename T>
    void test_resource(vecmem::vector<T>& test_vector) {

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
            (void)std::remove(reference_vector.begin(), reference_vector.end(),
                              i);
            (void)std::remove(test_vector.begin(), test_vector.end(), i);
        }
        // Make sure that they are still the same.
        EXPECT_EQ(reference_vector.size(), test_vector.size());
        EXPECT_TRUE(std::equal(reference_vector.begin(), reference_vector.end(),
                               test_vector.begin(), ::almost_equal<T>));
    }

};  // class core_memory_resource_test

/// Test the memory resource with an integer type.
TEST_P(core_memory_resource_test, int_value) {

    vecmem::vector<int> test_vector(GetParam());
    test_resource(test_vector);
}

/// Test the memory resource with a floating point type.
TEST_P(core_memory_resource_test, double_value) {

    vecmem::vector<double> test_vector(GetParam());
    test_resource(test_vector);
}

/// Test the memory resource with a custom type.
TEST_P(core_memory_resource_test, custom_value) {

    vecmem::vector< ::TestType> test_vector(GetParam());
    test_resource(test_vector);
}

/// Test case for the "stress tests"
class core_memory_resource_stress_test
    : public testing::TestWithParam<vecmem::memory_resource*> {};

/// Test that the memory resource would behave correctly with a large number
/// of allocations/de-allocations.
TEST_P(core_memory_resource_stress_test, stress_test) {

    // Repeat the allocations multiple times.
    for (int i = 0; i < 100; ++i) {

        // Create an object that would hold on to the allocated memory
        // "for one iteration".
        std::vector<vecmem::vector<int> > vectors;

        // Fill a random number of vectors.
        const int n_vectors = std::rand() % 100;
        for (int j = 0; j < n_vectors; ++j) {

            // Fill them with a random number of "constant" elements.
            vectors.emplace_back(GetParam());
            const int n_elements = std::rand() % 100;
            for (int k = 0; k < n_elements; ++k) {
                vectors.back().push_back(j);
            }
        }

        // Check that all vectors have the intended content after all of this.
        for (int j = 0; j < n_vectors; ++j) {
            for (int value : vectors.at(j)) {
                EXPECT_EQ(value, j);
            }
        }
    }
}

// Memory resources to use in the test.
static vecmem::host_memory_resource host_resource;
static vecmem::binary_page_memory_resource binary_resource(host_resource);
static vecmem::contiguous_memory_resource contiguous_resource(host_resource,
                                                              20000);
static vecmem::arena_memory_resource arena_resource(host_resource, 20000,
                                                    10000000);
static vecmem::instrumenting_memory_resource instrumenting_resource(
    host_resource);

// Instantiate the test suite(s).
INSTANTIATE_TEST_SUITE_P(
    core_memory_resource_tests, core_memory_resource_test,
    testing::Values(&host_resource, &binary_resource, &contiguous_resource,
                    &arena_resource, &instrumenting_resource),
    vecmem::testing::memory_resource_name_gen(
        {{&host_resource, "host_resource"},
         {&binary_resource, "binary_resource"},
         {&contiguous_resource, "contiguous_resource"},
         {&arena_resource, "arena_resource"},
         {&instrumenting_resource, "instrumenting_resource"}}));

INSTANTIATE_TEST_SUITE_P(
    core_memory_resource_stress_tests, core_memory_resource_stress_test,
    testing::Values(&host_resource, &binary_resource, &arena_resource,
                    &instrumenting_resource),
    vecmem::testing::memory_resource_name_gen(
        {{&host_resource, "host_resource"},
         {&binary_resource, "binary_resource"},
         {&arena_resource, "arena_resource"},
         {&instrumenting_resource, "instrumenting_resource"}}));
