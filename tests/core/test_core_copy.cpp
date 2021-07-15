/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include "vecmem/containers/data/jagged_vector_buffer.hpp"
#include "vecmem/containers/data/vector_buffer.hpp"
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/jagged_device_vector.hpp"
#include "vecmem/containers/jagged_vector.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/utils/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

/// Test case for testing @c vecmem::copy
class core_copy_test : public testing::Test {

protected:
    /// Memory resource for the test(s)
    vecmem::host_memory_resource m_resource;
    /// Copy object for the test(s)
    vecmem::copy m_copy;

};  // class core_copy_test

/// Tests for copying 1-dimensional vectors
TEST_F(core_copy_test, vector) {

    // Create a reference vector.
    vecmem::vector<int> reference = {{1, 5, 6, 74, 234, 43, 22}, &m_resource};

    // Create a view of its data.
    auto reference_data = vecmem::get_data(reference);

    // Make a copy of this reference.
    auto copy_data = m_copy.to(reference_data, m_resource);

    // Create device vectors over the two, and compare them.
    vecmem::device_vector<int> reference_vector(reference_data);
    vecmem::device_vector<int> copy_vector(copy_data);
    EXPECT_EQ(reference_vector.size(), copy_vector.size());
    auto reference_itr = reference_vector.begin();
    auto copy_itr = copy_vector.begin();
    for (; (reference_itr != reference_vector.end()) &&
           (copy_itr != copy_vector.end());
         ++reference_itr, ++copy_itr) {
        EXPECT_EQ(*reference_itr, *copy_itr);
    }
}

/// Tests for copying jagged vectors
TEST_F(core_copy_test, jagged_vector) {

    // Create a reference vector.
    vecmem::jagged_vector<int> reference = {
        {{{1, 2, 3, 4, 5}, &m_resource},
         {{6, 7}, &m_resource},
         {{8, 9, 10, 11}, &m_resource},
         {{12, 13, 14, 15, 16, 17, 18}, &m_resource},
         {{19}, &m_resource},
         {{20}, &m_resource}},
        &m_resource};

    // Create a view of its data.
    auto reference_data = vecmem::get_data(reference);

    // Make a copy of this reference.
    auto copy_data = m_copy.to(reference_data, m_resource);

    // Create device vectors over the two, and compare them.
    vecmem::jagged_device_vector<int> reference_vector(reference_data);
    vecmem::jagged_device_vector<int> copy_vector(copy_data);
    EXPECT_EQ(reference_vector.size(), copy_vector.size());
    auto reference_itr = reference_vector.begin();
    auto copy_itr = copy_vector.begin();
    for (; (reference_itr != reference_vector.end()) &&
           (copy_itr != copy_vector.end());
         ++reference_itr, ++copy_itr) {
        EXPECT_EQ(reference_itr->size(), copy_itr->size());
        auto reference_itr2 = reference_itr->begin();
        auto copy_itr2 = copy_itr->begin();
        for (; (reference_itr2 != reference_itr->end()) &&
               (copy_itr2 != copy_itr->end());
             ++reference_itr2, ++copy_itr2) {
            EXPECT_EQ(*reference_itr2, *copy_itr2);
        }
    }
}
