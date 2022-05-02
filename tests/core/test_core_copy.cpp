/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
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

// System include(s).
#include <tuple>

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

/// Tests for copying 1-dimensional constant vectors
TEST_F(core_copy_test, const_vector) {

    // Create a reference vector.
    const vecmem::vector<int> reference = {{1, 5, 6, 74, 234, 43, 22},
                                           &m_resource};

    // Create a view of its data.
    auto reference_data = vecmem::get_data(reference);

    // Make a copy of this reference.
    auto copy_data = m_copy.to(reference_data, m_resource);

    // Create device vectors over the two, and compare them.
    vecmem::device_vector<const int> reference_vector(reference_data);
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

/// Tests for copying constant jagged vectors
TEST_F(core_copy_test, const_jagged_vector) {

    // Create a reference vector.
    const vecmem::jagged_vector<int> reference = {
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
    vecmem::jagged_device_vector<const int> reference_vector(reference_data);
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

/// Tests with @c vecmem::copy::memset
TEST_F(core_copy_test, memset) {

    // Size for the 1-dimensional buffer(s).
    static const unsigned int BUFFER1_SIZE = 10;

    // Test(s) with a 1-dimensional buffer.
    vecmem::data::vector_buffer<int> buffer1(BUFFER1_SIZE, m_resource);
    m_copy.memset(buffer1, 5);
    vecmem::vector<int> vector1(&m_resource);
    m_copy(buffer1, vector1);
    EXPECT_EQ(vector1.size(), BUFFER1_SIZE);
    static const int REFERENCE = 0x05050505;
    for (int value : vector1) {
        EXPECT_EQ(value, REFERENCE);
    }

    vecmem::data::vector_buffer<std::tuple<unsigned int, float, double> >
        buffer2(BUFFER1_SIZE, m_resource);
    m_copy.memset(buffer2, 0);
    vecmem::vector<std::tuple<unsigned int, float, double> > vector2(
        &m_resource);
    m_copy(buffer2, vector2);
    EXPECT_EQ(vector2.size(), BUFFER1_SIZE);
    for (const std::tuple<unsigned int, float, double>& value : vector2) {
        EXPECT_EQ(std::get<0>(value), 0u);
        EXPECT_EQ(std::get<1>(value), 0.f);
        EXPECT_EQ(std::get<2>(value), 0.);
    }

    // Size(s) for the jagged buffer(s).
    static const std::vector<std::size_t> BUFFER2_SIZES = {3, 6, 6, 3, 0,
                                                           2, 7, 2, 4, 0};

    // Test(s) with a jagged buffer.
    vecmem::data::jagged_vector_buffer<int> buffer3(BUFFER2_SIZES, m_resource);
    m_copy.setup(buffer3);
    m_copy.memset(buffer3, 5);
    vecmem::jagged_vector<int> vector3(&m_resource);
    m_copy(buffer3, vector3);
    EXPECT_EQ(vector3.size(), BUFFER2_SIZES.size());
    for (std::size_t i = 0; i < vector3.size(); ++i) {
        EXPECT_EQ(vector3.at(i).size(), BUFFER2_SIZES.at(i));
        for (int value : vector3.at(i)) {
            EXPECT_EQ(value, REFERENCE);
        }
    }
}
