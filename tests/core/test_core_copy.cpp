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
#include <algorithm>
#include <tuple>
#include <vector>

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

/// Tests for copying 1-dimensional vector buffers
TEST_F(core_copy_test, vector_buffer) {

    // Set up a reference vector.
    const vecmem::vector<int> reference = {{1, 5, 6, 74, 234, 43, 22},
                                           &m_resource};
    using vecmem_size_type = vecmem::device_vector<int>::size_type;
    const vecmem_size_type size =
        static_cast<vecmem_size_type>(reference.size());

    // Create a non-resizable vector buffer.
    vecmem::data::vector_buffer<int> source_data(size, m_resource);
    m_copy.setup(source_data);
    // Fill it with data.
    vecmem::device_vector<int> source_data_vec(source_data);
    EXPECT_EQ(static_cast<std::size_t>(source_data_vec.size()),
              static_cast<std::size_t>(reference.size()));
    EXPECT_EQ(source_data_vec.capacity(), source_data_vec.size());
    for (vecmem_size_type i = 0; i < reference.size(); ++i) {
        source_data_vec.at(i) = reference.at(i);
    }

    // Make a copy of it into a buffer.
    auto copy_data = m_copy.to(source_data, m_resource);
    // Check the copy.
    vecmem::device_vector<int> copy_data_vec(copy_data);
    EXPECT_EQ(source_data_vec.size(), copy_data_vec.size());
    auto source_data_vec_itr = source_data_vec.begin();
    auto copy_data_vec_itr = copy_data_vec.begin();
    for (; (source_data_vec_itr != source_data_vec.end()) &&
           (copy_data_vec_itr != copy_data_vec.end());
         ++source_data_vec_itr, ++copy_data_vec_itr) {
        EXPECT_EQ(*source_data_vec_itr, *copy_data_vec_itr);
    }

    // Make a copy into a host vector.
    vecmem::vector<int> copy_vec(&m_resource);
    m_copy(source_data, copy_vec);
    // Check the copy.
    EXPECT_EQ(reference.size(), copy_vec.size());
    auto reference_itr = reference.begin();
    auto copy_vec_itr = copy_vec.begin();
    for (;
         (reference_itr != reference.end()) && (copy_vec_itr != copy_vec.end());
         ++reference_itr, ++copy_vec_itr) {
        EXPECT_EQ(*reference_itr, *copy_vec_itr);
    }
}

/// Tests for copying 1-dimensional resizable vector buffers
TEST_F(core_copy_test, resizable_vector_buffer) {

    // Set up a reference vector.
    const vecmem::vector<int> reference = {{1, 5, 6, 74, 234, 43, 22},
                                           &m_resource};

    // Create a resizable vector buffer.
    using vecmem_size_type = vecmem::device_vector<int>::size_type;
    const vecmem_size_type capacity =
        static_cast<vecmem_size_type>(reference.size() + 5);
    vecmem::data::vector_buffer<int> source_data(capacity, 0, m_resource);
    m_copy.setup(source_data);
    // Fill it with data.
    vecmem::device_vector<int> source_data_vec(source_data);
    EXPECT_EQ(source_data_vec.size(), 0u);
    EXPECT_EQ(source_data_vec.capacity(), capacity);
    for (int data : reference) {
        source_data_vec.push_back(data);
    }
    EXPECT_EQ(static_cast<std::size_t>(source_data_vec.size()),
              static_cast<std::size_t>(reference.size()));
    EXPECT_EQ(source_data_vec.capacity(), capacity);

    // Make a copy of it into a buffer.
    auto copy_data = m_copy.to(source_data, m_resource);
    // Check the copy.
    vecmem::device_vector<int> copy_data_vec(copy_data);
    EXPECT_EQ(source_data_vec.size(), copy_data_vec.size());
    auto source_data_vec_itr = source_data_vec.begin();
    auto copy_data_vec_itr = copy_data_vec.begin();
    for (; (source_data_vec_itr != source_data_vec.end()) &&
           (copy_data_vec_itr != copy_data_vec.end());
         ++source_data_vec_itr, ++copy_data_vec_itr) {
        EXPECT_EQ(*source_data_vec_itr, *copy_data_vec_itr);
    }

    // Make a copy into a host vector.
    vecmem::vector<int> copy_vec(&m_resource);
    m_copy(source_data, copy_vec);
    // Check the copy.
    EXPECT_EQ(reference.size(), copy_vec.size());
    auto reference_itr = reference.begin();
    auto copy_vec_itr = copy_vec.begin();
    for (;
         (reference_itr != reference.end()) && (copy_vec_itr != copy_vec.end());
         ++reference_itr, ++copy_vec_itr) {
        EXPECT_EQ(*reference_itr, *copy_vec_itr);
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

/// Tests for copying jagged vector buffers
TEST_F(core_copy_test, jagged_vector_buffer) {

    // Create a reference vector.
    const vecmem::jagged_vector<int> reference = {
        {{{1, 2, 3, 4, 5}, &m_resource},
         {{6, 7}, &m_resource},
         {{8, 9, 10, 11}, &m_resource},
         {{12, 13, 14, 15, 16, 17, 18}, &m_resource},
         {{19}, &m_resource},
         {{20}, &m_resource}},
        &m_resource};

    // Create a non-resizable vector buffer.
    std::vector<std::size_t> sizes(reference.size());
    std::transform(reference.begin(), reference.end(), sizes.begin(),
                   [](const auto& vec) { return vec.size(); });
    vecmem::data::jagged_vector_buffer<int> source_data(sizes, m_resource);
    m_copy.setup(source_data);
    // Fill it with data.
    using vecmem_size_type = vecmem::device_vector<int>::size_type;
    vecmem::jagged_device_vector<int> source_data_vec(source_data);
    EXPECT_EQ(reference.size(), source_data_vec.size());
    for (vecmem_size_type i = 0; i < reference.size(); ++i) {
        EXPECT_EQ(static_cast<std::size_t>(reference.at(i).size()),
                  static_cast<std::size_t>(source_data_vec.at(i).size()));
        for (vecmem_size_type j = 0; j < reference.at(i).size(); ++j) {
            source_data_vec.at(i).at(j) = reference.at(i).at(j);
        }
    }

    // Make a copy of it into a buffer.
    auto copy_data = m_copy.to(source_data, m_resource);
    // Check the copy.
    vecmem::jagged_device_vector<int> copy_data_vec(copy_data);
    EXPECT_EQ(source_data_vec.size(), copy_data_vec.size());
    auto source_data_vec_itr = source_data_vec.begin();
    auto copy_data_vec_itr = copy_data_vec.begin();
    for (; (source_data_vec_itr != source_data_vec.end()) &&
           (copy_data_vec_itr != copy_data_vec.end());
         ++source_data_vec_itr, ++copy_data_vec_itr) {
        EXPECT_EQ(source_data_vec_itr->size(), copy_data_vec_itr->size());
        auto source_data_vec_itr_itr = source_data_vec_itr->begin();
        auto copy_data_vec_itr_itr = copy_data_vec_itr->begin();
        for (; (source_data_vec_itr_itr != source_data_vec_itr->end()) &&
               (copy_data_vec_itr_itr != copy_data_vec_itr->end());
             ++source_data_vec_itr_itr, ++copy_data_vec_itr_itr) {
            EXPECT_EQ(*source_data_vec_itr_itr, *copy_data_vec_itr_itr);
        }
    }

    // Make a copy into a host vector.
    vecmem::jagged_vector<int> copy_vec(&m_resource);
    m_copy(source_data, copy_vec);
    // Check the copy.
    EXPECT_EQ(reference.size(), copy_vec.size());
    auto reference_itr = reference.begin();
    auto copy_vec_itr = copy_vec.begin();
    for (;
         (reference_itr != reference.end()) && (copy_vec_itr != copy_vec.end());
         ++reference_itr, ++copy_vec_itr) {
        EXPECT_EQ(reference_itr->size(), copy_vec_itr->size());
        auto reference_itr_itr = reference_itr->begin();
        auto copy_vec_itr_itr = copy_vec_itr->begin();
        for (; (reference_itr_itr != reference_itr->end()) &&
               (copy_vec_itr_itr != copy_vec_itr->end());
             ++reference_itr_itr, ++copy_vec_itr_itr) {
            EXPECT_EQ(*reference_itr_itr, *copy_vec_itr_itr);
        }
    }
}

/// Tests for copying resizable jagged vector buffers
TEST_F(core_copy_test, resizable_jagged_vector_buffer) {

    // Create a reference vector.
    const vecmem::jagged_vector<int> reference = {
        {{{1, 2, 3, 4, 5}, &m_resource},
         {{6, 7}, &m_resource},
         {{8, 9, 10, 11}, &m_resource},
         {{12, 13, 14, 15, 16, 17, 18}, &m_resource},
         {{19}, &m_resource},
         {{20}, &m_resource}},
        &m_resource};

    // Create a resizable vector buffer.
    std::vector<std::size_t> capacities(reference.size());
    std::transform(reference.begin(), reference.end(), capacities.begin(),
                   [](const auto& vec) { return vec.size() + 5; });
    vecmem::data::jagged_vector_buffer<int> source_data(
        std::vector<std::size_t>(capacities.size(), 0), capacities, m_resource);
    m_copy.setup(source_data);
    // Fill it with data.
    vecmem::jagged_device_vector<int> source_data_vec(source_data);
    EXPECT_EQ(reference.size(), source_data_vec.size());
    for (std::size_t i = 0; i < reference.size(); ++i) {
        for (std::size_t j = 0; j < reference.at(i).size(); ++j) {
            source_data_vec.at(i).push_back(reference.at(i).at(j));
        }
        EXPECT_EQ(static_cast<std::size_t>(reference.at(i).size()),
                  static_cast<std::size_t>(source_data_vec.at(i).size()));
        EXPECT_EQ(source_data_vec.at(i).capacity(),
                  source_data_vec.at(i).size() + 5);
    }

    // Make a copy of it into a buffer.
    auto copy_data = m_copy.to(source_data, m_resource);
    // Check the copy.
    vecmem::jagged_device_vector<int> copy_data_vec(copy_data);
    EXPECT_EQ(source_data_vec.size(), copy_data_vec.size());
    auto source_data_vec_itr = source_data_vec.begin();
    auto copy_data_vec_itr = copy_data_vec.begin();
    for (; (source_data_vec_itr != source_data_vec.end()) &&
           (copy_data_vec_itr != copy_data_vec.end());
         ++source_data_vec_itr, ++copy_data_vec_itr) {
        EXPECT_EQ(source_data_vec_itr->size(), copy_data_vec_itr->size());
        auto source_data_vec_itr_itr = source_data_vec_itr->begin();
        auto copy_data_vec_itr_itr = copy_data_vec_itr->begin();
        for (; (source_data_vec_itr_itr != source_data_vec_itr->end()) &&
               (copy_data_vec_itr_itr != copy_data_vec_itr->end());
             ++source_data_vec_itr_itr, ++copy_data_vec_itr_itr) {
            EXPECT_EQ(*source_data_vec_itr_itr, *copy_data_vec_itr_itr);
        }
    }

    // Make a copy into a host vector.
    vecmem::jagged_vector<int> copy_vec(&m_resource);
    m_copy(source_data, copy_vec);
    // Check the copy.
    EXPECT_EQ(reference.size(), copy_vec.size());
    auto reference_itr = reference.begin();
    auto copy_vec_itr = copy_vec.begin();
    for (;
         (reference_itr != reference.end()) && (copy_vec_itr != copy_vec.end());
         ++reference_itr, ++copy_vec_itr) {
        EXPECT_EQ(reference_itr->size(), copy_vec_itr->size());
        auto reference_itr_itr = reference_itr->begin();
        auto copy_vec_itr_itr = copy_vec_itr->begin();
        for (; (reference_itr_itr != reference_itr->end()) &&
               (copy_vec_itr_itr != copy_vec_itr->end());
             ++reference_itr_itr, ++copy_vec_itr_itr) {
            EXPECT_EQ(*reference_itr_itr, *copy_vec_itr_itr);
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
