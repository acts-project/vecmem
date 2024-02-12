/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/device_vector.hpp"
#include "vecmem/containers/vector.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

namespace {

/// Helper function for comparing the contents of device vectors
template <typename T>
void compare(const vecmem::device_vector<T>& v1,
             const vecmem::device_vector<T>& v2) {

    ASSERT_EQ(v1.size(), v2.size());
    for (typename vecmem::device_vector<T>::size_type i = 0; i < v1.size();
         ++i) {
        EXPECT_EQ(v1[i], v2[i]);
    }
}

}  // namespace

/// Test for copying 1-dimensional vectors
TEST_P(copy_tests, vector) {

    // Access the test parameters.
    vecmem::copy& copy = std::get<0>(GetParam());
    vecmem::memory_resource& main_mr = std::get<1>(GetParam());
    vecmem::memory_resource& host_mr = std::get<2>(GetParam());

    // Create a reference vector.
    vecmem::vector<int> reference = {{1, 5, 6, 74, 234, 43, 22}, &host_mr};

    // Create a view of its data.
    auto reference_data = vecmem::get_data(reference);

    // Make a copy of this reference.
    auto device_copy_data = copy.to(reference_data, main_mr);
    auto host_copy_data = copy.to(device_copy_data, host_mr);

    // Create device vectors for the comparison.
    vecmem::device_vector<int> reference_vector(reference_data);
    vecmem::device_vector<int> copy_vector(host_copy_data);
    compare(reference_vector, copy_vector);
}

/// Test for copying 1-dimensional (const) vectors
TEST_P(copy_tests, const_vector) {

    // Access the test parameters.
    vecmem::copy& copy = std::get<0>(GetParam());
    vecmem::memory_resource& main_mr = std::get<1>(GetParam());
    vecmem::memory_resource& host_mr = std::get<2>(GetParam());

    // Create a reference vector.
    const vecmem::vector<int> reference = {{1, 5, 6, 74, 234, 43, 22},
                                           &host_mr};

    // Create a view of its data.
    const auto reference_data = vecmem::get_data(reference);

    // Make a copy of this reference.
    const auto device_copy_data = copy.to(reference_data, main_mr);
    const auto host_copy_data = copy.to(device_copy_data, host_mr);

    // Create device vectors over the two, and compare them.
    vecmem::device_vector<const int> reference_vector(reference_data);
    vecmem::device_vector<const int> copy_vector(host_copy_data);
    compare(reference_vector, copy_vector);
}

/// Test for copying 1-dimensional, fixed size vector buffers
TEST_P(copy_tests, fixed_vector_buffer) {

    // Access the test parameters.
    vecmem::copy& copy = std::get<0>(GetParam());
    vecmem::memory_resource& main_mr = std::get<1>(GetParam());
    vecmem::memory_resource& host_mr = std::get<2>(GetParam());

    // Set up a reference vector.
    const vecmem::vector<int> reference = {{1, 5, 6, 74, 234, 43, 22},
                                           &host_mr};
    using vecmem_size_type = vecmem::device_vector<int>::size_type;
    const vecmem_size_type size =
        static_cast<vecmem_size_type>(reference.size());

    // Create non-resizable device and host buffers, with the "exact sizes".
    vecmem::data::vector_buffer<int> device_buffer(size, main_mr);
    copy.setup(device_buffer);
    vecmem::data::vector_buffer<int> host_buffer(size, host_mr);
    copy.setup(host_buffer);


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
         (reference_itr != reference.end()) && (copy_vec_itr !=
         copy_vec.end());
         ++reference_itr, ++copy_vec_itr) {
        EXPECT_EQ(*reference_itr, *copy_vec_itr);
    }
}
