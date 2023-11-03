/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "edm_simple_container_helpers.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

namespace vecmem::testing {

void fill(simple_container::host& obj) {

    obj.resize(10);
    simple_container::count::get(obj) = 55;
    simple_container::average::get(obj) = 3.141592f;
    for (std::size_t i = 0; i < obj.size(); ++i) {
        simple_container::measurement::get(obj)[i] =
            1.0f * static_cast<float>(i);
        simple_container::index::get(obj)[i] = static_cast<int>(i);
    }
}

void fill(simple_container::device& obj) {

    simple_container::count::get(obj) = 55;
    simple_container::average::get(obj) = 3.141592f;
    for (std::size_t i = 0; i < obj.capacity(); ++i) {
        const simple_container::device::size_type ii = obj.push_back_default();
        simple_container::measurement::get(obj)[ii] =
            1.0f * static_cast<float>(ii);
        simple_container::index::get(obj)[ii] = static_cast<int>(ii);
    }
}

template <typename T>
void compare(const vector<T>& lhs, const vector<T>& rhs) {

    // Check the size of the vectors.
    ASSERT_EQ(lhs.size(), rhs.size());

    // Check the content of the vectors.
    for (typename vector<T>::size_type i = 0; i < lhs.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            EXPECT_FLOAT_EQ(lhs[i], rhs[i]);
        } else {
            EXPECT_EQ(lhs[i], rhs[i]);
        }
    }
}

template <typename T>
void compare(const device_vector<T>& lhs, const device_vector<T>& rhs) {

    // Check the size of the vectors.
    ASSERT_EQ(lhs.size(), rhs.size());

    // Check the content of the vectors.
    for (typename device_vector<T>::size_type i = 0; i < lhs.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            EXPECT_FLOAT_EQ(lhs[i], rhs[i]);
        } else {
            EXPECT_EQ(lhs[i], rhs[i]);
        }
    }
}

template <typename T>
void compare(const vector<T>& lhs, const device_vector<const T>& rhs) {

    // Check the size of the vectors.
    ASSERT_EQ(static_cast<int>(lhs.size()), static_cast<int>(rhs.size()));

    // Check the content of the vectors.
    for (typename device_vector<T>::size_type i = 0; i < lhs.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            EXPECT_FLOAT_EQ(lhs[i], rhs[i]);
        } else {
            EXPECT_EQ(lhs[i], rhs[i]);
        }
    }
}

void compare(const simple_container::host& lhs,
             const simple_container::host& rhs) {

    // Check the size of the containers.
    EXPECT_EQ(lhs.size(), rhs.size());

    // Compare the scalar variables.
    EXPECT_EQ(simple_container::count::get(lhs),
              simple_container::count::get(rhs));
    EXPECT_FLOAT_EQ(simple_container::average::get(lhs),
                    simple_container::average::get(rhs));

    // Compare the vector variables.
    compare(simple_container::measurement::get(lhs),
            simple_container::measurement::get(rhs));
    compare(simple_container::index::get(lhs),
            simple_container::index::get(rhs));
}

void compare(const simple_container::const_device& lhs,
             const simple_container::const_device& rhs) {

    // Check the size of the containers.
    EXPECT_EQ(lhs.size(), rhs.size());

    // Compare the scalar variables.
    EXPECT_EQ(simple_container::count::get(lhs),
              simple_container::count::get(rhs));
    EXPECT_FLOAT_EQ(simple_container::average::get(lhs),
                    simple_container::average::get(rhs));

    // Compare the vector variables.
    compare(simple_container::measurement::get(lhs),
            simple_container::measurement::get(rhs));
    compare(simple_container::index::get(lhs),
            simple_container::index::get(rhs));
}

void compare(const simple_container::host& lhs,
             const simple_container::const_device& rhs) {

    // Check the size of the containers.
    EXPECT_EQ(lhs.size(), rhs.size());

    // Compare the scalar variables.
    EXPECT_EQ(simple_container::count::get(lhs),
              simple_container::count::get(rhs));
    EXPECT_FLOAT_EQ(simple_container::average::get(lhs),
                    simple_container::average::get(rhs));

    // Compare the vector variables.
    compare(simple_container::measurement::get(lhs),
            simple_container::measurement::get(rhs));
    compare(simple_container::index::get(lhs),
            simple_container::index::get(rhs));
}

}  // namespace vecmem::testing
