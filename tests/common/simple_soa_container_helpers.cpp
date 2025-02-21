/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "simple_soa_container_helpers.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <type_traits>

namespace vecmem::testing {

void fill(simple_soa_container::host& obj) {

    obj.reserve(10);
    for (std::size_t i = 0; i < 10; ++i) {
        obj.push_back(
            {55, 1.0f * static_cast<float>(i), 3.141592f, static_cast<int>(i)});
    }
}

void compare(const simple_soa_container::const_view& view1,
             const simple_soa_container::const_view& view2) {

    // Create device containers on top of the views.
    const simple_soa_container::const_device device1{view1};
    const simple_soa_container::const_device device2{view2};

    // Check the size of the containers.
    EXPECT_EQ(device1.size(), device2.size());

    // Compare the scalar variables.
    EXPECT_EQ(device1.count(), device2.count());
    EXPECT_FLOAT_EQ(device1.average(), device2.average());

    // Compare the vector variables. Both vectors need to have the same
    // variables but not necessarily in the same order.
    auto compare_vector = [](const auto& lhs, const auto& rhs) {
        ASSERT_EQ(lhs.size(), rhs.size());
        std::vector<bool> matched(rhs.size(), false);
        for (unsigned int i = 0; i < lhs.size(); ++i) {
            for (unsigned int j = 0; j < rhs.size(); ++j) {
                if (lhs[i] == rhs[j] && !matched[j]) {
                    matched[j] = true;
                    break;
                }
                if (j == rhs.size() - 1) {
                    FAIL() << "The vectors are not equal";
                }
            }
        }
    };
    compare_vector(device1.measurement(), device2.measurement());
    compare_vector(device1.index(), device2.index());
}

void make_buffer(simple_soa_container::buffer& buffer, memory_resource& main_mr,
                 memory_resource&, data::buffer_type buffer_type) {

    switch (buffer_type) {
        case data::buffer_type::fixed_size:
            buffer = {10, main_mr, buffer_type};
            break;
        case data::buffer_type::resizable:
            buffer = {20, main_mr, buffer_type};
            break;
        default:
            throw std::runtime_error("Unsupported buffer type");
    }
}

}  // namespace vecmem::testing
