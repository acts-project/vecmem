/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/edm_simple_container.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/utils/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

/// Test case for the EDM code
class core_edm_test : public testing::Test {

protected:
    /// Memory resource for the test(s)
    vecmem::host_memory_resource m_resource;
    /// Helper object for the memory copies.
    vecmem::copy m_copy;

};  // class core_device_container_test

TEST_F(core_edm_test, simple_view) {

    // Test the creation of simple views.
    vecmem::testing::simple_container::view view1;
    vecmem::testing::simple_container::const_view view2{view1};
}

TEST_F(core_edm_test, simple_buffer) {

    // Test the creation of fixed sized and resizable, at the same time "simple"
    // buffers.
    vecmem::testing::simple_container::buffer buffer1(
        10, m_resource, vecmem::data::buffer_type::fixed_size);
    vecmem::testing::simple_container::buffer buffer2(
        10, m_resource, vecmem::data::buffer_type::resizable);

    // "Create" views off of the buffers.
    vecmem::testing::simple_container::view view1{buffer1}, view2{buffer2};
    vecmem::testing::simple_container::const_view view3{buffer1},
        view4{buffer2};

    // Check the views.
    EXPECT_NE(view1.get<0>(), nullptr);
    EXPECT_EQ(view1.get<1>().size(), 10u);
    EXPECT_EQ(view1.get<1>().capacity(), 10u);
    EXPECT_NE(view1.get<2>(), nullptr);
    EXPECT_EQ(view1.get<3>().size(), 10u);
    EXPECT_EQ(view1.get<3>().capacity(), 10u);

    EXPECT_NE(view2.get<0>(), nullptr);
    EXPECT_EQ(view2.get<1>().size(), 0u);
    EXPECT_EQ(view2.get<1>().capacity(), 10u);
    EXPECT_NE(view2.get<2>(), nullptr);
    EXPECT_EQ(view2.get<3>().size(), 0u);
    EXPECT_EQ(view2.get<3>().capacity(), 10u);

    EXPECT_NE(view3.get<0>(), nullptr);
    EXPECT_EQ(view3.get<1>().size(), 10u);
    EXPECT_EQ(view3.get<1>().capacity(), 10u);
    EXPECT_NE(view3.get<2>(), nullptr);
    EXPECT_EQ(view3.get<3>().size(), 10u);
    EXPECT_EQ(view3.get<3>().capacity(), 10u);

    EXPECT_NE(view4.get<0>(), nullptr);
    EXPECT_EQ(view4.get<1>().size(), 0u);
    EXPECT_EQ(view4.get<1>().capacity(), 10u);
    EXPECT_NE(view4.get<2>(), nullptr);
    EXPECT_EQ(view4.get<3>().size(), 0u);
    EXPECT_EQ(view4.get<3>().capacity(), 10u);

    // Create device containers from the views.
    vecmem::testing::simple_container::device device1{view1}, device2{view2};
    vecmem::testing::simple_container::const_device device3{view1},
        device4{view2}, device5{view3}, device6{view4};

    // Set some data on the device containers.
    vecmem::testing::simple_container::count::get(device1) = 10;
    vecmem::testing::simple_container::average::get(device1) = 3.f;
    for (int i = 0; i < 10; ++i) {
        vecmem::testing::simple_container::measurement::get(device1)[i] =
            static_cast<float>(i);
        vecmem::testing::simple_container::index::get(device1)[i] = i;
    }

    vecmem::testing::simple_container::count::get(device2) = 5;
    vecmem::testing::simple_container::average::get(device2) = 6.f;
    for (int i = 0; i < 5; ++i) {
        auto index = device2.push_back_default();
        vecmem::testing::simple_container::measurement::get(device2)[index] =
            2.f * static_cast<float>(i);
        vecmem::testing::simple_container::index::get(device2)[index] = 2 * i;
    }

    // Check the device containers.
    EXPECT_EQ(device1.size(), 10u);
    EXPECT_EQ(device1.capacity(), 10u);
    EXPECT_EQ(vecmem::testing::simple_container::count::get(device1), 10);
    EXPECT_FLOAT_EQ(vecmem::testing::simple_container::average::get(device1),
                    3.f);
    EXPECT_EQ(
        vecmem::testing::simple_container::measurement::get(device1).size(),
        10u);
    EXPECT_EQ(
        vecmem::testing::simple_container::measurement::get(device1).capacity(),
        10u);
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device1).size(),
              10u);
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device1).capacity(),
              10u);
    EXPECT_EQ(device3.size(), 10u);
    EXPECT_EQ(device3.capacity(), 10u);
    EXPECT_EQ(vecmem::testing::simple_container::count::get(device3), 10);
    EXPECT_FLOAT_EQ(vecmem::testing::simple_container::average::get(device3),
                    3.f);
    EXPECT_EQ(
        vecmem::testing::simple_container::measurement::get(device3).size(),
        10u);
    EXPECT_EQ(
        vecmem::testing::simple_container::measurement::get(device3).capacity(),
        10u);
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device3).size(),
              10u);
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device3).capacity(),
              10u);
    EXPECT_EQ(device5.size(), 10u);
    EXPECT_EQ(device5.capacity(), 10u);
    EXPECT_EQ(vecmem::testing::simple_container::count::get(device5), 10);
    EXPECT_FLOAT_EQ(vecmem::testing::simple_container::average::get(device5),
                    3.f);
    EXPECT_EQ(
        vecmem::testing::simple_container::measurement::get(device5).size(),
        10u);
    EXPECT_EQ(
        vecmem::testing::simple_container::measurement::get(device5).capacity(),
        10u);
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device5).size(),
              10u);
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device5).capacity(),
              10u);
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(
            vecmem::testing::simple_container::measurement::get(device1)[i],
            static_cast<float>(i));
        EXPECT_EQ(vecmem::testing::simple_container::index::get(device1)[i], i);
        EXPECT_FLOAT_EQ(
            vecmem::testing::simple_container::measurement::get(device3)[i],
            static_cast<float>(i));
        EXPECT_EQ(vecmem::testing::simple_container::index::get(device3)[i], i);
        EXPECT_FLOAT_EQ(
            vecmem::testing::simple_container::measurement::get(device5)[i],
            static_cast<float>(i));
        EXPECT_EQ(vecmem::testing::simple_container::index::get(device5)[i], i);
    }

    EXPECT_EQ(device2.size(), 5u);
    EXPECT_EQ(device2.capacity(), 10u);
    EXPECT_EQ(vecmem::testing::simple_container::count::get(device2), 5);
    EXPECT_FLOAT_EQ(vecmem::testing::simple_container::average::get(device2),
                    6.f);
    EXPECT_EQ(
        vecmem::testing::simple_container::measurement::get(device2).size(),
        5u);
    EXPECT_EQ(
        vecmem::testing::simple_container::measurement::get(device2).capacity(),
        10u);
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device2).size(),
              5u);
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device2).capacity(),
              10u);
    EXPECT_EQ(device4.size(), 5u);
    EXPECT_EQ(device4.capacity(), 10u);
    EXPECT_EQ(vecmem::testing::simple_container::count::get(device4), 5);
    EXPECT_FLOAT_EQ(vecmem::testing::simple_container::average::get(device4),
                    6.f);
    EXPECT_EQ(
        vecmem::testing::simple_container::measurement::get(device4).size(),
        5u);
    EXPECT_EQ(
        vecmem::testing::simple_container::measurement::get(device4).capacity(),
        10u);
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device4).size(),
              5u);
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device4).capacity(),
              10u);
    EXPECT_EQ(device6.size(), 5u);
    EXPECT_EQ(device6.capacity(), 10u);
    EXPECT_EQ(vecmem::testing::simple_container::count::get(device6), 5);
    EXPECT_FLOAT_EQ(vecmem::testing::simple_container::average::get(device6),
                    6.f);
    EXPECT_EQ(
        vecmem::testing::simple_container::measurement::get(device6).size(),
        5u);
    EXPECT_EQ(
        vecmem::testing::simple_container::measurement::get(device6).capacity(),
        10u);
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device6).size(),
              5u);
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device6).capacity(),
              10u);
    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(
            vecmem::testing::simple_container::measurement::get(device2)[i],
            2.f * static_cast<float>(i));
        EXPECT_EQ(vecmem::testing::simple_container::index::get(device2)[i],
                  2 * i);
        EXPECT_FLOAT_EQ(
            vecmem::testing::simple_container::measurement::get(device4)[i],
            2.f * static_cast<float>(i));
        EXPECT_EQ(vecmem::testing::simple_container::index::get(device4)[i],
                  2 * i);
        EXPECT_FLOAT_EQ(
            vecmem::testing::simple_container::measurement::get(device6)[i],
            2.f * static_cast<float>(i));
        EXPECT_EQ(vecmem::testing::simple_container::index::get(device6)[i],
                  2 * i);
    }
}

TEST_F(core_edm_test, simple_host) {

    // Test the creation of a simple host container.
    vecmem::testing::simple_container::host host1{m_resource};

    // Fill the host container with some data.
    vecmem::testing::simple_container::count::get(host1) = 10;
    vecmem::testing::simple_container::measurement::get(host1).push_back(1.0f);
    vecmem::testing::simple_container::average::get(host1) = 2.f;
    vecmem::testing::simple_container::index::get(host1).push_back(3);

    // Check the contents of the host container.
    const auto& host1c = host1;
    EXPECT_EQ(host1.size(), 1u);
    EXPECT_EQ(vecmem::testing::simple_container::count::get(host1c), 10);
    EXPECT_FLOAT_EQ(vecmem::testing::simple_container::average::get(host1c),
                    2.f);
    EXPECT_EQ(
        vecmem::testing::simple_container::measurement::get(host1c).size(), 1u);
    EXPECT_FLOAT_EQ(
        vecmem::testing::simple_container::measurement::get(host1c)[0], 1.0f);
    EXPECT_EQ(vecmem::testing::simple_container::index::get(host1c).size(), 1u);
    EXPECT_EQ(vecmem::testing::simple_container::index::get(host1c)[0], 3);

    // Check that resizing the container works.
    host1.resize(5);
    EXPECT_EQ(host1.size(), 5u);

    // Make views out of it.
    vecmem::testing::simple_container::view ncview1 = vecmem::get_data(host1);
    vecmem::testing::simple_container::const_view cview1 =
        [](const auto& host) { return vecmem::get_data(host); }(host1);

    // Make trivial checks on the contents of the view(s).
    EXPECT_EQ(ncview1.size(), host1.size());
    EXPECT_EQ(ncview1.capacity(), host1.size());
    EXPECT_EQ(cview1.size(), host1.size());
    EXPECT_EQ(cview1.capacity(), host1.size());
    EXPECT_EQ(ncview1.get<0>(), host1.get<0>().get());
    EXPECT_EQ(cview1.get<1>().size(), host1.get<1>().size());
    EXPECT_EQ(ncview1.get<2>(), host1.get<2>().get());
    EXPECT_EQ(cview1.get<3>().size(), host1.get<3>().size());

    // Make device containers out of the views.
    vecmem::testing::simple_container::device device1{ncview1};
    vecmem::testing::simple_container::const_device device2{ncview1},
        device3{cview1};

    // Check the contents of the device containers.
    EXPECT_EQ(device1.size(), host1.size());
    EXPECT_EQ(device1.capacity(), host1.size());
    EXPECT_EQ(device2.size(), host1.size());
    EXPECT_EQ(device2.capacity(), host1.size());
    EXPECT_EQ(device3.size(), host1.size());
    EXPECT_EQ(device3.capacity(), host1.size());
    EXPECT_EQ(vecmem::testing::simple_container::count::get(device1), 10);
    EXPECT_EQ(vecmem::testing::simple_container::count::get(device2), 10);
    EXPECT_EQ(vecmem::testing::simple_container::count::get(device3), 10);
    EXPECT_FLOAT_EQ(vecmem::testing::simple_container::average::get(device1),
                    2.f);
    EXPECT_FLOAT_EQ(vecmem::testing::simple_container::average::get(device2),
                    2.f);
    EXPECT_FLOAT_EQ(vecmem::testing::simple_container::average::get(device3),
                    2.f);
    EXPECT_EQ(
        vecmem::testing::simple_container::measurement::get(device1).size(),
        vecmem::testing::simple_container::measurement::get(host1).size());
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device1).size(),
              vecmem::testing::simple_container::index::get(host1).size());
    EXPECT_EQ(
        vecmem::testing::simple_container::measurement::get(device1).capacity(),
        vecmem::testing::simple_container::measurement::get(host1).size());
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device1).capacity(),
              vecmem::testing::simple_container::index::get(host1).size());
    EXPECT_EQ(
        vecmem::testing::simple_container::measurement::get(device2).size(),
        vecmem::testing::simple_container::measurement::get(host1).size());
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device2).size(),
              vecmem::testing::simple_container::index::get(host1).size());
    EXPECT_EQ(
        vecmem::testing::simple_container::measurement::get(device2).capacity(),
        vecmem::testing::simple_container::measurement::get(host1).size());
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device2).capacity(),
              vecmem::testing::simple_container::index::get(host1).size());
    EXPECT_EQ(
        vecmem::testing::simple_container::measurement::get(device3).size(),
        vecmem::testing::simple_container::measurement::get(host1).size());
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device3).size(),
              vecmem::testing::simple_container::index::get(host1).size());
    EXPECT_EQ(
        vecmem::testing::simple_container::measurement::get(device3).capacity(),
        vecmem::testing::simple_container::measurement::get(host1).size());
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device3).capacity(),
              vecmem::testing::simple_container::index::get(host1).size());
    EXPECT_FLOAT_EQ(
        vecmem::testing::simple_container::measurement::get(device1)[0], 1.0f);
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device1)[0], 3);
    EXPECT_FLOAT_EQ(
        vecmem::testing::simple_container::measurement::get(device2)[0], 1.0f);
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device2)[0], 3);
    EXPECT_FLOAT_EQ(
        vecmem::testing::simple_container::measurement::get(device3)[0], 1.0f);
    EXPECT_EQ(vecmem::testing::simple_container::index::get(device3)[0], 3);
}
