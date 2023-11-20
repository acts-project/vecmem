/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "../common/edm_jagged_container.hpp"
#include "../common/edm_simple_container.hpp"
#include "vecmem/edm/details/tuple.hpp"
#include "vecmem/edm/details/tuple_traits.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/utils/copy.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <type_traits>

/// Test case for the EDM code
class core_edm_test : public testing::Test {

protected:
    /// Memory resource for the test(s)
    vecmem::host_memory_resource m_resource;
    /// Helper object for the memory copies.
    vecmem::copy m_copy;

};  // class core_edm_test

TEST_F(core_edm_test, tuple) {

    // Construct trivial tuples in a few different ways.
    vecmem::edm::details::tuple<int, float, double> t1;
    vecmem::edm::details::tuple<float, int> t2{2.f, 3};
    vecmem::edm::details::tuple<double, int> t3{
        t2};  // Type mismatch on purpose!

    // Get/set elements in those tuples.
    EXPECT_FLOAT_EQ(vecmem::edm::details::get<0>(t2), 2.f);
    EXPECT_EQ(vecmem::edm::details::get<1>(t2), 3);

    vecmem::edm::details::get<0>(t3) = 4.;
    vecmem::edm::details::get<1>(t3) = 6;
    EXPECT_DOUBLE_EQ(vecmem::edm::details::get<0>(t3), 4.f);
    EXPECT_EQ(vecmem::edm::details::get<1>(t3), 6);

    // Exercise vecmem::edm::details::tie(...).
    int value1 = 0;
    float value2 = 1.f;
    double value3 = 2.;
    auto t4 = vecmem::edm::details::tie(value1, value2, value3);
    EXPECT_EQ(vecmem::edm::details::get<0>(t4), 0);
    EXPECT_FLOAT_EQ(vecmem::edm::details::get<1>(t4), 1.f);
    EXPECT_DOUBLE_EQ(vecmem::edm::details::get<2>(t4), 2.);

    // Exercise vecmem::edm::details::tuple_element.
    static constexpr bool type_check1 = std::is_same_v<
        vecmem::edm::details::tuple_element<
            1, vecmem::edm::details::tuple<int, float, double>>::type,
        float>;
    EXPECT_TRUE(type_check1);
    static constexpr bool type_check2 =
        std::is_same_v<vecmem::edm::details::tuple_element_t<
                           2, vecmem::edm::details::tuple<int, float, double>>,
                       double>;
    EXPECT_TRUE(type_check2);

    // Exercise vecmem::edm::details::make_tuple(...).
    auto t5 = vecmem::edm::details::make_tuple(1, 2u, 3.f, 4.);
    EXPECT_EQ(vecmem::edm::details::get<0>(t5), 1);
    EXPECT_EQ(vecmem::edm::details::get<1>(t5), 2);
    EXPECT_FLOAT_EQ(vecmem::edm::details::get<2>(t5), 3.f);
    EXPECT_DOUBLE_EQ(vecmem::edm::details::get<3>(t5), 4.);
    static constexpr bool type_check3 =
        std::is_same_v<vecmem::edm::details::tuple_element_t<0, decltype(t5)>,
                       int>;
    EXPECT_TRUE(type_check3);
    static constexpr bool type_check4 =
        std::is_same_v<vecmem::edm::details::tuple_element_t<1, decltype(t5)>,
                       unsigned int>;
    EXPECT_TRUE(type_check4);
    static constexpr bool type_check5 =
        std::is_same_v<vecmem::edm::details::tuple_element_t<2, decltype(t5)>,
                       float>;
    EXPECT_TRUE(type_check5);
    static constexpr bool type_check6 =
        std::is_same_v<vecmem::edm::details::tuple_element_t<3, decltype(t5)>,
                       double>;
    EXPECT_TRUE(type_check6);
}

TEST_F(core_edm_test, simple_view) {

    // Test the creation of simple views.
    vecmem::testing::simple_container::view view1;
    vecmem::testing::simple_container::const_view view2{view1};
}

TEST_F(core_edm_test, simple_buffer) {

    // The capacity of the tested buffer(s).
    static constexpr unsigned int capacity = 10;

    // Test the creation of fixed sized and resizable, at the same time "simple"
    // buffers.
    vecmem::testing::simple_container::buffer buffer1(
        capacity, m_resource, vecmem::data::buffer_type::fixed_size);
    vecmem::testing::simple_container::buffer buffer2(
        capacity, m_resource, vecmem::data::buffer_type::resizable);
    m_copy.setup(buffer1);
    m_copy.setup(buffer2);

    // "Create" views off of the buffers.
    vecmem::testing::simple_container::view view1{buffer1}, view2{buffer2};
    vecmem::testing::simple_container::const_view view3{buffer1},
        view4{buffer2};

    // Helper lambdas for checking the contents of the views.
    auto check_scalar_view = [](const auto& view) { EXPECT_NE(view, nullptr); };
    auto check_fixed_vector_view = [](const auto& view) {
        EXPECT_EQ(view.size(), capacity);
        EXPECT_EQ(view.capacity(), capacity);
        EXPECT_EQ(view.size_ptr(), nullptr);
        if (view.size() > 0) {
            EXPECT_NE(view.ptr(), nullptr);
        }
    };
    auto check_resizable_vector_view = [](const auto& view) {
        EXPECT_EQ(view.size(), 0u);
        EXPECT_EQ(view.capacity(), capacity);
        EXPECT_NE(view.size_ptr(), nullptr);
        if (view.capacity() > 0) {
            EXPECT_NE(view.ptr(), nullptr);
        }
    };
    auto check_fixed_view = [&](const auto& view) {
        check_scalar_view(view.template get<0>());
        check_fixed_vector_view(view.template get<1>());
        check_scalar_view(view.template get<2>());
        check_fixed_vector_view(view.template get<3>());
    };
    auto check_resizable_view = [&](const auto& view) {
        check_scalar_view(view.template get<0>());
        check_resizable_vector_view(view.template get<1>());
        check_scalar_view(view.template get<2>());
        check_resizable_vector_view(view.template get<3>());
    };

    // Check the views.
    check_fixed_view(view1);
    check_resizable_view(view2);
    check_fixed_view(view3);
    check_resizable_view(view4);

    // The size of the resizable vector(s).
    static constexpr unsigned int size = 5;

    // Create device containers from the views.
    vecmem::testing::simple_container::device device1{view1}, device2{view2};
    vecmem::testing::simple_container::const_device device3{view1},
        device4{view2}, device5{view3}, device6{view4};

    // Set some data on the device containers.
    vecmem::testing::simple_container::count::get(device1) = 10;
    vecmem::testing::simple_container::average::get(device1) = 3.f;
    for (unsigned int i = 0; i < capacity; ++i) {
        vecmem::testing::simple_container::measurement::get(device1)[i] =
            static_cast<float>(i);
        vecmem::testing::simple_container::index::get(device1)[i] =
            static_cast<int>(i);
    }

    vecmem::testing::simple_container::count::get(device2) = 5;
    vecmem::testing::simple_container::average::get(device2) = 6.f;
    for (unsigned int i = 0; i < size; ++i) {
        auto index = device2.push_back_default();
        vecmem::testing::simple_container::measurement::get(device2)[index] =
            2.f * static_cast<float>(i);
        vecmem::testing::simple_container::index::get(device2)[index] =
            2 * static_cast<int>(i);
    }

    // Helper lambdas for checking the device containers.
    auto check_fixed_device_scalars = [](const auto& device) {
        EXPECT_EQ(vecmem::testing::simple_container::count::get(device), 10);
        EXPECT_FLOAT_EQ(vecmem::testing::simple_container::average::get(device),
                        3.f);
    };
    auto check_fixed_device_vectors = [](const auto& device) {
        EXPECT_EQ(
            vecmem::testing::simple_container::measurement::get(device).size(),
            capacity);
        EXPECT_EQ(vecmem::testing::simple_container::measurement::get(device)
                      .capacity(),
                  capacity);
        EXPECT_EQ(vecmem::testing::simple_container::index::get(device).size(),
                  capacity);
        EXPECT_EQ(
            vecmem::testing::simple_container::index::get(device).capacity(),
            capacity);
        for (unsigned int i = 0; i < capacity; ++i) {
            EXPECT_FLOAT_EQ(
                vecmem::testing::simple_container::measurement::get(device)[i],
                static_cast<float>(i));
            EXPECT_EQ(vecmem::testing::simple_container::index::get(device)[i],
                      static_cast<int>(i));
        }
    };
    auto check_resizable_device_scalars = [](const auto& device) {
        EXPECT_EQ(vecmem::testing::simple_container::count::get(device), 5);
        EXPECT_FLOAT_EQ(vecmem::testing::simple_container::average::get(device),
                        6.f);
    };
    auto check_resizable_device_vectors = [](const auto& device) {
        EXPECT_EQ(
            vecmem::testing::simple_container::measurement::get(device).size(),
            size);
        EXPECT_EQ(vecmem::testing::simple_container::measurement::get(device)
                      .capacity(),
                  capacity);
        EXPECT_EQ(vecmem::testing::simple_container::index::get(device).size(),
                  size);
        EXPECT_EQ(
            vecmem::testing::simple_container::index::get(device).capacity(),
            capacity);
        for (unsigned int i = 0; i < size; ++i) {
            EXPECT_FLOAT_EQ(
                vecmem::testing::simple_container::measurement::get(device)[i],
                2.f * static_cast<float>(i));
            EXPECT_EQ(vecmem::testing::simple_container::index::get(device)[i],
                      2 * static_cast<int>(i));
        }
    };
    auto check_fixed_device = [&](const auto& device) {
        EXPECT_EQ(device.size(), capacity);
        EXPECT_EQ(device.capacity(), capacity);
        check_fixed_device_scalars(device);
        check_fixed_device_vectors(device);
    };
    auto check_resizable_device = [&](const auto& device) {
        EXPECT_EQ(device.size(), size);
        EXPECT_EQ(device.capacity(), capacity);
        check_resizable_device_scalars(device);
        check_resizable_device_vectors(device);
    };

    // Check the device containers.
    check_fixed_device(device1);
    check_resizable_device(device2);
    check_fixed_device(device3);
    check_resizable_device(device4);
    check_fixed_device(device5);
    check_resizable_device(device6);
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
    EXPECT_EQ(ncview1.get<0>(), &(host1.get<0>()));
    EXPECT_EQ(cview1.get<1>().size(), host1.get<1>().size());
    EXPECT_EQ(ncview1.get<2>(), &(host1.get<2>()));
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

TEST_F(core_edm_test, jagged_view) {

    // Test the creation of jagged views.
    vecmem::testing::jagged_container::view view1;
    vecmem::testing::jagged_container::const_view view2{view1};
}

TEST_F(core_edm_test, jagged_data) {

    // Test the creation of jagged data objects.
    vecmem::testing::jagged_container::data data1{10, m_resource};
    vecmem::testing::jagged_container::view view1{data1};
    vecmem::testing::jagged_container::const_view view2{view1};
}

TEST_F(core_edm_test, jagged_buffer) {

    // Test the creation of fixed sized and resizable, at the same time "simple"
    // buffers.
    std::vector<std::size_t> capacities = {0, 2, 5, 0, 8};
    vecmem::testing::jagged_container::buffer buffer1(
        capacities, m_resource, nullptr, vecmem::data::buffer_type::fixed_size);
    vecmem::testing::jagged_container::buffer buffer2(
        capacities, m_resource, nullptr, vecmem::data::buffer_type::resizable);
    m_copy.setup(buffer1);
    m_copy.setup(buffer2);

    // "Create" views off of the buffers.
    vecmem::testing::jagged_container::view view1{buffer1}, view2{buffer2};
    vecmem::testing::jagged_container::const_view view3{buffer1},
        view4{buffer2};

    // Helper lambdas for checking the contents of the views.
    auto check_scalar = [](const auto& view) { EXPECT_NE(view, nullptr); };
    auto check_fixed_vector = [&capacities](const auto& view) {
        EXPECT_EQ(view.size(), capacities.size());
        EXPECT_EQ(view.capacity(), capacities.size());
        EXPECT_EQ(view.size_ptr(), nullptr);
        if (view.size() > 0) {
            EXPECT_NE(view.ptr(), nullptr);
        }
    };
    auto check_fixed_jagged_vector = [&capacities](const auto& view) {
        EXPECT_EQ(view.size(), capacities.size());
        EXPECT_EQ(view.ptr(), view.host_ptr());
        for (std::size_t i = 0; i < capacities.size(); ++i) {
            EXPECT_EQ(view.host_ptr()[i].size(), capacities[i]);
            EXPECT_EQ(view.host_ptr()[i].capacity(), capacities[i]);
            EXPECT_EQ(view.host_ptr()[i].size_ptr(), nullptr);
            if (view.host_ptr()[i].size() > 0) {
                EXPECT_NE(view.host_ptr()[i].ptr(), nullptr);
            }
        }
    };
    auto check_resizable_jagged_vector = [&capacities](const auto& view) {
        EXPECT_EQ(view.size(), capacities.size());
        EXPECT_EQ(view.ptr(), view.host_ptr());
        for (std::size_t i = 0; i < capacities.size(); ++i) {
            EXPECT_EQ(view.host_ptr()[i].size(), 0u);
            EXPECT_EQ(view.host_ptr()[i].capacity(), capacities[i]);
            EXPECT_NE(view.host_ptr()[i].size_ptr(), nullptr);
            if (view.host_ptr()[i].capacity() > 0) {
                EXPECT_NE(view.host_ptr()[i].ptr(), nullptr);
            }
        }
    };
    auto check_fixed_view = [&](const auto& view) {
        check_scalar(view.template get<0>());
        check_fixed_vector(view.template get<1>());
        check_fixed_jagged_vector(view.template get<2>());
        check_scalar(view.template get<3>());
        check_fixed_jagged_vector(view.template get<4>());
        check_fixed_vector(view.template get<5>());
    };
    auto check_resizable_view = [&](const auto& view) {
        check_scalar(view.template get<0>());
        check_fixed_vector(view.template get<1>());  // The 1D vectors are fixed
        check_resizable_jagged_vector(view.template get<2>());
        check_scalar(view.template get<3>());
        check_resizable_jagged_vector(view.template get<4>());
        check_fixed_vector(view.template get<5>());  // The 1D vectors are fixed
    };

    // Check the views.
    check_fixed_view(view1);
    check_resizable_view(view2);
    check_fixed_view(view3);
    check_resizable_view(view4);

    // Create device containers from the views.
    vecmem::testing::jagged_container::device device1{view1}, device2{view2};
    vecmem::testing::jagged_container::const_device device3{view1},
        device4{view2}, device5{view3}, device6{view4};
}

TEST_F(core_edm_test, jagged_host) {

    // Test the creation of a simple host container.
    vecmem::testing::jagged_container::host host1{m_resource};

    // Fill the host container with some data.
    vecmem::testing::jagged_container::count::get(host1) = 10;
    vecmem::testing::jagged_container::average::get(host1) = 2.34f;
    host1.resize(2);
    vecmem::testing::jagged_container::measurement::get(host1)[0] = 1.0f;
    vecmem::testing::jagged_container::measurement::get(host1)[1] = 2.0f;
    vecmem::testing::jagged_container::index::get(host1)[0] = 3;
    vecmem::testing::jagged_container::index::get(host1)[1] = 4;
    vecmem::testing::jagged_container::measurements::get(host1)[0].push_back(
        1.1);
    vecmem::testing::jagged_container::measurements::get(host1)[1].push_back(
        2.1);
    vecmem::testing::jagged_container::indices::get(host1)[0].push_back(31);
    vecmem::testing::jagged_container::indices::get(host1)[1].push_back(41);

    // Check the contents of the host container.
    const auto& host1c = host1;
    EXPECT_EQ(host1.size(), 2u);
    EXPECT_EQ(vecmem::testing::jagged_container::count::get(host1c), 10);
    EXPECT_FLOAT_EQ(vecmem::testing::jagged_container::average::get(host1),
                    2.34f);
    EXPECT_EQ(
        vecmem::testing::jagged_container::measurement::get(host1c).size(), 2u);
    EXPECT_FLOAT_EQ(
        vecmem::testing::jagged_container::measurement::get(host1)[0], 1.0f);
    EXPECT_FLOAT_EQ(
        vecmem::testing::jagged_container::measurement::get(host1c)[1], 2.0f);
    EXPECT_EQ(vecmem::testing::jagged_container::index::get(host1c).size(), 2u);
    EXPECT_EQ(vecmem::testing::jagged_container::index::get(host1)[0], 3);
    EXPECT_EQ(vecmem::testing::jagged_container::index::get(host1c)[1], 4);
    EXPECT_EQ(
        vecmem::testing::jagged_container::measurements::get(host1)[0].size(),
        1u);
    EXPECT_EQ(
        vecmem::testing::jagged_container::measurements::get(host1c)[1].size(),
        1u);
    EXPECT_DOUBLE_EQ(
        vecmem::testing::jagged_container::measurements::get(host1)[0][0], 1.1);
    EXPECT_DOUBLE_EQ(
        vecmem::testing::jagged_container::measurements::get(host1c)[1][0],
        2.1);
    EXPECT_EQ(vecmem::testing::jagged_container::indices::get(host1)[0].size(),
              1u);
    EXPECT_EQ(vecmem::testing::jagged_container::indices::get(host1c)[1].size(),
              1u);
    EXPECT_EQ(vecmem::testing::jagged_container::indices::get(host1)[0][0], 31);
    EXPECT_EQ(vecmem::testing::jagged_container::indices::get(host1c)[1][0],
              41);

    // Make views out of it.
    vecmem::testing::jagged_container::data data1 = vecmem::get_data(host1);
    vecmem::testing::jagged_container::view ncview1 = data1;
    vecmem::testing::jagged_container::const_view cview1 = data1;

    // Make trivial checks on the contents of the view(s).
    EXPECT_EQ(ncview1.size(), host1.size());
    EXPECT_EQ(ncview1.capacity(), host1.size());
    EXPECT_EQ(cview1.size(), host1.size());
    EXPECT_EQ(cview1.capacity(), host1.size());
    EXPECT_EQ(ncview1.get<0>(), &(host1.get<0>()));
    EXPECT_EQ(cview1.get<1>().size(), host1.get<1>().size());
    EXPECT_EQ(cview1.get<2>().size(), host1.get<2>().size());
    EXPECT_EQ(ncview1.get<3>(), &(host1.get<3>()));
    EXPECT_EQ(ncview1.get<4>().size(), host1.get<4>().size());
    EXPECT_EQ(cview1.get<5>().size(), host1.get<5>().size());

    // Make device containers out of the views.
    vecmem::testing::jagged_container::device device1{ncview1};
    vecmem::testing::jagged_container::const_device device2{ncview1},
        device3{cview1};

    // Check the contents of the device containers.
    EXPECT_EQ(device1.size(), host1.size());
    EXPECT_EQ(device1.capacity(), host1.size());
    EXPECT_EQ(device2.size(), host1.size());
    EXPECT_EQ(device2.capacity(), host1.size());
    EXPECT_EQ(device3.size(), host1.size());
    EXPECT_EQ(device3.capacity(), host1.size());

    EXPECT_EQ(vecmem::testing::jagged_container::count::get(device1), 10);
    EXPECT_EQ(vecmem::testing::jagged_container::count::get(device2), 10);
    EXPECT_EQ(vecmem::testing::jagged_container::count::get(device3), 10);

    EXPECT_FLOAT_EQ(vecmem::testing::jagged_container::average::get(device1),
                    2.34f);
    EXPECT_FLOAT_EQ(vecmem::testing::jagged_container::average::get(device2),
                    2.34f);
    EXPECT_FLOAT_EQ(vecmem::testing::jagged_container::average::get(device3),
                    2.34f);

    EXPECT_EQ(
        vecmem::testing::jagged_container::measurement::get(device1).size(),
        vecmem::testing::jagged_container::measurement::get(host1).size());
    EXPECT_EQ(vecmem::testing::jagged_container::index::get(device1).size(),
              vecmem::testing::jagged_container::index::get(host1).size());
    EXPECT_EQ(
        vecmem::testing::jagged_container::measurements::get(device1).size(),
        vecmem::testing::jagged_container::measurements::get(host1).size());
    EXPECT_EQ(vecmem::testing::jagged_container::indices::get(device1).size(),
              vecmem::testing::jagged_container::indices::get(host1).size());
    EXPECT_EQ(
        vecmem::testing::jagged_container::measurement::get(device2).size(),
        vecmem::testing::jagged_container::measurement::get(host1).size());
    EXPECT_EQ(vecmem::testing::jagged_container::index::get(device2).size(),
              vecmem::testing::jagged_container::index::get(host1).size());
    EXPECT_EQ(
        vecmem::testing::jagged_container::measurements::get(device2).size(),
        vecmem::testing::jagged_container::measurements::get(host1).size());
    EXPECT_EQ(vecmem::testing::jagged_container::indices::get(device2).size(),
              vecmem::testing::jagged_container::indices::get(host1).size());
    EXPECT_EQ(
        vecmem::testing::jagged_container::measurement::get(device3).size(),
        vecmem::testing::jagged_container::measurement::get(host1).size());
    EXPECT_EQ(vecmem::testing::jagged_container::index::get(device3).size(),
              vecmem::testing::jagged_container::index::get(host1).size());
    EXPECT_EQ(
        vecmem::testing::jagged_container::measurements::get(device3).size(),
        vecmem::testing::jagged_container::measurements::get(host1).size());
    EXPECT_EQ(vecmem::testing::jagged_container::indices::get(device3).size(),
              vecmem::testing::jagged_container::indices::get(host1).size());

    EXPECT_FLOAT_EQ(
        vecmem::testing::jagged_container::measurement::get(device1)[0], 1.0f);
    EXPECT_FLOAT_EQ(
        vecmem::testing::jagged_container::measurement::get(device1)[1], 2.0f);
    EXPECT_FLOAT_EQ(
        vecmem::testing::jagged_container::measurement::get(device2)[0], 1.0f);
    EXPECT_FLOAT_EQ(
        vecmem::testing::jagged_container::measurement::get(device2)[1], 2.0f);
    EXPECT_FLOAT_EQ(
        vecmem::testing::jagged_container::measurement::get(device3)[0], 1.0f);
    EXPECT_FLOAT_EQ(
        vecmem::testing::jagged_container::measurement::get(device3)[1], 2.0f);

    EXPECT_EQ(vecmem::testing::jagged_container::index::get(device1)[0], 3);
    EXPECT_EQ(vecmem::testing::jagged_container::index::get(device1)[1], 4);
    EXPECT_EQ(vecmem::testing::jagged_container::index::get(device2)[0], 3);
    EXPECT_EQ(vecmem::testing::jagged_container::index::get(device2)[1], 4);
    EXPECT_EQ(vecmem::testing::jagged_container::index::get(device3)[0], 3);
    EXPECT_EQ(vecmem::testing::jagged_container::index::get(device3)[1], 4);

    EXPECT_EQ(
        vecmem::testing::jagged_container::measurements::get(device1)[0].size(),
        1u);
    EXPECT_EQ(
        vecmem::testing::jagged_container::measurements::get(device1)[1].size(),
        1u);
    EXPECT_EQ(
        vecmem::testing::jagged_container::measurements::get(device2)[0].size(),
        1u);
    EXPECT_EQ(
        vecmem::testing::jagged_container::measurements::get(device2)[1].size(),
        1u);
    EXPECT_EQ(
        vecmem::testing::jagged_container::measurements::get(device3)[0].size(),
        1u);
    EXPECT_EQ(
        vecmem::testing::jagged_container::measurements::get(device3)[1].size(),
        1u);

    EXPECT_DOUBLE_EQ(
        vecmem::testing::jagged_container::measurements::get(device1)[0][0],
        1.1);
    EXPECT_DOUBLE_EQ(
        vecmem::testing::jagged_container::measurements::get(device1)[1][0],
        2.1);
    EXPECT_DOUBLE_EQ(
        vecmem::testing::jagged_container::measurements::get(device2)[0][0],
        1.1);
    EXPECT_DOUBLE_EQ(
        vecmem::testing::jagged_container::measurements::get(device2)[1][0],
        2.1);
    EXPECT_DOUBLE_EQ(
        vecmem::testing::jagged_container::measurements::get(device3)[0][0],
        1.1);
    EXPECT_DOUBLE_EQ(
        vecmem::testing::jagged_container::measurements::get(device3)[1][0],
        2.1);

    EXPECT_EQ(
        vecmem::testing::jagged_container::indices::get(device1)[0].size(), 1u);
    EXPECT_EQ(
        vecmem::testing::jagged_container::indices::get(device1)[1].size(), 1u);
    EXPECT_EQ(
        vecmem::testing::jagged_container::indices::get(device2)[0].size(), 1u);
    EXPECT_EQ(
        vecmem::testing::jagged_container::indices::get(device2)[1].size(), 1u);
    EXPECT_EQ(
        vecmem::testing::jagged_container::indices::get(device3)[0].size(), 1u);
    EXPECT_EQ(
        vecmem::testing::jagged_container::indices::get(device3)[1].size(), 1u);

    EXPECT_EQ(vecmem::testing::jagged_container::indices::get(device1)[0][0],
              31);
    EXPECT_EQ(vecmem::testing::jagged_container::indices::get(device1)[1][0],
              41);
    EXPECT_EQ(vecmem::testing::jagged_container::indices::get(device2)[0][0],
              31);
    EXPECT_EQ(vecmem::testing::jagged_container::indices::get(device2)[1][0],
              41);
    EXPECT_EQ(vecmem::testing::jagged_container::indices::get(device3)[0][0],
              31);
    EXPECT_EQ(vecmem::testing::jagged_container::indices::get(device3)[1][0],
              41);
}
