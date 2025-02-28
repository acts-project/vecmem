/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/edm/device.hpp"
#include "vecmem/edm/host.hpp"
#include "vecmem/memory/host_memory_resource.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

/// Test case for @c vecmem::edm::host.
class core_edm_host_test : public testing::Test {

protected:
    /// Schema for the tests.
    using schema =
        vecmem::edm::schema<vecmem::edm::type::scalar<int>,
                            vecmem::edm::type::vector<float>,
                            vecmem::edm::type::jagged_vector<double>>;
    /// Constant schema used for the test.
    using const_schema = vecmem::edm::details::add_const_t<schema>;
    /// Interface for the test container.
    template <typename BASE>
    struct interface : public BASE {
        using BASE::BASE;
        using BASE::operator=;
        auto& scalar() { return BASE::template get<0>(); }
        const auto& scalar() const { return BASE::template get<0>(); }
        auto& vector() { return BASE::template get<1>(); }
        const auto& vector() const { return BASE::template get<1>(); }
        auto& jagged_vector() { return BASE::template get<2>(); }
        const auto& jagged_vector() const { return BASE::template get<2>(); }
    };
    /// Host container type used in the test.
    using host_type = interface<vecmem::edm::host<schema, interface>>;

    /// Create a host container object with some non-trivial content.
    host_type create() {

        // Create a host container, and fill it.
        host_type host{m_resource};
        static constexpr std::size_t SIZE = 5;
        for (std::size_t i = 0; i < SIZE; ++i) {
            vecmem::vector<double> v{&m_resource};
            for (std::size_t j = 0; j < i + 1; ++j) {
                v.push_back(3. + static_cast<double>(i + j));
            }
            host.push_back(host_type::object_type{1, 2.f + 1, v});
        }
        // Give it to the caller.
        return host;
    }

    /// Memory resource for the test(s)
    vecmem::host_memory_resource m_resource;

};  // class core_edm_host_test

TEST_F(core_edm_host_test, construct_assign) {

    // Construct a host container, and make sure that it looks okay.
    host_type host1 = create();
    EXPECT_EQ(host1.size(), host1.get<1>().size());
    EXPECT_EQ(host1.size(), host1.get<2>().size());

    // Lambda comparing two host containers.
    auto compare = [](const host_type& h1, const host_type& h2) {
        EXPECT_EQ(h1.size(), h2.size());
        EXPECT_EQ(h1.get<0>(), h2.get<0>());
        EXPECT_EQ(h1.get<1>(), h2.get<1>());
        EXPECT_EQ(h1.get<2>(), h2.get<2>());
    };

    // Construct a copy of it, and check that it is the same.
    host_type host2{host1};
    compare(host1, host2);

    // Create a new host container, and assign the first one to it. Check that
    // this also works as it should.
    host_type host3{m_resource};
    host3 = host1;
    compare(host1, host3);
}

TEST_F(core_edm_host_test, get_data) {

    // Construct a host container.
    host_type host1 = create();

    // Lambda checking a data object against a host container.
    auto compare = [](const auto& data,
                      const vecmem::edm::host<schema, interface>& host) {
        EXPECT_EQ(data.capacity(), host.size());
        EXPECT_EQ(data.template get<0>(), &(host.get<0>()));
        EXPECT_EQ(data.template get<1>().size(), host.size());
        EXPECT_EQ(data.template get<1>().capacity(), host.size());
        EXPECT_EQ(data.template get<1>().ptr(), host.get<1>().data());
        ASSERT_EQ(data.template get<2>().size(), host.size());
        EXPECT_EQ(data.template get<2>().capacity(), host.size());
        for (std::size_t i = 0; i < host.size(); ++i) {
            EXPECT_EQ(data.template get<2>().host_ptr(),
                      data.template get<2>().ptr());
            EXPECT_EQ(data.template get<2>().host_ptr()[i].size(),
                      host.get<2>()[i].size());
            EXPECT_EQ(data.template get<2>().host_ptr()[i].capacity(),
                      host.get<2>()[i].size());
            EXPECT_EQ(data.template get<2>().host_ptr()[i].ptr(),
                      host.get<2>()[i].data());
        }
    };

    // Get a non-const data object for it, and check its contents.
    vecmem::edm::data<schema> data1 = vecmem::get_data(host1, &m_resource);
    compare(data1, host1);

    // Get a const data object for it, and check its contents.
    vecmem::edm::data<const_schema> data2 =
        [](const vecmem::edm::host<schema, interface>& host) {
            return vecmem::get_data(host);
        }(host1);
    compare(data2, host1);
}

TEST_F(core_edm_host_test, device) {

    // Construct a host container.
    host_type host1 = create();

    // Lambda comparing the contents of a host and a device container.
    auto compare = [](const auto& device,
                      const vecmem::edm::host<schema, interface>& host) {
        ASSERT_EQ(device.size(), host.size());
        EXPECT_EQ(device.template get<0>(), host.get<0>());
        for (vecmem::edm::details::size_type i = 0; i < device.size(); ++i) {
            EXPECT_EQ(device.template get<1>()[i], host.get<1>()[i]);
            ASSERT_EQ(device.template get<2>()[i].size(),
                      host.get<2>()[i].size());
            for (vecmem::edm::details::size_type j = 0;
                 j < device.template get<2>()[i].size(); ++j) {
                EXPECT_EQ(device.template get<2>()[i][j], host.get<2>()[i][j]);
            }
        }
    };

    // Create a non-const device object for it, and check its contents.
    auto data1 = vecmem::get_data(host1);
    vecmem::edm::device<schema, interface> device1{data1};
    compare(device1, host1);

    // Create constant device objects for it, and check their contents.
    auto data2 = [](const vecmem::edm::host<schema, interface>& host) {
        return vecmem::get_data(host);
    }(host1);
    vecmem::edm::device<const_schema, interface> device2{data2};
    compare(device2, host1);
    vecmem::edm::device<const_schema, interface> device3{data1};
    compare(device3, host1);
}

TEST_F(core_edm_host_test, proxy) {

    // Construct a host container.
    host_type host = create();

    // Compare the contents retrieved through the proxy interface, with the
    // contents retrieved through its container interface.
    for (std::size_t i = 0; i < host.size(); ++i) {
        EXPECT_EQ(host.at(i).scalar(), host.scalar());
        EXPECT_EQ(host[i].scalar(), host.scalar());
        EXPECT_FLOAT_EQ(host.at(i).vector(), host.vector().at(i));
        EXPECT_FLOAT_EQ(host[i].vector(), host.vector()[i]);
        ASSERT_EQ(host.at(i).jagged_vector().size(),
                  host.jagged_vector().at(i).size());
        ASSERT_EQ(host[i].jagged_vector().size(),
                  host.jagged_vector()[i].size());
        for (std::size_t j = 0; j < host.jagged_vector().at(i).size(); ++j) {
            EXPECT_DOUBLE_EQ(host.at(i).jagged_vector().at(j),
                             host.jagged_vector().at(i).at(j));
            EXPECT_DOUBLE_EQ(host[i].jagged_vector()[j],
                             host.jagged_vector()[i][j]);
        }
    }
}

TEST_F(core_edm_host_test, const_proxy) {

    // Construct a host container.
    host_type nchost = create();
    const host_type& host = nchost;

    // Compare the contents retrieved through the proxy interface, with the
    // contents retrieved through its container interface.
    for (std::size_t i = 0; i < host.size(); ++i) {
        EXPECT_EQ(host.at(i).scalar(), host.scalar());
        EXPECT_EQ(host[i].scalar(), host.scalar());
        EXPECT_FLOAT_EQ(host.at(i).vector(), host.vector().at(i));
        EXPECT_FLOAT_EQ(host[i].vector(), host.vector()[i]);
        ASSERT_EQ(host.at(i).jagged_vector().size(),
                  host.jagged_vector().at(i).size());
        ASSERT_EQ(host[i].jagged_vector().size(),
                  host.jagged_vector()[i].size());
        for (std::size_t j = 0; j < host.jagged_vector().at(i).size(); ++j) {
            EXPECT_DOUBLE_EQ(host.at(i).jagged_vector().at(j),
                             host.jagged_vector().at(i).at(j));
            EXPECT_DOUBLE_EQ(host[i].jagged_vector()[j],
                             host.jagged_vector()[i][j]);
        }
    }
}

TEST_F(core_edm_host_test, proxy_assign) {

    // Helper lambda for comparing two containers.
    auto compare = [](const host_type& h1, const host_type& h2) {
        EXPECT_EQ(h1.size(), h2.size());
        for (std::size_t i = 0; i < h1.size(); ++i) {
            EXPECT_EQ(h1.at(i).scalar(), h2.at(i).scalar());
            EXPECT_FLOAT_EQ(h1.at(i).vector(), h2.at(i).vector());
            ASSERT_EQ(h1.at(i).jagged_vector().size(),
                      h2.at(i).jagged_vector().size());
            for (std::size_t j = 0; j < h1.at(i).jagged_vector().size(); ++j) {
                EXPECT_DOUBLE_EQ(h1.at(i).jagged_vector().at(j),
                                 h2.at(i).jagged_vector().at(j));
            }
        }
    };

    // Construct the host containers.
    host_type orig = create();

    // Make a copy using push_back-s.
    host_type copy1{m_resource};
    for (std::size_t i = 0; i < orig.size(); ++i) {
        copy1.push_back(orig.at(i));
    }
    compare(orig, copy1);

    // Make a copy using assignment.
    host_type copy2{m_resource};
    copy2.resize(orig.size());
    for (std::size_t i = 0; i < orig.size(); ++i) {
        copy2.at(i) = orig.at(i);
    }
    compare(orig, copy2);
}

namespace {
using simple_schema = vecmem::edm::schema<vecmem::edm::type::scalar<int>,
                                          vecmem::edm::type::vector<float>>;
template <typename BASE>
struct simple_interface : public BASE {
    using BASE::BASE;
    using BASE::operator=;
    auto& scalar() { return BASE::template get<0>(); }
    const auto& scalar() const { return BASE::template get<0>(); }
    auto& vector() { return BASE::template get<1>(); }
    const auto& vector() const { return BASE::template get<1>(); }
};
using simple_host_type =
    simple_interface<vecmem::edm::host<simple_schema, simple_interface>>;
}  // namespace

TEST_F(core_edm_host_test, push_back_simple) {

    // Set up a simple type, without any jagged vectors.
    simple_host_type host{m_resource};

    // Add a couple of elements to it. Since simple_interface has an explicit
    // constructor, the formalism here is fairly simple.
    host.push_back({1, 2.f});
    host.push_back({2, 3.f});

    // Check that the contents are as expected.
    ASSERT_EQ(host.size(), 2);
    EXPECT_EQ(host.at(0).scalar(), 2);
    EXPECT_EQ(host.at(1).scalar(), 2);
    EXPECT_FLOAT_EQ(host.at(0).vector(), 2.f);
    EXPECT_FLOAT_EQ(host.at(1).vector(), 3.f);
}

TEST_F(core_edm_host_test, push_back_jagged) {

    // Create an empty host container.
    host_type host{m_resource};

    // Add a couple of elements to it. Since interface has no constructor of
    // its own, because with jagged vectors that's not so easy to do, one needs
    // to use double braces in the following expression.
    host.push_back({1, 2.f, vecmem::vector<double>{3., 4., 5.}});
    host.push_back({2, 3.f, vecmem::vector<double>{4., 5., 6.}});

    // Check that the contents are as expected.
    ASSERT_EQ(host.size(), 2);
    EXPECT_EQ(host.at(0).scalar(), 2);
    EXPECT_EQ(host.at(1).scalar(), 2);
    EXPECT_FLOAT_EQ(host.at(0).vector(), 2.f);
    EXPECT_FLOAT_EQ(host.at(1).vector(), 3.f);
    ASSERT_EQ(host.at(0).jagged_vector().size(), 3);
    ASSERT_EQ(host.at(1).jagged_vector().size(), 3);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(host.at(0).jagged_vector().at(i),
                         3. + static_cast<double>(i));
        EXPECT_DOUBLE_EQ(host.at(1).jagged_vector().at(i),
                         4. + static_cast<double>(i));
    }

    // Construct a filled host container.
    host_type host2 = create();
    host_type host2_ref = host2;

    // Add elements to it from the previous container.
    for (std::size_t i = 0; i < host.size(); ++i) {
        host2.push_back(host.at(i));
    }

    // Check that the contents are as expected.
    ASSERT_EQ(host2.size(), 7);
    for (std::size_t i = 0; i < host2.size(); ++i) {
        const auto test_proxy = host2.at(i);
        const auto ref_proxy = (i < host2_ref.size())
                                   ? host2_ref.at(i)
                                   : host.at(i - host2_ref.size());
        EXPECT_EQ(test_proxy.scalar(), host.at(0).scalar());
        EXPECT_FLOAT_EQ(test_proxy.vector(), ref_proxy.vector());
        ASSERT_EQ(test_proxy.jagged_vector().size(),
                  ref_proxy.jagged_vector().size());
        for (std::size_t j = 0; j < test_proxy.jagged_vector().size(); ++j) {
            EXPECT_DOUBLE_EQ(test_proxy.jagged_vector().at(j),
                             ref_proxy.jagged_vector().at(j));
        }
    }
}
