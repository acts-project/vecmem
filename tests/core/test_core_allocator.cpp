/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "vecmem/memory/allocator.hpp"
#include "vecmem/memory/host_memory_resource.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <type_traits>
#include <vector>

class test_class {
    public:
    test_class() : m_int_2(11), m_bool_2(true) {}
    test_class(int n) : m_int_2(n), m_bool_2(false) {}
    test_class(int n, int m) : m_int_2(n), m_bool_2(m > 100) {}

    int m_int_1 = 5;
    int m_int_2;

    bool m_bool_1 = false;
    bool m_bool_2;
};

class core_allocator_test : public testing::Test {
    protected:
    vecmem::host_memory_resource m_upstream;
    vecmem::allocator * m_alloc;

    void SetUp() override {
        m_alloc = new vecmem::allocator(m_upstream);
    }
};

TEST_F(core_allocator_test, basic) {
    void * p = m_alloc->allocate_bytes(1024);

    EXPECT_TRUE(p != nullptr);

    m_alloc->deallocate_bytes(p, 1024);
}

TEST_F(core_allocator_test, primitive) {
    int * p = m_alloc->allocate_object<int>();

    ASSERT_TRUE(p != nullptr);

    *p = 5;

    EXPECT_TRUE(*p == 5);

    m_alloc->deallocate_object<int>(p);
}

TEST_F(core_allocator_test, array) {
    int * p = m_alloc->allocate_object<int>(10);

    ASSERT_TRUE(p != nullptr);

    for (int i = 0; i < 10; ++i) {
        p[i] = i;
    }

    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(p[i] == i);
    }

    m_alloc->deallocate_object<int>(p, 10);
}

TEST_F(core_allocator_test, constructor) {
    test_class * p1 = m_alloc->new_object<test_class>();
    test_class * p2 = m_alloc->new_object<test_class>(12);
    test_class * p3 = m_alloc->new_object<test_class>(21, 611);
    test_class * p4 = m_alloc->new_object<test_class>(21, 15);

    ASSERT_TRUE(p1 != nullptr);
    ASSERT_TRUE(p2 != nullptr);
    ASSERT_TRUE(p3 != nullptr);
    ASSERT_TRUE(p4 != nullptr);

    EXPECT_TRUE(p1->m_int_1 == 5);
    EXPECT_TRUE(p1->m_int_2 == 11);
    EXPECT_TRUE(p1->m_bool_1 == false);
    EXPECT_TRUE(p1->m_bool_2 == true);

    EXPECT_TRUE(p2->m_int_1 == 5);
    EXPECT_TRUE(p2->m_int_2 == 12);
    EXPECT_TRUE(p2->m_bool_1 == false);
    EXPECT_TRUE(p2->m_bool_2 == false);

    EXPECT_TRUE(p3->m_int_1 == 5);
    EXPECT_TRUE(p3->m_int_2 == 21);
    EXPECT_TRUE(p3->m_bool_1 == false);
    EXPECT_TRUE(p3->m_bool_2 == true);

    EXPECT_TRUE(p4->m_int_1 == 5);
    EXPECT_TRUE(p4->m_int_2 == 21);
    EXPECT_TRUE(p4->m_bool_1 == false);
    EXPECT_TRUE(p4->m_bool_2 == false);
}
