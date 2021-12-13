/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <gtest/gtest.h>

#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/memory/unique_ptr.hpp"

class CoreUniqueAllocPtrTest : public testing::Test {
protected:
    vecmem::host_memory_resource mr;
};

TEST_F(CoreUniqueAllocPtrTest, EmptyPointer) {
    vecmem::unique_alloc_ptr<int> ptr;

    ASSERT_EQ(ptr, nullptr);
}

TEST_F(CoreUniqueAllocPtrTest, NullPointer) {
    vecmem::unique_alloc_ptr<int> ptr = nullptr;

    ASSERT_EQ(ptr, nullptr);
}

TEST_F(CoreUniqueAllocPtrTest, AllocSingle) {
    vecmem::unique_alloc_ptr<int> ptr = vecmem::make_unique_alloc<int>(mr);

    ASSERT_NE(ptr, nullptr);

    *ptr = 8;

    ASSERT_EQ(*ptr, 8);
}

TEST_F(CoreUniqueAllocPtrTest, AllocUnboundedArray) {
    vecmem::unique_alloc_ptr<int[]> ptr =
        vecmem::make_unique_alloc<int[]>(mr, 4);

    ASSERT_NE(ptr, nullptr);

    ptr[0] = 0;
    ptr[1] = 1;
    ptr[2] = 2;
    ptr[3] = 3;

    ASSERT_EQ(ptr[0], 0);
    ASSERT_EQ(ptr[1], 1);
    ASSERT_EQ(ptr[2], 2);
    ASSERT_EQ(ptr[3], 3);
}

TEST_F(CoreUniqueAllocPtrTest, AllocUnbounded2DArray) {
    vecmem::unique_alloc_ptr<int[][3]> ptr =
        vecmem::make_unique_alloc<int[][3]>(mr, 2);

    ASSERT_NE(ptr, nullptr);

    ptr[0][0] = 0;
    ptr[0][1] = 1;
    ptr[0][2] = 2;
    ptr[1][0] = 3;
    ptr[1][1] = 4;
    ptr[1][2] = 5;

    ASSERT_EQ(ptr[0][0], 0);
    ASSERT_EQ(ptr[0][1], 1);
    ASSERT_EQ(ptr[0][2], 2);
    ASSERT_EQ(ptr[1][0], 3);
    ASSERT_EQ(ptr[1][1], 4);
    ASSERT_EQ(ptr[1][2], 5);
}

TEST_F(CoreUniqueAllocPtrTest, MoveSingle) {
    vecmem::unique_alloc_ptr<int> ptr1 = vecmem::make_unique_alloc<int>(mr);
    vecmem::unique_alloc_ptr<int> ptr2;

    ASSERT_NE(ptr1, nullptr);
    ASSERT_EQ(ptr2, nullptr);

    *ptr1 = 8;

    ASSERT_EQ(*ptr1, 8);

    ptr2 = std::move(ptr1);

    ASSERT_EQ(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);

    ASSERT_EQ(*ptr2, 8);
}

TEST_F(CoreUniqueAllocPtrTest, MoveUnboundedArray) {
    vecmem::unique_alloc_ptr<int[]> ptr1 =
        vecmem::make_unique_alloc<int[]>(mr, 4);
    vecmem::unique_alloc_ptr<int[]> ptr2;

    ASSERT_NE(ptr1, nullptr);
    ASSERT_EQ(ptr2, nullptr);

    ptr1[0] = 8;

    ASSERT_EQ(ptr1[0], 8);

    ptr2 = std::move(ptr1);

    ASSERT_EQ(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);

    ASSERT_EQ(ptr2[0], 8);
}

TEST_F(CoreUniqueAllocPtrTest, DeallocateSingle) {
    vecmem::unique_alloc_ptr<int> ptr1 = vecmem::make_unique_alloc<int>(mr);

    ptr1 = nullptr;

    ASSERT_EQ(ptr1, nullptr);
}
