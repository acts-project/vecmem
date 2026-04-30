/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local includes
#include "../../hip/src/utils/hip_error_handling.hpp"
#include "vecmem/memory/device_atomic_ref.hpp"

// HIP include(s).
#include <hip/hip_runtime.h>

// GoogleTest include(s).
#include <gtest/gtest.h>

// Suppress Clang's C++17 attribute-extension warning triggered by GoogleTest's
// use of [[maybe_unused]] inside the TEST() macro, which expands at the
// HIP_TEST() call sites below. The HIP test target compiles as C++14, but
// amdclang++ flags the attribute as a C++17 extension under -pedantic -Werror.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wc++17-attribute-extensions"
#endif

#define HIP_ASSERT_TRUE(o, v) \
    do {                      \
        *out &= (!!v);        \
    } while (0)
#define HIP_ASSERT_FALSE(o, v) \
    do {                       \
        *out &= (!v);          \
    } while (0)
#define HIP_ASSERT_EQ(o, i, j) \
    do {                       \
        *out &= (i == j);      \
    } while (0)

#define HIP_TEST(s, n, k)                                             \
    TEST(s, n) {                                                      \
        bool *b, out;                                                 \
        int* i;                                                       \
        VECMEM_HIP_ERROR_CHECK(hipMalloc(&b, sizeof(bool)));          \
        VECMEM_HIP_ERROR_CHECK(hipMalloc(&i, sizeof(int)));           \
        VECMEM_HIP_ERROR_CHECK(hipMemset(b, 1, sizeof(bool)));        \
        k<<<1, 1>>>(b, i);                                            \
        VECMEM_HIP_ERROR_CHECK(hipGetLastError());                    \
        VECMEM_HIP_ERROR_CHECK(hipDeviceSynchronize());               \
        VECMEM_HIP_ERROR_CHECK(                                       \
            hipMemcpy(&out, b, sizeof(bool), hipMemcpyDeviceToHost)); \
        ASSERT_TRUE(out);                                             \
        VECMEM_HIP_ERROR_CHECK(hipFree(b));                           \
        VECMEM_HIP_ERROR_CHECK(hipFree(i));                           \
    }

__global__ void atomic_ref_operator_equals_kernel(bool* out, int* i) {
    *i = 0;
    vecmem::device_atomic_ref<int> a(*i);
    a = 5;
    HIP_ASSERT_EQ(out, a.load(), 5);
}

HIP_TEST(hip_device_atomic_test, atomic_operator_equals,
         atomic_ref_operator_equals_kernel);

__global__ void atomic_ref_load_kernel(bool* out, int* i) {
    *i = 17;
    vecmem::device_atomic_ref<int> a(*i);
    HIP_ASSERT_EQ(out, a.load(), 17);
}

HIP_TEST(hip_device_atomic_test, atomic_load, atomic_ref_load_kernel);

__global__ void atomic_ref_compare_exchange_strong_kernel(bool* out, int* i) {
    *i = 0;
    int zero = 0, five = 5;
    vecmem::device_atomic_ref<int> a(*i);
    // This passes, as i == 0, zero == 0. This will set zero = i = 0 and i = 5;
    HIP_ASSERT_TRUE(out, a.compare_exchange_strong(zero, 5));
    HIP_ASSERT_EQ(out, zero, 0);
    // This fails, as i == 5, zero == 0. This will set zero = i = 5;
    HIP_ASSERT_FALSE(out, a.compare_exchange_strong(zero, 5));
    HIP_ASSERT_EQ(out, zero, 5);
    // This succeeds, as i == 5, five == 5. This will set five = i = 5 and i =
    // 0;
    HIP_ASSERT_TRUE(out, a.compare_exchange_strong(five, 0));
    HIP_ASSERT_EQ(out, five, 5);
    // This fails, as i == 0, five == 5. This will set five = i = 0;
    HIP_ASSERT_FALSE(out, a.compare_exchange_strong(five, 0));
    HIP_ASSERT_EQ(out, five, 0);
}

HIP_TEST(hip_device_atomic_test, atomic_compare_exchange_strong,
         atomic_ref_compare_exchange_strong_kernel);

__global__ void atomic_ref_store_kernel(bool* out, int* i) {
    *i = 0;
    vecmem::device_atomic_ref<int> a(*i);
    HIP_ASSERT_EQ(out, a.load(), 0);
    a.store(5);
    HIP_ASSERT_EQ(out, a.load(), 5);
}

HIP_TEST(hip_device_atomic_test, atomic_store, atomic_ref_store_kernel);

__global__ void atomic_ref_exchange_kernel(bool* out, int* i) {
    *i = 0;
    vecmem::device_atomic_ref<int> a(*i);
    HIP_ASSERT_EQ(out, a.load(), 0);
    HIP_ASSERT_EQ(out, a.exchange(5), 0);
    HIP_ASSERT_EQ(out, a.load(), 5);
    HIP_ASSERT_EQ(out, a.exchange(3), 5);
    HIP_ASSERT_EQ(out, a.load(), 3);
}

HIP_TEST(hip_device_atomic_test, atomic_exchange, atomic_ref_exchange_kernel);

__global__ void atomic_ref_fetch_add_kernel(bool* out, int* i) {
    *i = 0;
    vecmem::device_atomic_ref<int> a(*i);
    HIP_ASSERT_EQ(out, a.load(), 0);
    HIP_ASSERT_EQ(out, a.fetch_add(5), 0);
    HIP_ASSERT_EQ(out, a.load(), 5);
    HIP_ASSERT_EQ(out, a.fetch_add(3), 5);
    HIP_ASSERT_EQ(out, a.load(), 8);
}

HIP_TEST(hip_device_atomic_test, atomic_fetch_add, atomic_ref_fetch_add_kernel);

__global__ void atomic_ref_fetch_sub_kernel(bool* out, int* i) {
    *i = 0;
    vecmem::device_atomic_ref<int> a(*i);
    HIP_ASSERT_EQ(out, a.load(), 0);
    HIP_ASSERT_EQ(out, a.fetch_sub(5), 0);
    HIP_ASSERT_EQ(out, a.load(), -5);
    HIP_ASSERT_EQ(out, a.fetch_sub(3), -5);
    HIP_ASSERT_EQ(out, a.load(), -8);
}

HIP_TEST(hip_device_atomic_test, atomic_fetch_sub, atomic_ref_fetch_sub_kernel);

__global__ void atomic_ref_fetch_and_kernel(bool* out, int* i) {
    *i = 0b0101;
    vecmem::device_atomic_ref<int> a(*i);
    HIP_ASSERT_EQ(out, a.load(), 0b0101);
    HIP_ASSERT_EQ(out, a.fetch_and(0b1100), 0b0101);
    HIP_ASSERT_EQ(out, a.load(), 0b0100);
    HIP_ASSERT_EQ(out, a.fetch_and(0b0000), 0b0100);
    HIP_ASSERT_EQ(out, a.load(), 0b0000);
}

HIP_TEST(hip_device_atomic_test, atomic_fetch_and, atomic_ref_fetch_and_kernel);

__global__ void atomic_ref_fetch_or_kernel(bool* out, int* i) {
    *i = 0b0101;
    vecmem::device_atomic_ref<int> a(*i);
    HIP_ASSERT_EQ(out, a.load(), 0b0101);
    HIP_ASSERT_EQ(out, a.fetch_or(0b1100), 0b0101);
    HIP_ASSERT_EQ(out, a.load(), 0b1101);
    HIP_ASSERT_EQ(out, a.fetch_or(0b0000), 0b1101);
    HIP_ASSERT_EQ(out, a.load(), 0b1101);
}

HIP_TEST(hip_device_atomic_test, atomic_fetch_or, atomic_ref_fetch_or_kernel);

__global__ void atomic_ref_fetch_xor_kernel(bool* out, int* i) {
    *i = 0b0101;
    vecmem::device_atomic_ref<int> a(*i);
    HIP_ASSERT_EQ(out, a.load(), 0b0101);
    HIP_ASSERT_EQ(out, a.fetch_xor(0b1100), 0b0101);
    HIP_ASSERT_EQ(out, a.load(), 0b1001);
    HIP_ASSERT_EQ(out, a.fetch_xor(0b0000), 0b1001);
    HIP_ASSERT_EQ(out, a.load(), 0b1001);
}

HIP_TEST(hip_device_atomic_test, atomic_fetch_xor, atomic_ref_fetch_xor_kernel);

__global__ void atomic_ref_fetch_max_kernel(bool* out, int* i) {
    *i = 0;
    vecmem::device_atomic_ref<int> a(*i);
    HIP_ASSERT_EQ(out, a.load(), 0);
    HIP_ASSERT_EQ(out, a.fetch_max(5), 0);
    HIP_ASSERT_EQ(out, a.load(), 5);
    HIP_ASSERT_EQ(out, a.fetch_max(3), 5);
    HIP_ASSERT_EQ(out, a.load(), 5);
}

HIP_TEST(hip_device_atomic_test, atomic_fetch_max, atomic_ref_fetch_max_kernel);

__global__ void atomic_ref_fetch_min_kernel(bool* out, int* i) {
    *i = 0;
    vecmem::device_atomic_ref<int> a(*i);
    HIP_ASSERT_EQ(out, a.load(), 0);
    HIP_ASSERT_EQ(out, a.fetch_min(-5), 0);
    HIP_ASSERT_EQ(out, a.load(), -5);
    HIP_ASSERT_EQ(out, a.fetch_min(3), -5);
    HIP_ASSERT_EQ(out, a.load(), -5);
}

HIP_TEST(hip_device_atomic_test, atomic_fetch_min, atomic_ref_fetch_min_kernel);