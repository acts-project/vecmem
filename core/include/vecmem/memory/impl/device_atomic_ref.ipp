/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

#include <cassert>
#include <cstring>

// vecmem includes
#include "vecmem/memory/memory_order.hpp"

// HIP include
#if defined(__HIP_DEVICE_COMPILE__)
#include <hip/hip_runtime.h>
#endif

// SYCL include(s).
#if defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
#include <CL/sycl.hpp>
#endif

/// Helpers for explicit calls to the SYCL atomic functions
#if defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)

namespace vecmem::details {
template <device_address_space address>
struct sycl_address_space {};

template <>
struct sycl_address_space<device_address_space::global> {
    static constexpr cl::sycl::access::address_space add =
        cl::sycl::access::address_space::global_space;

    template <typename T>
    using ptr_t = cl::sycl::global_ptr<T>;
};
template <>
struct sycl_address_space<device_address_space::local> {
    static constexpr cl::sycl::access::address_space add =
        cl::sycl::access::address_space::local_space;
    template <typename T>
    using ptr_t = cl::sycl::local_ptr<T>;
};
}  // namespace vecmem::details

#define __VECMEM_SYCL_ATOMIC_CALL0(FNAME, PTR)                             \
    cl::sycl::atomic_##FNAME<value_type,                                   \
                             details::sycl_address_space<address>::add>(   \
        cl::sycl::atomic<value_type,                                       \
                         details::sycl_address_space<address>::add>(       \
            typename details::sycl_address_space<address>::template ptr_t< \
                value_type>(PTR)))
#define __VECMEM_SYCL_ATOMIC_CALL1(FNAME, PTR, ARG1)                       \
    cl::sycl::atomic_##FNAME<value_type,                                   \
                             details::sycl_address_space<address>::add>(   \
        cl::sycl::atomic<value_type,                                       \
                         details::sycl_address_space<address>::add>(       \
            typename details::sycl_address_space<address>::template ptr_t< \
                value_type>(PTR)),                                         \
        ARG1)
#define __VECMEM_SYCL_ATOMIC_CALL2(FNAME, PTR, ARG1, ARG2)                 \
    cl::sycl::atomic_##FNAME<value_type,                                   \
                             details::sycl_address_space<address>::add>(   \
        cl::sycl::atomic<value_type,                                       \
                         details::sycl_address_space<address>::add>(       \
            typename details::sycl_address_space<address>::template ptr_t< \
                value_type>(PTR)),                                         \
        ARG1, ARG2)
#endif

#if defined __has_builtin
#if __has_builtin(__atomic_load_n)
#define VECMEM_HAVE_BUILTIN_ATOMIC_LOAD
#endif
#if __has_builtin(__atomic_store_n)
#define VECMEM_HAVE_BUILTIN_ATOMIC_STORE
#endif
#if __has_builtin(__atomic_exchange_n)
#define VECMEM_HAVE_BUILTIN_ATOMIC_EXCHANGE
#endif
#if __has_builtin(__atomic_compare_exchange_n)
#define VECMEM_HAVE_BUILTIN_ATOMIC_COMPARE_EXCHANGE
#endif
#if __has_builtin(__atomic_fetch_add)
#define VECMEM_HAVE_BUILTIN_ATOMIC_FETCH_ADD
#endif
#if __has_builtin(__atomic_fetch_sub)
#define VECMEM_HAVE_BUILTIN_ATOMIC_FETCH_SUB
#endif
#if __has_builtin(__atomic_fetch_and)
#define VECMEM_HAVE_BUILTIN_ATOMIC_FETCH_AND
#endif
#if __has_builtin(__atomic_fetch_xor)
#define VECMEM_HAVE_BUILTIN_ATOMIC_FETCH_XOR
#endif
#if __has_builtin(__atomic_fetch_or)
#define VECMEM_HAVE_BUILTIN_ATOMIC_FETCH_OR
#endif
#endif

namespace vecmem {

#if defined(__ATOMIC_RELAXED) && defined(__ATOMIC_CONSUME) && \
    defined(__ATOMIC_ACQUIRE) && defined(__ATOMIC_RELEASE) && \
    defined(__ATOMIC_ACQ_REL) && defined(__ATOMIC_SEQ_CST)
constexpr int __memorder_vecmem_to_builtin(memory_order o) {
    switch (o) {
        case memory_order::relaxed:
            return __ATOMIC_RELAXED;
        case memory_order::consume:
            return __ATOMIC_CONSUME;
        case memory_order::acquire:
            return __ATOMIC_ACQUIRE;
        case memory_order::release:
            return __ATOMIC_RELEASE;
        case memory_order::acq_rel:
            return __ATOMIC_ACQ_REL;
        case memory_order::seq_cst:
            return __ATOMIC_SEQ_CST;
        default:
            assert(false);
            return 0;
    }
}
#define VECMEM_HAVE_MEMORDER_DEFINITIONS
#endif

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE device_atomic_ref<T, address>::device_atomic_ref(
    reference ref)
    : m_ptr(&ref) {}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE device_atomic_ref<T, address>::device_atomic_ref(
    const device_atomic_ref& parent)
    : m_ptr(parent.m_ptr) {}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::operator=(
    value_type data) const -> value_type {

    store(data);
    return data;
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE void device_atomic_ref<T, address>::store(
    value_type data, memory_order order) const {
    (void)order;
#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!(defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)))
    volatile pointer addr = m_ptr;
    __threadfence();
    *addr = data;
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    __VECMEM_SYCL_ATOMIC_CALL1(store, m_ptr, data);
#elif defined(VECMEM_HAVE_BUILTIN_ATOMIC_STORE) && \
    defined(VECMEM_HAVE_MEMORDER_DEFINITIONS)
    __atomic_store_n(m_ptr, data, __memorder_vecmem_to_builtin(order));
#elif defined(__clang__) || defined(__GNUC__) || defined(__GNUG__) || \
    defined(__CUDACC__)
    __atomic_store_n(m_ptr, data, __memorder_vecmem_to_builtin(order));
#else
    exchange(data, order);
#endif
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::load(
    memory_order order) const -> value_type {
    (void)order;
#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!(defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)))
    volatile pointer addr = m_ptr;
    __threadfence();
    const value_type value = *addr;
    __threadfence();
    return value;
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL0(load, m_ptr);
#elif defined(VECMEM_HAVE_BUILTIN_ATOMIC_LOAD) && \
    defined(VECMEM_HAVE_MEMORDER_DEFINITIONS)
    return __atomic_load_n(m_ptr, __memorder_vecmem_to_builtin(order));
#elif defined(__clang__) || defined(__GNUC__) || defined(__GNUG__) || \
    defined(__CUDACC__)
    return __atomic_load_n(m_ptr, __memorder_vecmem_to_builtin(order));
#else
    value_type tmp = 0;
    compare_exchange_strong(tmp, tmp, order);
    return tmp;
#endif
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::exchange(
    value_type data, memory_order order) const -> value_type {
    (void)order;
#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!(defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)))
    return atomicExch(m_ptr, data);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL1(exchange, m_ptr, data);
#elif defined(VECMEM_HAVE_BUILTIN_ATOMIC_EXCHANGE) && \
    defined(VECMEM_HAVE_MEMORDER_DEFINITIONS)
    return __atomic_exchange_n(m_ptr, data,
                               __memorder_vecmem_to_builtin(order));
#elif defined(__clang__) || defined(__GNUC__) || defined(__GNUG__) || \
    defined(__CUDACC__)
    return __atomic_exchange_n(m_ptr, data,
                               __memorder_vecmem_to_builtin(order));
#else
    value_type tmp = load();
    while (!compare_exchange_strong(tmp, data, order))
        ;
    return tmp;
#endif
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE bool
device_atomic_ref<T, address>::compare_exchange_strong(
    reference expected, value_type desired, memory_order order) const {
    if (order == memory_order::acq_rel) {
        return compare_exchange_strong(expected, desired, order,
                                       memory_order::acquire);
    } else if (order == memory_order::release) {
        return compare_exchange_strong(expected, desired, order,
                                       memory_order::relaxed);
    } else {
        return compare_exchange_strong(expected, desired, order, order);
    }
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE bool
device_atomic_ref<T, address>::compare_exchange_strong(
    reference expected, value_type desired, memory_order success,
    memory_order failure) const {
    (void)success, (void)failure;
    assert(failure != memory_order::release &&
           failure != memory_order::acq_rel);
#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!(defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)))
    value_type r = atomicCAS(m_ptr, expected, desired);
    // atomicCAS returns the old value, so the change will have succeeded if
    // the old value was the expected value.
    if (r == expected) {
        return true;
    } else {
        expected = r;
        return false;
    }
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL2(compare_exchange_strong, m_ptr, expected,
                                      desired);
#elif defined(VECMEM_HAVE_BUILTIN_ATOMIC_CAS) && \
    defined(VECMEM_HAVE_MEMORDER_DEFINITIONS)
    return __atomic_compare_exchange_n(m_ptr, &expected, desired, false,
                                       __memorder_vecmem_to_builtin(success),
                                       __memorder_vecmem_to_builtin(failure));
#elif defined(__clang__) || defined(__GNUC__) || defined(__GNUG__) || \
    defined(__CUDACC__)
    return __atomic_compare_exchange_n(m_ptr, &expected, desired, false,
                                       __memorder_vecmem_to_builtin(success),
                                       __memorder_vecmem_to_builtin(failure));
#else
    // This is **NOT** a sane implementation of CAS!
    value_type old = *m_ptr;
    if (old == expected) {
        *m_ptr = desired;
        return true;
    } else {
        expected = old;
        return false;
    }
#endif
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_add(
    value_type data, memory_order order) const -> value_type {
    (void)order;
#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!(defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)))
    return atomicAdd(m_ptr, data);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_add, m_ptr, data);
#elif defined(VECMEM_HAVE_BUILTIN_ATOMIC_ADD) && \
    defined(VECMEM_HAVE_MEMORDER_DEFINITIONS)
    return __atomic_fetch_add(m_ptr, data, __memorder_vecmem_to_builtin(order));
#elif defined(__clang__) || defined(__GNUC__) || defined(__GNUG__) || \
    defined(__CUDACC__)
    return __atomic_fetch_add(m_ptr, data, __memorder_vecmem_to_builtin(order));
#else
    value_type tmp = load();
    while (!compare_exchange_strong(tmp, tmp + data, order))
        ;
    return tmp;
#endif
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_sub(
    value_type data, memory_order order) const -> value_type {
    (void)order;
#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!(defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)))
    return atomicSub(m_ptr, data);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_sub, m_ptr, data);
#elif defined(VECMEM_HAVE_BUILTIN_ATOMIC_SUB) && \
    defined(VECMEM_HAVE_MEMORDER_DEFINITIONS)
    return __atomic_fetch_sub(m_ptr, data, __memorder_vecmem_to_builtin(order));
#elif defined(VECMEM_HAVE_BUILTIN_ATOMIC_ADD) && \
    defined(VECMEM_HAVE_MEMORDER_DEFINITIONS)
    return __atomic_fetch_add(m_ptr, -data,
                              __memorder_vecmem_to_builtin(order));
#elif defined(__clang__) || defined(__GNUC__) || defined(__GNUG__) || \
    defined(__CUDACC__)
    return __atomic_fetch_sub(m_ptr, data, __memorder_vecmem_to_builtin(order));
#else
    value_type tmp = load();
    while (!compare_exchange_strong(tmp, tmp - data, order))
        ;
    return tmp;
#endif
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_and(
    value_type data, memory_order order) const -> value_type {
    (void)order;
#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!(defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)))
    return atomicAnd(m_ptr, data);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_and, m_ptr, data);
#elif defined(VECMEM_HAVE_BUILTIN_ATOMIC_AND) && \
    defined(VECMEM_HAVE_MEMORDER_DEFINITIONS)
    return __atomic_fetch_and(m_ptr, data, __memorder_vecmem_to_builtin(order));
#elif defined(__clang__) || defined(__GNUC__) || defined(__GNUG__) || \
    defined(__CUDACC__)
    return __atomic_fetch_and(m_ptr, data, __memorder_vecmem_to_builtin(order));
#else
    value_type tmp = load();
    while (!compare_exchange_strong(tmp, tmp & data, order))
        ;
    return tmp;
#endif
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_or(
    value_type data, memory_order order) const -> value_type {
    (void)order;
#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!(defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)))
    return atomicOr(m_ptr, data);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_or, m_ptr, data);
#elif defined(VECMEM_HAVE_BUILTIN_ATOMIC_OR) && \
    defined(VECMEM_HAVE_MEMORDER_DEFINITIONS)
    return __atomic_fetch_or(m_ptr, data, __memorder_vecmem_to_builtin(order));
#elif defined(__clang__) || defined(__GNUC__) || defined(__GNUG__) || \
    defined(__CUDACC__)
    return __atomic_fetch_or(m_ptr, data, __memorder_vecmem_to_builtin(order));
#else
    value_type tmp = load();
    while (!compare_exchange_strong(tmp, tmp | data, order))
        ;
    return tmp;
#endif
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_xor(
    value_type data, memory_order order) const -> value_type {
    (void)order;
#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!(defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)))
    return atomicXor(m_ptr, data);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_xor, m_ptr, data);
#elif defined(VECMEM_HAVE_BUILTIN_ATOMIC_XOR) && \
    defined(VECMEM_HAVE_MEMORDER_DEFINITIONS)
    return __atomic_fetch_xor(m_ptr, data, __memorder_vecmem_to_builtin(order));
#elif defined(__clang__) || defined(__GNUC__) || defined(__GNUG__) || \
    defined(__CUDACC__)
    return __atomic_fetch_xor(m_ptr, data, __memorder_vecmem_to_builtin(order));
#else
    value_type tmp = load();
    while (!compare_exchange_strong(tmp, tmp ^ data, order))
        ;
    return tmp;
#endif
}

}  // namespace vecmem

// Clean up after the SYCL macros.
#if defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
#undef __VECMEM_SYCL_ATOMIC_CALL0
#undef __VECMEM_SYCL_ATOMIC_CALL1
#undef __VECMEM_SYCL_ATOMIC_CALL2
#endif

#undef VECMEM_HAVE_BUILTIN_ATOMIC_LOAD
#undef VECMEM_HAVE_BUILTIN_ATOMIC_STORE
#undef VECMEM_HAVE_BUILTIN_ATOMIC_EXCHANGE
#undef VECMEM_HAVE_BUILTIN_ATOMIC_COMPARE_EXCHANGE
#undef VECMEM_HAVE_BUILTIN_ATOMIC_FETCH_ADD
#undef VECMEM_HAVE_BUILTIN_ATOMIC_FETCH_SUB
#undef VECMEM_HAVE_BUILTIN_ATOMIC_FETCH_AND
#undef VECMEM_HAVE_BUILTIN_ATOMIC_FETCH_XOR
#undef VECMEM_HAVE_BUILTIN_ATOMIC_FETCH_OR
#undef VECMEM_HAVE_MEMORDER_DEFINITIONS
