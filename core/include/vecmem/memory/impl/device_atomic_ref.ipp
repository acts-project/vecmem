/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

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

namespace vecmem {

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
    return load();
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE void device_atomic_ref<T, address>::store(
    value_type data, memory_order) const {

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!(defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)))
    volatile pointer addr = m_ptr;
    __threadfence();
    *addr = data;
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    __VECMEM_SYCL_ATOMIC_CALL1(store, m_ptr, data);
#else
    *m_ptr = data;
#endif
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::load(
    memory_order) const -> value_type {

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!(defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)))
    volatile pointer addr = m_ptr;
    __threadfence();
    const value_type value = *addr;
    __threadfence();
    return value;
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL0(load, m_ptr);
#else
    return *m_ptr;
#endif
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::exchange(
    value_type data, memory_order) const -> value_type {

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!(defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)))
    return atomicExch(m_ptr, data);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL1(exchange, m_ptr, data);
#else
    value_type current_value = *m_ptr;
    *m_ptr = data;
    return current_value;
#endif
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE bool
device_atomic_ref<T, address>::compare_exchange_strong(reference expected,
                                                       value_type desired,
                                                       memory_order,
                                                       memory_order) const {

    return compare_exchange_strong(expected, desired);
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE bool
device_atomic_ref<T, address>::compare_exchange_strong(reference expected,
                                                       value_type desired,
                                                       memory_order) const {

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!(defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)))
    return atomicCAS(m_ptr, expected, desired);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL2(compare_exchange_strong, m_ptr, expected,
                                      desired);
#else
    if (*m_ptr == expected) {
        *m_ptr = desired;
        return true;
    } else {
        expected = *m_ptr;
        return false;
    }
#endif
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_add(
    value_type data, memory_order) const -> value_type {

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!(defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)))
    return atomicAdd(m_ptr, data);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_add, m_ptr, data);
#else
    const value_type result = *m_ptr;
    *m_ptr += data;
    return result;
#endif
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_sub(
    value_type data, memory_order) const -> value_type {

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!(defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)))
    return atomicSub(m_ptr, data);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_sub, m_ptr, data);
#else
    const value_type result = *m_ptr;
    *m_ptr -= data;
    return result;
#endif
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_and(
    value_type data, memory_order) const -> value_type {

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!(defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)))
    return atomicAnd(m_ptr, data);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_and, m_ptr, data);
#else
    const value_type result = *m_ptr;
    *m_ptr &= data;
    return result;
#endif
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_or(
    value_type data, memory_order) const -> value_type {

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!(defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)))
    return atomicOr(m_ptr, data);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_or, m_ptr, data);
#else
    const value_type result = *m_ptr;
    *m_ptr |= data;
    return result;
#endif
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_xor(
    value_type data, memory_order) const -> value_type {

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)) && \
    (!(defined(SYCL_LANGUAGE_VERSION) || defined(CL_SYCL_LANGUAGE_VERSION)))
    return atomicXor(m_ptr, data);
#elif defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
    return __VECMEM_SYCL_ATOMIC_CALL1(fetch_xor, m_ptr, data);
#else
    const value_type result = *m_ptr;
    *m_ptr ^= data;
    return result;
#endif
}

}  // namespace vecmem

// Clean up after the SYCL macros.
#if defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)
#undef __VECMEM_SYCL_ATOMIC_CALL0
#undef __VECMEM_SYCL_ATOMIC_CALL1
#undef __VECMEM_SYCL_ATOMIC_CALL2
#endif
