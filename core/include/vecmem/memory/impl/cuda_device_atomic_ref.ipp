/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem {
namespace cuda {

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
    value_type data, memory_order) const {

    volatile pointer addr = m_ptr;
    __threadfence();
    *addr = data;
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::load(
    memory_order) const -> value_type {

    volatile pointer addr = m_ptr;
    __threadfence();
    const value_type value = *addr;
    __threadfence();
    return value;
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::exchange(
    value_type data, memory_order) const -> value_type {

    return atomicExch(m_ptr, data);
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
    reference expected, value_type desired, memory_order,
    memory_order failure) const {

    (void)failure;
    assert(failure != memory_order::release &&
           failure != memory_order::acq_rel);

    const value_type r = atomicCAS(m_ptr, expected, desired);
    // atomicCAS returns the old value, so the change will have succeeded if
    // the old value was the expected value.
    if (r == expected) {
        return true;
    } else {
        expected = r;
        return false;
    }
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_add(
    value_type data, memory_order) const -> value_type {

    return atomicAdd(m_ptr, data);
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_sub(
    value_type data, memory_order) const -> value_type {

    return atomicSub(m_ptr, data);
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_and(
    value_type data, memory_order) const -> value_type {

    return atomicAnd(m_ptr, data);
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_or(
    value_type data, memory_order) const -> value_type {

    return atomicOr(m_ptr, data);
}

template <typename T, device_address_space address>
VECMEM_HOST_AND_DEVICE auto device_atomic_ref<T, address>::fetch_xor(
    value_type data, memory_order) const -> value_type {

    return atomicXor(m_ptr, data);
}

}  // namespace cuda
}  // namespace vecmem
