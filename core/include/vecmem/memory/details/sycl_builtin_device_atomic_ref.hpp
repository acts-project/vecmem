/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/details/device_address_space.hpp"

// SYCL include(s).
#include <CL/sycl.hpp>

namespace vecmem {
namespace sycl {
namespace details {

/// Helper trait for setting up an atomic reference on global or local memory
///
/// @tparam address The address space to use
///
template <device_address_space address>
struct builtin_address_space {};

/// Specialization for global device memory
template <>
struct builtin_address_space<device_address_space::global> {
    static constexpr cl::sycl::memory_order ord =
        cl::sycl::memory_order::relaxed;
    static constexpr cl::sycl::memory_scope scp =
        cl::sycl::memory_scope::device;
    static constexpr cl::sycl::access::address_space add =
        cl::sycl::access::address_space::global_space;
};

/// Specialization for local device memory
template <>
struct builtin_address_space<device_address_space::local> {
    static constexpr cl::sycl::memory_order ord =
        cl::sycl::memory_order::relaxed;
    static constexpr cl::sycl::memory_scope scp =
        cl::sycl::memory_scope::work_group;
    static constexpr cl::sycl::access::address_space add =
        cl::sycl::access::address_space::local_space;
};

}  // namespace details

/// Atomic reference based on @c ::sycl::atomic_ref
template <typename T,
          device_address_space address = device_address_space::global>
using builtin_device_atomic_ref =
    cl::sycl::atomic_ref<T, details::builtin_address_space<address>::ord,
                         details::builtin_address_space<address>::scp,
                         details::builtin_address_space<address>::add>;

}  // namespace sycl
}  // namespace vecmem
