/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2022-2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <atomic>

// vecmem includes
#include "vecmem/memory/memory_order.hpp"

namespace vecmem {

/// Custom definition for the adress space
enum class device_address_space { global = 0, local = 1 };

}  // namespace vecmem

#if (defined(CL_SYCL_LANGUAGE_VERSION) || defined(SYCL_LANGUAGE_VERSION)) && \
    defined(VECMEM_HAVE_SYCL_ATOMIC_REF)

// SYCL include(s).
#include <CL/sycl.hpp>

namespace vecmem {
namespace details {
template <device_address_space address>
struct sycl_address_space {};

template <>
struct sycl_address_space<device_address_space::global> {
    static constexpr ::sycl::memory_order ord = ::sycl::memory_order::relaxed;
    static constexpr ::sycl::memory_scope scp = ::sycl::memory_scope::device;
    static constexpr ::sycl::access::address_space add =
        ::sycl::access::address_space::global_space;
};
template <>
struct sycl_address_space<device_address_space::local> {
    static constexpr ::sycl::memory_order ord = ::sycl::memory_order::relaxed;
    static constexpr ::sycl::memory_scope scp =
        ::sycl::memory_scope::work_group;
    static constexpr ::sycl::access::address_space add =
        ::sycl::access::address_space::local_space;
};
}  // namespace details

/// @c vecmem::atomic_ref equals @c sycl::atomic_ref with "modern SYCL"
template <typename T,
          device_address_space address = device_address_space::global>
using device_atomic_ref =
    ::sycl::atomic_ref<T, details::sycl_address_space<address>::ord,
                       details::sycl_address_space<address>::scp,
                       details::sycl_address_space<address>::add>;

}  // namespace vecmem

#elif ((!defined(__CUDA_ARCH__)) && (!defined(__HIP_DEVICE_COMPILE__)) && \
       (!defined(CL_SYCL_LANGUAGE_VERSION)) &&                            \
       (!defined(SYCL_LANGUAGE_VERSION)) && __cpp_lib_atomic_ref)

namespace vecmem {
/// @c vecmem::atomic_ref equals @c std::atomic_ref in host code with C++20
template <typename T, device_address_space = device_address_space::global>
using device_atomic_ref = std::atomic_ref<T>;

}  // namespace vecmem

#else

// VecMem include(s).
#include "vecmem/utils/types.hpp"

// System include(s).
#include <type_traits>

namespace vecmem {
/// Class providing atomic operations for the VecMem code
///
/// It is only meant to be used with primitive types. Ones that CUDA, HIP and
/// SYCL built-in functions exist for. So no structs, or even pointers.
///
/// Note that it does not perform atomic operations in host code! That is only
/// implemented with @c std::atomic_ref in C++20. With earlier C++ standards all
/// operations in host code are performed as "regular" operations.
///
template <typename T, device_address_space = device_address_space::global>
class device_atomic_ref {

public:
    /// @name Type definitions
    /// @{

    /// Type managed by the object
    typedef T value_type;
    /// Difference between two objects
    typedef value_type difference_type;
    /// Pointer to the value in global memory
    typedef value_type* pointer;
    /// Reference to a value given by the user
    typedef value_type& reference;

    /// @}

    /// @name Check(s) on the value type
    /// @{

    static_assert(std::is_integral<value_type>::value,
                  "vecmem::atomic_ref only accepts built-in integral types");

    /// @}

    /// Constructor, with a pointer to the managed variable
    VECMEM_HOST_AND_DEVICE
    explicit device_atomic_ref(reference ref);
    /// Copy constructor
    VECMEM_HOST_AND_DEVICE
    device_atomic_ref(const device_atomic_ref& parent);

    /// Disable the assignment operator
    device_atomic_ref& operator=(const device_atomic_ref&) = delete;

    /// @name Value setter/getter functions
    /// @{

    /// Assigns a value desired to the referenced object
    ///
    /// @see vecmem::device_atomic_ref::store
    ///
    VECMEM_HOST_AND_DEVICE
    value_type operator=(value_type data) const;

    /// Set the variable to the desired value
    VECMEM_HOST_AND_DEVICE
    void store(value_type data,
               memory_order order = memory_order::seq_cst) const;
    /// Get the value of the variable
    VECMEM_HOST_AND_DEVICE
    value_type load(memory_order order = memory_order::seq_cst) const;

    /// Exchange the current value of the variable with a different one
    VECMEM_HOST_AND_DEVICE
    value_type exchange(value_type data,
                        memory_order order = memory_order::seq_cst) const;

    /// Compare against the current value, and exchange only if different
    VECMEM_HOST_AND_DEVICE
    bool compare_exchange_strong(reference expected, value_type desired,
                                 memory_order success,
                                 memory_order failure) const;
    /// Compare against the current value, and exchange only if different
    VECMEM_HOST_AND_DEVICE
    bool compare_exchange_strong(
        reference expected, value_type desired,
        memory_order order = memory_order::seq_cst) const;

    /// @}

    /// @name Value modifier functions
    /// @{

    /// Add a chosen value to the stored variable
    VECMEM_HOST_AND_DEVICE
    value_type fetch_add(value_type data,
                         memory_order order = memory_order::seq_cst) const;
    /// Substitute a chosen value from the stored variable
    VECMEM_HOST_AND_DEVICE
    value_type fetch_sub(value_type data,
                         memory_order order = memory_order::seq_cst) const;

    /// Replace the current value with the specified value AND-ed to it
    VECMEM_HOST_AND_DEVICE
    value_type fetch_and(value_type data,
                         memory_order order = memory_order::seq_cst) const;
    /// Replace the current value with the specified value OR-d to it
    VECMEM_HOST_AND_DEVICE
    value_type fetch_or(value_type data,
                        memory_order order = memory_order::seq_cst) const;
    /// Replace the current value with the specified value XOR-d to it
    VECMEM_HOST_AND_DEVICE
    value_type fetch_xor(value_type data,
                         memory_order order = memory_order::seq_cst) const;

    /// @}

private:
    /// Pointer to the value to perform atomic operations on
    pointer m_ptr;

};  // class device_atomic_ref

}  // namespace vecmem

// Include the implementation.
#include "vecmem/memory/impl/device_atomic_ref.ipp"

#if __cpp_concepts >= 201907L
#include "vecmem/concepts/atomic_ref.hpp"
static_assert(
    vecmem::concepts::atomic_ref<vecmem::device_atomic_ref<uint32_t> >,
    "Atomic reference on uint32_t is incompletely defined.");
static_assert(
    vecmem::concepts::atomic_ref<vecmem::device_atomic_ref<uint64_t> >,
    "Atomic reference on uint64_t is incompletely defined.");
static_assert(
    vecmem::concepts::atomic_ref<vecmem::device_atomic_ref<std::size_t> >,
    "Atomic reference on std::size_t is incompletely defined.");
#endif

#endif  // Platform selection
