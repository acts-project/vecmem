/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/containers/details/jagged_device_vector_iterator.hpp"
#include "vecmem/containers/details/reverse_iterator.hpp"
#include "vecmem/utils/type_traits.hpp"
#include "vecmem/utils/types.hpp"

// System include(s).
#include <cstddef>
#include <type_traits>
#include <vector>

namespace vecmem {
namespace data {
/**
 * @brief A view for jagged vectors.
 *
 * A jagged vector is a two-dimensional vector in which the inner vectors do
 * not necessarily have the same size. For example, a jagged vector might
 * look like this:
 *
 * [[0, 1, 2],
 *  [3, 4],
 *  [],
 *  [5, 6, 7]]
 *
 * This class is a view of existing two-dimensional vectors created using a
 * vector-of-vectors formalism. Elements cannot be added or removed through
 * this view, but individual elements can be accessed and modified.
 *
 * @warning This view class shares memory with the vectors from which it was
 * constructed. Operating on the underlying vectors while an instance of
 * this class exists deriving from it is undefined and may leave the view in
 * an undefined state.
 */
template <typename T>
class jagged_vector_view : public vector_view<vector_view<T>> {

    /// We cannot use boolean types.
    static_assert(!std::is_same<std::remove_cv_t<T>, bool>::value,
                  "bool is not supported in VecMem containers");

public:
    /// @name Type definitions inherited from @c vecmem::data::vector_view
    /// @{

    using typename vector_view<vector_view<T>>::size_type;
    using typename vector_view<vector_view<T>>::pointer;

    /// @}

    // Inherit the base class's constructors.
    using vector_view<vector_view<T>>::vector_view;
    /**
     * Default constructor
     */
    jagged_vector_view() = default;
    /**
     * Constructor with all the information held by the object.
     */
    VECMEM_HOST_AND_DEVICE
    jagged_vector_view(size_type size, pointer ptr, pointer host_ptr = nullptr);

    /**
     * Constructor from a "slightly different" @c
     * vecmem::details::jagged_vector_view object
     *
     * Only enabled if the wrapped type is different, but only by
     * const-ness. This complication is necessary to avoid problems from
     * SYCL. Which is very particular about having default copy
     * constructors for the types that it sends to kernels.
     */
    template <
        typename OTHERTYPE,
        std::enable_if_t<details::is_same_nc<T, OTHERTYPE>::value, bool> = true>
    VECMEM_HOST_AND_DEVICE jagged_vector_view(
        const jagged_vector_view<OTHERTYPE>& parent);

    /// Assignment operator from a "slightly different" object
    template <
        typename OTHERTYPE,
        std::enable_if_t<details::is_same_nc<T, OTHERTYPE>::value, bool> = true>
    VECMEM_HOST_AND_DEVICE jagged_vector_view& operator=(
        const jagged_vector_view<OTHERTYPE>& rhs);

    /// Access the host accessible array describing the inner vectors
    ///
    /// This may or may not return the same pointer as @c ptr(). If the
    /// underlying data is stored in host-accessible memory, then the two will
    /// be the same.
    ///
    /// If not, then @c ptr() will return the device accessible array, and this
    /// function returns a host-accessible one.
    ///
    /// @return A host-accessible pointer to the array describing the inner
    ///         vectors
    ///
    VECMEM_HOST_AND_DEVICE
    pointer host_ptr() const;

private:
    /// Host-accessible pointer to the inner vector array
    pointer m_host_ptr;

};  // struct jagged_vector_view

/// Get the capacities of the inner vectors of a jagged vector
///
/// @tparam The type held by the jagged vector
///
/// @param data The jagged vector to get the capacities from
/// @return The vector of capacities of the inner vectors
///
template <typename T>
VECMEM_HOST std::vector<typename vector_view<T>::size_type> get_capacities(
    const jagged_vector_view<T>& data);

}  // namespace data
}  // namespace vecmem

// Include the implementation.
#include "vecmem/containers/impl/jagged_vector_view.ipp"
