/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/buffer_type.hpp"
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/memory/unique_ptr.hpp"

// System include(s).
#include <cstddef>
#include <memory>
#include <type_traits>

namespace vecmem {
namespace data {

/// Object owning the data held by it
///
/// This can come in handy in a number of cases, especially when using
/// device-only memory blocks.
///
template <typename TYPE>
class vector_buffer : public vector_view<TYPE> {

public:
    /// The base type used by this class
    using base_type = vector_view<TYPE>;
    /// Size type definition coming from the base class
    using size_type = typename base_type::size_type;
    /// Size pointer type definition coming from the base class
    using size_pointer = typename base_type::size_pointer;
    /// Pointer type definition coming from the base class
    using pointer = typename base_type::pointer;

    /// @name Checks on the type of the array element
    /// @{

    /// Make sure that the template type does not have a custom destructor
    static_assert(std::is_trivially_destructible_v<TYPE>,
                  "vecmem::data::vector_buffer can not handle types with "
                  "custom destructors");

    /// @}

    /// Default constructor
    vector_buffer();
    /// Standard constructor
    vector_buffer(size_type capacity, memory_resource& resource,
                  buffer_type type = buffer_type::fixed_size);
    /// Move constructor
    vector_buffer(vector_buffer&&) noexcept = default;

    /// Move assignment
    vector_buffer& operator=(vector_buffer&&) noexcept = default;

private:
    /// Data object owning the allocated memory
    vecmem::unique_alloc_ptr<char[]> m_memory;

};  // class vector_buffer

}  // namespace data

/// Helper function creating a @c vecmem::data::vector_view object
template <typename TYPE>
data::vector_view<TYPE>& get_data(data::vector_buffer<TYPE>& data);

/// Helper function creating a @c vecmem::data::vector_view object
template <typename TYPE>
const data::vector_view<TYPE>& get_data(const data::vector_buffer<TYPE>& data);

}  // namespace vecmem

// Include the implementation.
#include "vecmem/containers/impl/vector_buffer.ipp"
