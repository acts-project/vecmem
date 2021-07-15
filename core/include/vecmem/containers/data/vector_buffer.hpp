/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/data/vector_view.hpp"
#include "vecmem/memory/deallocator.hpp"
#include "vecmem/memory/memory_resource.hpp"

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
    typedef vector_view<TYPE> base_type;
    /// Size type definition coming from the base class
    typedef typename base_type::size_type size_type;
    /// Size pointer type definition coming from the base class
    typedef typename base_type::size_pointer size_pointer;
    /// Pointer type definition coming from the base class
    typedef typename base_type::pointer pointer;

    /// @name Checks on the type of the array element
    /// @{

    /// Make sure that the template type does not have a custom destructor
    static_assert(std::is_trivially_destructible<TYPE>::value,
                  "vecmem::data::vector_buffer can not handle types with "
                  "custom destructors");

    /// @}

    /// Constant size data constructor
    vector_buffer(size_type size, memory_resource& resource);
    /// Resizable data constructor
    vector_buffer(size_type capacity, size_type size,
                  memory_resource& resource);

private:
    /// Data object owning the allocated memory
    std::unique_ptr<char, details::deallocator> m_memory;

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
