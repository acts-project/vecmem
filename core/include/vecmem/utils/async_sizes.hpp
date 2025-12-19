/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/vector.hpp"
#include "vecmem/utils/abstract_event.hpp"

// System include(s).
#include <memory>
#include <type_traits>

namespace vecmem {

/// Return type for asynchronous size vector retrievals
///
/// @tparam SIZE_TYPE Type of the size(s) being retrieved asynchronously
///
template <typename SIZE_TYPE>
class async_sizes : public abstract_event {

public:
    /// Size type used
    using size_type = SIZE_TYPE;
    /// Underlying type that stores the size variables on the heap
    using storage_type = vector<size_type>;
    /// Constant (lvalue) reference to the stored size
    using const_reference =
        std::add_lvalue_reference_t<std::add_const_t<storage_type>>;
    /// Type of the held event
    using event_type = std::unique_ptr<abstract_event>;

    /// Constructor taking ownership of a size and event
    ///
    /// @param sizea The vector holding the size variables
    /// @param event Event to wait on before accessing the size
    ///
    async_sizes(storage_type&& sizes, event_type event);

    /// Access the async/future value
    ///
    /// @return Reference to the value
    ///
    const_reference get() const;

    /// @name Function(s) implemented from @c vecmem::abstract_event
    /// @{

    /// Function that would block the current thread until the event is
    /// complete
    void wait() override;

    /// Function telling the object not to wait for the underlying event
    void ignore() override;

    /// @}

private:
    /// Size storage
    storage_type m_sizes;
    /// Underlying event
    event_type m_event;

};  // class async_sizes

}  // namespace vecmem

// Include the implementation.
#include "vecmem/utils/impl/async_sizes.ipp"
