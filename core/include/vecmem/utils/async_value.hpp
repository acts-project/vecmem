/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/unique_ptr.hpp"
#include "vecmem/utils/abstract_event.hpp"

// Standard include(s).
#include <type_traits>

namespace vecmem {

/// Return type for asynchronous value retrievals
///
/// @tparam T Type of the value being retrieved asynchronously
///
template <typename T>
class async_value : public abstract_event {

public:
    /// Value type
    using value_type = T;
    /// Rvalue reference type
    using rvalue_ref = std::add_rvalue_reference_t<value_type>;
    /// Constant lvalue reference type
    using const_lvalue_ref =
        std::add_lvalue_reference_t<std::add_const_t<value_type>>;

    /// Constructor
    ///
    /// @param ref The value object to hold
    /// @param event Event to wait on before accessing the value
    ///
    async_value(rvalue_ref ref, std::unique_ptr<abstract_event> event);

    /// Access the async/future value
    ///
    /// @return Reference to the value
    ///
    const_lvalue_ref get() const;

    /// @name Function(s) implemented from @c vecmem::abstract_event
    /// @{

    /// Function that would block the current thread until the event is
    /// complete
    void wait() override;

    /// Function telling the object not to wait for the underlying event
    void ignore() override;

    /// @}

private:
    /// Unique allocation pointer to the (future) value
    value_type m_value;
    /// Event to wait on before accessing the value
    std::unique_ptr<abstract_event> m_event;

};  // class async_value

}  // namespace vecmem

// Include the implementation.
#include "vecmem/utils/impl/async_value.ipp"
