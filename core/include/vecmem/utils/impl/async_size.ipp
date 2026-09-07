/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System includes.
#include <cassert>

namespace vecmem {

template <typename SIZE_TYPE>
async_size<SIZE_TYPE>::async_size(storage_type size, event_type event)
    : m_size{std::move(size)}, m_event{std::move(event)} {}

template <typename SIZE_TYPE>
auto async_size<SIZE_TYPE>::get() const -> const_reference {

    // Wait for the event to complete before accessing the value
    assert(m_event != nullptr);
    m_event->wait();
    assert(m_size != nullptr);
    return (*m_size);
}

template <typename SIZE_TYPE>
auto async_size<SIZE_TYPE>::unsafe_get() const -> const_reference {

    // Access the value assuming the event is complete
    assert(m_event != nullptr);
    assert(m_event->is_ready());
    m_event->ignore();
    assert(m_size != nullptr);
    return (*m_size);
}

template <typename SIZE_TYPE>
void async_size<SIZE_TYPE>::wait() {

    assert(m_event != nullptr);
    m_event->wait();
}

template <typename SIZE_TYPE>
bool async_size<SIZE_TYPE>::is_ready() const {

    assert(m_event != nullptr);
    return m_event->is_ready();
}

template <typename SIZE_TYPE>
void async_size<SIZE_TYPE>::ignore() {

    assert(m_event != nullptr);
    m_event->ignore();
}

}  // namespace vecmem
