/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem {

template <typename SIZE_TYPE>
async_size<SIZE_TYPE>::async_size(storage_type size, event_type event)
    : m_size{std::move(size)}, m_event{std::move(event)} {}

template <typename SIZE_TYPE>
auto async_size<SIZE_TYPE>::get() const -> const_reference {

    // Wait for the event to complete before accessing the value
    m_event->wait();
    return (*m_size);
}

template <typename SIZE_TYPE>
void async_size<SIZE_TYPE>::wait() {

    m_event->wait();
}

template <typename SIZE_TYPE>
void async_size<SIZE_TYPE>::ignore() {

    m_event->ignore();
}

}  // namespace vecmem
