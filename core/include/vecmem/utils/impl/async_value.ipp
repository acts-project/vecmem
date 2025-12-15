/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem {

template <typename T>
async_value<T>::async_value(rvalue_ref ref,
                            std::unique_ptr<abstract_event> event)
    : m_value(std::move(ref)), m_event(std::move(event)) {}

template <typename T>
auto async_value<T>::get() const -> const_lvalue_ref {
    // Wait for the event to complete before accessing the value
    m_event->wait();
    return m_value;
}

template <typename T>
void async_value<T>::wait() {
    m_event->wait();
}

template <typename T>
void async_value<T>::ignore() {
    m_event->ignore();
}

}  // namespace vecmem
