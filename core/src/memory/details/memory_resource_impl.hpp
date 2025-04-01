/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/utils/debug.hpp"

// System include(s).
#include <cassert>
#include <stdexcept>

/// Helper macro for implementing all standard constructors, copy/move
/// operators and functions for a memory resource that uses PIMPL.
///
/// @param[in] CLASSNAME The name of the class to implement the constructors
///                      and various functions for.
///
#define VECMEM_MEMORY_RESOURCE_PIMPL_IMPL(CLASSNAME)                        \
    CLASSNAME::CLASSNAME(CLASSNAME&&) noexcept = default;                   \
    CLASSNAME::~CLASSNAME() = default;                                      \
    CLASSNAME& CLASSNAME::operator=(CLASSNAME&&) noexcept = default;        \
    void* CLASSNAME::do_allocate(std::size_t size, std::size_t alignment) { \
        if (size == 0) {                                                    \
            throw std::bad_alloc();                                         \
        }                                                                   \
        assert(m_impl);                                                     \
        void* ptr = m_impl->allocate(size, alignment);                      \
        VECMEM_DEBUG_MSG(2, "Allocated %lu bytes at %p", size, ptr);        \
        return ptr;                                                         \
    }                                                                       \
    void CLASSNAME::do_deallocate(void* ptr, std::size_t size,              \
                                  std::size_t alignment) {                  \
        assert(ptr != nullptr);                                             \
        if (size == 0u) {                                                   \
            return;                                                         \
        }                                                                   \
        VECMEM_DEBUG_MSG(2, "De-allocating memory at %p", ptr);             \
        assert(m_impl);                                                     \
        m_impl->deallocate(ptr, size, alignment);                           \
    }
