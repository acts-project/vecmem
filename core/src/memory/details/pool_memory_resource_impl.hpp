/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/memory/pool_options.hpp"

// System include(s).
#include <functional>
#include <vector>

namespace vecmem::details {

/// Implementation of @c vecmem::pool_memory_resource
class pool_memory_resource_impl {

public:
    /// Constructor, on top of another memory resource
    pool_memory_resource_impl(memory_resource& upstream,
                              const pool_options& opts);

    /// Destructor, freeing all allocations
    ~pool_memory_resource_impl();

    /// Allocate memory
    void* allocate(std::size_t bytes, std::size_t alignment);

    /// Deallocate memory
    void deallocate(void* ptr, std::size_t bytes, std::size_t alignment);

private:
    /// The upstream memory resource
    std::reference_wrapper<memory_resource> m_upstream;
    /// The options for the pool memory resource
    pool_options m_options;

    struct block_descriptor {
        block_descriptor* next = nullptr;
    };

    struct chunk_descriptor {
        std::size_t size;
        chunk_descriptor* next = nullptr;
    };

    struct oversized_block_descriptor {
        std::size_t size;
        std::size_t alignment;
        oversized_block_descriptor* prev = nullptr;
        oversized_block_descriptor* next = nullptr;
        oversized_block_descriptor* next_cached = nullptr;
    };

    struct pool {
        block_descriptor* free_list = nullptr;
        std::size_t previous_allocated_count;
    };

    const std::size_t m_smallest_block_log2;

    std::vector<pool> m_pools;
    chunk_descriptor* m_allocated = nullptr;
    oversized_block_descriptor* m_oversized = nullptr;
    oversized_block_descriptor* m_cached_oversized = nullptr;

};  // class pool_memory_resource_impl

}  // namespace vecmem::details
