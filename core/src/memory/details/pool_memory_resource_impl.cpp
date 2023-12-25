/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "pool_memory_resource_impl.hpp"

#include "../../utils/integer_math.hpp"
#include "vecmem/memory/details/is_aligned.hpp"
#include "vecmem/utils/debug.hpp"

// System include(s).
#include <cassert>
#include <sstream>
#include <stdexcept>

/// Helper macro for implementing the @c check_valid function
#define CHECK_VALID(EXP)                            \
    if (EXP) {                                      \
        std::ostringstream msg;                     \
        msg << __FILE__ << ":" << __LINE__          \
            << " Invalid pool option(s): " << #EXP; \
        throw std::invalid_argument(msg.str());     \
    }

namespace vecmem::details {
namespace {

/// Function checking whether a given set of options are valid/consistent
///
/// @param opts The options to check
///
void check_valid(const pool_options& opts) {

    CHECK_VALID(!vecmem::details::is_power_of_2(opts.smallest_block_size));
    CHECK_VALID(!vecmem::details::is_power_of_2(opts.largest_block_size));
    CHECK_VALID(!vecmem::details::is_power_of_2(opts.alignment));

    CHECK_VALID((opts.max_bytes_per_chunk == 0) ||
                (opts.max_blocks_per_chunk == 0));
    CHECK_VALID((opts.smallest_block_size == 0) ||
                (opts.largest_block_size == 0));

    CHECK_VALID(opts.min_blocks_per_chunk > opts.max_blocks_per_chunk);
    CHECK_VALID(opts.min_bytes_per_chunk > opts.max_bytes_per_chunk);

    CHECK_VALID(opts.smallest_block_size > opts.largest_block_size);

    CHECK_VALID((opts.min_blocks_per_chunk * opts.smallest_block_size) >
                opts.max_bytes_per_chunk);
    CHECK_VALID((opts.min_blocks_per_chunk * opts.largest_block_size) >
                opts.max_bytes_per_chunk);

    CHECK_VALID((opts.max_blocks_per_chunk * opts.largest_block_size) <
                opts.min_bytes_per_chunk);
    CHECK_VALID((opts.max_blocks_per_chunk * opts.smallest_block_size) <
                opts.min_bytes_per_chunk);

    CHECK_VALID(opts.alignment > opts.smallest_block_size);
}

}  // namespace

pool_memory_resource_impl::pool_memory_resource_impl(memory_resource& upstream,
                                                     const pool_options& opts)
    : m_upstream(upstream),
      m_options(opts),
      m_smallest_block_log2(
          vecmem::details::log2_ri(opts.smallest_block_size)) {

    check_valid(opts);
    pool p = {nullptr, 0};
    const std::size_t n = vecmem::details::log2_ri(opts.largest_block_size) -
                          m_smallest_block_log2 + 1;
    VECMEM_DEBUG_MSG(5, "Creating %lu pools", n);
    m_pools.resize(n, p);
}

pool_memory_resource_impl::~pool_memory_resource_impl() {

    // Deallocate memory allocated for the buckets.
    while (m_allocated != nullptr) {

        // Get the current upstream allocation chunk.
        chunk_descriptor* alloc = m_allocated;
        m_allocated = m_allocated->next;

        // Deallocate the chunk, with the appropriate pointer math.
        void* p = static_cast<void*>(
            static_cast<char*>(static_cast<void*>(alloc)) - alloc->size);
        m_upstream.get().deallocate(p, alloc->size + sizeof(chunk_descriptor),
                                    m_options.alignment);
    }

    // Deallocate cached oversized/overaligned memory.
    while (m_oversized != nullptr) {

        // Get the current oversized upstream allocation chunk.
        oversized_block_descriptor* alloc = m_oversized;
        m_oversized = m_oversized->next;

        // Deallocate the oversized chunk, with the appropriate pointer math.
        void* p = static_cast<void*>(
            static_cast<char*>(static_cast<void*>(alloc)) - alloc->size);
        m_upstream.get().deallocate(
            p, alloc->size + sizeof(oversized_block_descriptor),
            alloc->alignment);
    }
}

void* pool_memory_resource_impl::allocate(std::size_t bytes,
                                          std::size_t alignment) {

    // Tell the user what's happening.
    VECMEM_DEBUG_MSG(5,
                     "Requested the allocation of %lu bytes with alignment %lu",
                     bytes, alignment);

    // Decide on the size of the block to allocate.
    bytes = std::max(bytes, m_options.smallest_block_size);
    assert(vecmem::details::is_power_of_2(alignment));
    VECMEM_DEBUG_MSG(5, "Adjusted allocation size to %lu bytes", bytes);

    // An oversized and/or overaligned allocation requested; needs to be
    // allocated separately.
    if ((bytes > m_options.largest_block_size) ||
        (alignment > m_options.alignment)) {

        // Tell the user what's happening.
        VECMEM_DEBUG_MSG(5,
                         "Oversized and/or overaligned allocation being done");

        // If caching for oversized / overaligned allocations is enabled, try to
        // find a cached block that fits the request.
        if (m_options.cache_oversized) {
            oversized_block_descriptor* ptr = m_cached_oversized;
            oversized_block_descriptor** previous = &m_cached_oversized;
            while (ptr != nullptr) {
                oversized_block_descriptor desc = *ptr;
                bool is_good =
                    (desc.size >= bytes) && (desc.alignment >= alignment);

                // If the size is bigger than the requested size by a factor
                // bigger than or equal to the specified cutoff for size,
                // allocate a new block.
                if (is_good) {
                    const std::size_t size_factor = desc.size / bytes;
                    if (size_factor >= m_options.cached_size_cutoff_factor) {
                        is_good = false;
                    }
                }

                // If the alignment is bigger than the requested one by a factor
                // bigger than or equal to the specified cutoff for alignment,
                // allocate a new block.
                if (is_good) {
                    const std::size_t alignment_factor =
                        desc.alignment / alignment;
                    if (alignment_factor >=
                        m_options.cached_alignment_cutoff_factor) {
                        is_good = false;
                    }
                }

                // If we found a good block, remove it from the cache and return
                // it.
                if (is_good) {
                    if (previous != &m_cached_oversized) {
                        // If the block is not the first in the cache, update
                        // the previous block's next pointer.
                        oversized_block_descriptor previous_desc = **previous;
                        previous_desc.next_cached = desc.next_cached;
                        **previous = previous_desc;
                    } else {
                        // If the block is the first in the cache, update the
                        // cache's head pointer.
                        m_cached_oversized = desc.next_cached;
                    }

                    // Take this block out of the cache.
                    desc.next_cached = nullptr;
                    *ptr = desc;

                    // Tell the user what's happening.
                    VECMEM_DEBUG_MSG(5,
                                     "Found a cached block of size %lu and "
                                     "alignment %lu at %p",
                                     desc.size, desc.alignment, ptr);

                    // Return the allocated memory block to the user.
                    return static_cast<void*>(
                        static_cast<char*>(static_cast<void*>(ptr)) -
                        desc.size);
                }

                // Continue the search in the cache.
                previous = &(ptr->next_cached);
                ptr = *previous;
            }
        }

        // Tell the user what's happening.
        VECMEM_DEBUG_MSG(5, "Allocating a new oversized/overaligned block");

        // No fitting cached block was found; allocate a new one that's just up
        // to the specs.
        void* allocated = m_upstream.get().allocate(
            bytes + sizeof(oversized_block_descriptor), alignment);
        oversized_block_descriptor* block =
            static_cast<oversized_block_descriptor*>(
                static_cast<void*>(static_cast<char*>(allocated) + bytes));

        // Fill in the block descriptor.
        oversized_block_descriptor desc;
        desc.size = bytes;
        desc.alignment = alignment;
        desc.prev = nullptr;
        desc.next = m_oversized;
        desc.next_cached = nullptr;
        *block = desc;

        // Update the list of oversized blocks, to start with the new
        // allocation.
        m_oversized = block;

        // Update the previous pointer of the next block, if there is one.
        if (desc.next != nullptr) {
            oversized_block_descriptor next = *(desc.next);
            next.prev = block;
            *desc.next = next;
        }

        // Return the allocated memory block to the user.
        return allocated;
    }

    // The request is NOT for oversized and/or overaligned memory,
    // so allocate a block from an appropriate bucket.
    std::size_t bytes_log2 = vecmem::details::log2_ri(bytes);
    std::size_t bucket_idx = bytes_log2 - m_smallest_block_log2;
    pool& bucket = m_pools[bucket_idx];
    VECMEM_DEBUG_MSG(5, "Using pool under index %lu", bucket_idx);

    bytes = static_cast<std::size_t>(1) << bytes_log2;
    VECMEM_DEBUG_MSG(5, "Adjusted allocation size to %lu bytes", bytes);

    // If the free list of the bucket has no elements, allocate a new chunk
    // and split it into blocks pushed to the free list.
    if (bucket.free_list == nullptr) {
        std::size_t n = bucket.previous_allocated_count;
        if (n == 0) {
            n = m_options.min_blocks_per_chunk;
            if (n < (m_options.min_bytes_per_chunk >> bytes_log2)) {
                n = m_options.min_bytes_per_chunk >> bytes_log2;
            }
        } else {
            n = n * 3 / 2;
            if (n > (m_options.max_bytes_per_chunk >> bytes_log2)) {
                n = m_options.max_bytes_per_chunk >> bytes_log2;
            }
            if (n > m_options.max_blocks_per_chunk) {
                n = m_options.max_blocks_per_chunk;
            }
        }

        std::size_t descriptor_size =
            std::max(sizeof(block_descriptor), m_options.alignment);
        std::size_t block_size = bytes + descriptor_size;
        block_size += m_options.alignment - block_size % m_options.alignment;
        std::size_t chunk_size = block_size * n;

        void* allocated = m_upstream.get().allocate(
            chunk_size + sizeof(chunk_descriptor), m_options.alignment);
        chunk_descriptor* chunk = static_cast<chunk_descriptor*>(
            static_cast<void*>(static_cast<char*>(allocated) + chunk_size));

        chunk_descriptor chunk_desc;
        chunk_desc.size = chunk_size;
        chunk_desc.next = m_allocated;
        *chunk = chunk_desc;
        m_allocated = chunk;

        for (std::size_t i = 0; i < n; ++i) {
            block_descriptor* block =
                static_cast<block_descriptor*>(static_cast<void*>(
                    static_cast<char*>(allocated) + block_size * i + bytes));

            block_descriptor block_desc;
            block_desc.next = bucket.free_list;
            *block = block_desc;
            bucket.free_list = block;
        }
    }

    // Allocate a block from the front of the bucket's free list.
    block_descriptor* block = bucket.free_list;
    bucket.free_list = block->next;
    return static_cast<void*>(static_cast<char*>(static_cast<void*>(block)) -
                              bytes);
}

void pool_memory_resource_impl::deallocate(void* ptr, std::size_t bytes,
                                           std::size_t alignment) {

    bytes = std::max(bytes, m_options.smallest_block_size);
    assert(vecmem::details::is_power_of_2(alignment));
    assert(vecmem::details::is_aligned(ptr, alignment));

    // The deallocated block is oversized and/or overaligned.
    if ((bytes > m_options.largest_block_size) ||
        (alignment > m_options.alignment)) {

        oversized_block_descriptor* block =
            static_cast<oversized_block_descriptor*>(
                static_cast<void*>(static_cast<char*>(ptr) + bytes));

        oversized_block_descriptor desc = *block;

        if (m_options.cache_oversized) {
            desc.next_cached = m_cached_oversized;
            *block = desc;
            m_cached_oversized = block;

            return;
        }

        if (desc.prev == nullptr) {
            assert(m_oversized == block);
            m_oversized = desc.next;
        } else {
            oversized_block_descriptor prev = *desc.prev;
            assert(prev.next == block);
            prev.next = desc.next;
            *desc.prev = prev;
        }

        if (desc.next != nullptr) {
            oversized_block_descriptor next = *desc.next;
            assert(next.prev == block);
            next.prev = desc.prev;
            *desc.next = next;
        }

        m_upstream.get().deallocate(
            ptr, desc.size + sizeof(oversized_block_descriptor),
            desc.alignment);

        return;
    }

    // Push the block to the front of the appropriate bucket's free list.
    std::size_t n_log2 = vecmem::details::log2_ri(bytes);
    std::size_t bucket_idx = n_log2 - m_smallest_block_log2;
    pool& bucket = m_pools[bucket_idx];

    bytes = static_cast<std::size_t>(1) << n_log2;

    block_descriptor* block = static_cast<block_descriptor*>(
        static_cast<void*>(static_cast<char*>(ptr) + bytes));

    block_descriptor desc;
    desc.next = bucket.free_list;
    *block = desc;
    bucket.free_list = block;
}

}  // namespace vecmem::details
