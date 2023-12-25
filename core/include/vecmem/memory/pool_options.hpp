/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <cstddef>

namespace vecmem {

/// Runtime options for @c pool_memory_resource
struct pool_options {

    /// The minimal number of blocks, i.e. pieces of memory handed off to the
    /// user from a pool of a given size, in a single chunk allocated from
    /// upstream.
    std::size_t min_blocks_per_chunk = 16;
    /// The minimal number of bytes in a single chunk allocated from upstream.
    std::size_t min_bytes_per_chunk = 1024;
    /// The maximal number of blocks, i.e. pieces of memory handed off to the
    /// user from a pool of a given size, in a single chunk allocated from
    /// upstream.
    std::size_t max_blocks_per_chunk = static_cast<std::size_t>(1) << 20;
    /// The maximal number of bytes in a single chunk allocated from upstream.
    std::size_t max_bytes_per_chunk = static_cast<std::size_t>(1) << 30;

    /// The size of blocks in the smallest pool covered by the pool resource.
    /// All allocation requests below this size will be rounded up to this size.
    std::size_t smallest_block_size = alignof(std::max_align_t);
    /// The size of blocks in the largest pool covered by the pool resource. All
    /// allocation requests above this size will be considered oversized,
    /// allocated directly from upstream (and not from a pool), and cached only
    /// if @c cache_oversized is @c true.
    std::size_t largest_block_size = static_cast<std::size_t>(1) << 20;

    /// The alignment of all blocks in internal pools of the pool resource. All
    /// allocation requests above this alignment will be considered oversized,
    /// allocated directly from upstream (and not from a pool), and cached only
    /// if @c cache_oversized is @c true.
    std::size_t alignment = alignof(std::max_align_t);

    /// Decides whether oversized and overaligned blocks are cached for later
    /// use, or immediately return it to the upstream resource.
    bool cache_oversized = true;

    /// The size factor at which a cached allocation is considered too
    /// ridiculously oversized to use to fulfill an allocation request. For
    /// instance: the user requests an allocation of size 1024 bytes. A block of
    /// size 32 * 1024 bytes is cached. If @c cached_size_cutoff_factor is 32 or
    /// less, this block will be considered too big for that allocation request.
    std::size_t cached_size_cutoff_factor = 16;
    /// The alignment factor at which a cached allocation is considered too
    /// ridiculously overaligned to use to fulfill an allocation request. For
    /// instance: the user requests an allocation aligned to 32 bytes. A block
    /// aligned to 1024 bytes is cached. If @c cached_size_cutoff_factor is 32
    /// or less, this block will be considered too overaligned for that
    /// allocation request.
    std::size_t cached_alignment_cutoff_factor = 16;

};  // struct options

}  // namespace vecmem
