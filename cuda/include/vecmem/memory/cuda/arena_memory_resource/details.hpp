// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "vecmem/utils/cuda/cuda_stream_view.hpp"

// CUDA include(s).
#include <cuda_runtime_api.h>

// System include(s).
#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

namespace vecmem::cuda {

// Minimum size of a Superblock (256 KiB)
//constexpr std::size_t minimum_superblock_size = 1u << 18u;

class block {

  public:
    // construct a default block.
    block() = default;

    // construct a block given a pointer and size.
    // 
    // @param[in] pointer the address for the beginning of the block.
    // @param[in] size the size of the block
    block(char* pointer, std::size_t size);

    // construct a block given a pointer and size.
    // 
    // @param[in] pointer the address for the beginning of the block.
    // @param[in] size the size of the block
    block(void* pointer, std::size_t size);

    // returns the underlying pointer
    void* pointer() const;

    // returns the size of the block
    std::size_t size() const;

    // returns true if this block is valid (non-null), false otherwise
    bool is_valid() const;

    // returns true if this block is a Superblock, false otherwise
    bool is_superblock() const;

    // verifies wheter this block can be merged to the beginning of block b
    //
    // @param[in] b the block to check for contiguity
    // @return true if this block's `pointer` + `size` == `b.ptr`, and `not b.isHead`,
    // false otherwise 
    bool is_contiguous_before(block const& b) const;

    // is this block large enough to fit that size of bytes?
    //
    // @param[in] size_of_bytes the size in bytes to check for fit
    // @return true if this block is at least size_of_bytes
    bool fits(std::size_t size_of_bytes) const;

    // split this block into two by the given size
    //
    // @param[in] size the size in bytes of the first block
    // @return std::pair<block, block> a pair of blocks split by size
    std::pair<block, block> split(std::size_t size) const;

    // coalesce two contiguos blocks into one, this->is_contiguous_before(b) 
    // must be true
    // 
    // @param[in] b block to merge
    // @return block the merged block
    block merge(block const& b) const;

    // used by std::set to compare blocks
    bool operator<(block const& b) const;

  private:
    char* pointer_;      // raw memory pointer
    std::size_t size_{}; // size in bytes
}; // class block

// the required allocation alignment
//constexpr std::size_t allocation_alignment = 256;

// align up to the allocation alignment
//
// @param[in] value value to align
// @return the aligned value
constexpr std::size_t align_up(std::size_t value) noexcept;

// align down to the allocation alignment
//
// @param[in] value value to align
// @return the aligned value
constexpr std::size_t align_down(std::size_t value) noexcept;

// get the first free block of at least `size` bytes
//
// @param[in] free_blocks the adress-ordered set of free blocks
// @param[in] size the number of bytes to allocate
// @return a block of memory of at least `size` bytes, or an empty if
// not found
inline block first_fit(std::set<block>& free_blocks, std::size_t size);

// coalesce the given block with other free blocks
//
// @param[in] free_blocks the address-ordered set of free blocks.
// @param b the block to coalesce.
// @return the coalesced block.
inline block coalesce_block(std::set<block>& free_blocks, block const&b);

template <typename Upstream>
class global_arena final {
  public:
    // default initial size for the global arena
    static constexpr std::size_t default_initial_size = std::numeric_limits<std::size_t>::max();
    // default maximum size for the global arena
    static constexpr std::size_t default_maximum_size = std::numeric_limits<std::size_t>::max();
    // reserved memory that should not be allocated (64 MiB)
    static constexpr std::size_t reserverd_size = 1u << 26u;

    // Disable copy (and move) semantics
    global_arena(const global_arena&) = delete;
    global_arena& operator=(const global_arena&) = delete;

    // construct a global arena
    //
    // @param[in] upstream_memory_resource the memory resource from which to allocate 
    // blocks for the pool
    // @param[in] initialSize minimum size, in bytes, of the initial global arena. 
    // Defaults to all the available memory on the current device.
    // @param[in] maximum_size maximum size, in bytes, that the global arena can grow to. 
    // Defaults to all of the available memory on the current device
    global_arena(Upstream* upstream_memory_resource, std::size_t initialSize, std::size_t maximum_size);

    // Destroy the gloabl arean and deallocate all memory using the upstream resource
    ~global_arena();
    // Allocates memory of size at least `size_of_bytes`
    //
    // @param[in] sizeOfBytes the size in bytes of the allocation
    // @retyrb block pointer to the newly allocated memory
    block allocate(std::size_t size_of_bytes);

    // Deallocate memory pointer to by the block b
    // 
    // @param[in] b pointer of block to be deallocated
    void deallocate(block const& b);
    // Deallocate memory of a set of blocks
    //
    // @param[in] free_blocks set of block to be free
    void deallocate(std::set<block> const& free_blocks);

  private:
    using lock_guard = std::lock_guard<std::mutex>;

    // Get an available memory block of at least `size` bytes
    //
    // @param[in] size the number of bytes to allocate
    // @return a block of memory of at least `size` bytes
    block get_block(std::size_t size);

    // Get the size to grow the global arena given the requested `size` bytes
    //
    // @param[in] size the number of bytes required
    // @return the size for the arena to grow
    constexpr std::size_t size_to_grow(std::size_t size) const;

    // Allocate space from upstream to supply the arena and return a sufficiently 
    // sized block.
    //
    // @param[in] size the minimum size to allocate
    // @return a bock of at least `size` bytes
    block expand_arena(std::size_t size);

    // The upstream resource to allocate memory from
    Upstream* upstream_memory_resource_;
    // The maximum size of the global arena
    std::size_t maximum_size_;
    // The current size of the global arena
    std::size_t current_size_{};
    // Address-ordered set of free blocks
    std::set<block> free_blocks_;
    // blocks allocated from upstream so that they can be quickly freed
    std::vector<block> upstream_blocks_;
    // Mutex for exclusive lock
    mutable std::mutex mtx_;
};// class global_arena


// An arena for allocating memory for a thread
// An arena is a per-thread or per-non-default-stream memory pool. It allocates
// superblocks from the global arena, and return them when the superblocks become empty
//
// @tparam Upstream Memory resource to use for allocating the global arena. Implements
// MemoryResource::DeviceMemoryResource interface
template <typename Upstream>
class arena {
  public:
    // Construct an `arena`
    //
    // @param[in] global_arena the global arena from withc to allocate superblocks
    explicit arena(global_arena<Upstream>& global_arena);

    // Allocates memory of size at least `bytes`
    //
    // @param[in] bytes the size in bytes of the allocation
    // @return void* pointer to the newly allocated memory
    void* allocate(std::size_t bytes);

    // Deallocate memory pointed to by `p`, and possibly return superblocks to upstream.
    // return the block to the set that have the free blocks
    //
    // @param[in] p the pointer of the memory
    // @param[in] bytes the size in bytes of the deallocation
    // @return if the allocation was found, false otherwise
    bool deallocate(void* p, std::size_t bytes, cuda_stream_view stream);

    // Deallocate memory pointed to by `p`, and keeping all free superblocks.
    // return the block to the set that have the free blocks. This method
    // is used when deallocating from another arena, since we don't have access the
    // respective cuda_stream_view of the areana.
    //
    // @param[in] p the pointer of the memory
    // @param[in] bytes the size in bytes of the deallocation
    // @return if the allocation was found, false otherwise
    bool deallocate(void* p, std::size_t bytes);

    // Clean the arena and deallocate free blocks from the global arena.
      //
      // This is only needed when a per-thread arena is about to die.
    void clean();

  private:
    using lock_guard = std::lock_guard<std::mutex>;

    // @brief Get an available memory block of at least `size` bytes.
    //
    // @param[in] size The number of bytes to allocate.
    // @return block A block of memory of at least `size` bytes.
    block get_block(std::size_t size);

    // Allocate space from upstream to supply the arena and return a superblock.
    // 
    // @return block A superblock.
    block expand_arena(std::size_t size);

    // Finds, frees and returns the block associated with pointer `p`.
    //
    // @param[in] p The pointer to the memory to free.
    // @param[in] size The size of the memory to free. Must be equal to the original allocation size.
    // return The (now freed) block associated with `p`. The caller is expected to return the block
    // to the arena.
    block free_block(void* p, std::size_t size) noexcept;

    // Shrink this arena by returning free superblocks to upstream.
    //
    // @param b The block that can be used to shrink the arena.
    // @param stream Stream on which to perform shrinking.
    void shrink_arena(block const& b, cuda_stream_view stream);

    global_arena<Upstream>& global_arena_;
    std::set<block> free_blocks_;
    std::unordered_map<void*, block> allocated_blocks_;
    mutable std::mutex mtx_;
};// class arena

template <typename Upstream>
class arena_cleaner {
 public:
  explicit arena_cleaner(std::shared_ptr<arena<Upstream>> const& a);

  // Disable copy (and move) semantics.
  arena_cleaner(const arena_cleaner&) = delete;
  arena_cleaner& operator=(const arena_cleaner&) = delete;

  ~arena_cleaner();

 private:
  /// A non-owning pointer to the arena that may need cleaning.
  std::weak_ptr<arena<Upstream>> arena_;
};// class arena_cleaner


}