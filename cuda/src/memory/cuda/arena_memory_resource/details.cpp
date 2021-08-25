// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "../../../utils/cuda/aligment.cpp"
#include "../../../utils/cuda/cuda_stream_view.cpp"
#include "vecmem/memory/cuda/arena_memory_resource/details.hpp"

namespace vecmem::cuda {

constexpr std::size_t minimum_superblock_size = 1u << 18u;

block::block(char* pointer, std::size_t size) : pointer_(pointer), size_(size) {}

block::block(void* pointer, std::size_t size) : pointer_(static_cast<char*>(pointer)), size_(size){}

void* block::pointer() const { return this->pointer_; }

std::size_t block::size() const { return this->size_; }

bool block::is_valid() const { return this->pointer_ != nullptr; }

bool block::is_superblock() const { return this->size_ >= minimum_superblock_size; }

bool block::is_contiguous_before(block const& b) const { return this->pointer_ + this->size_ == b.pointer_; }

bool block::fits(std::size_t size_of_bytes) const { return this->size_ >= size_of_bytes; }

std::pair<block, block> block::split(std::size_t size) const {
  //assert condition of size_ >= size
  if(this->size_ > size) {
    return {{this->pointer_, size}, {this->pointer_ + size, this->size_ - size}};
  } else {
    return {*this, {}};
  }
}

block block::merge(block const& b) const {
	//assert condition is_contiguous_before(b)
	return {this->pointer_, this->size_ + b.size_};
}

bool block::operator<(block const& b) const { return this->pointer_ < b.pointer_; }

constexpr std::size_t allocation_alignment = 256;

constexpr std::size_t align_up(std::size_t value) noexcept {
  return align_up(value, allocation_alignment);
} 

constexpr std::size_t align_down(std::size_t value) noexcept {
  return align_down(value, allocation_alignment);
}

inline block first_fit(std::set<block>& free_blocks, std::size_t size){
  auto const iter = std::find_if(free_blocks.cbegin(), free_blocks.cend(), [size](auto const& b) { return b.fits(size); });

  if(iter == free_blocks.cend()) {
    return {};
  } else {
    // remove the block from the freeList
    auto const b = *iter;
    auto const i = free_blocks.erase(iter);

    if(b.size() > size) {
      // split the block and put the remainder back.
      auto const split = b.split(size);
      free_blocks.insert(i, split.second);
      return split.first;
    } else {
      // b.size == size then return b
      return b;
    }
  }
}

inline block coalesce_block(std::set<block>& free_blocks, block const&b){
  // return the given block in case is not valid
  if(!b.is_valid()) return b;

  // find the right place (in ascending address order) to insert the block
  auto const next = free_blocks.lower_bound(b);
  auto const previous = next == free_blocks.cend() ? next : std::prev(next);

  // coalesce with neighboring blocks
  bool const merge_prev = previous->is_contiguous_before(b);
  bool const merge_next = next != free_blocks.cend() && b.is_contiguous_before(*next);

  block merged{};
  if(merge_prev && merge_next) {
    // if can merge with prev and next neighbors
    merged = previous->merge(b).merge(*next);

    free_blocks.erase(previous);

    auto const i = free_blocks.erase(next);
    free_blocks.insert(i, merged);
  } else if(merge_prev) {
    // if only can merge with prev neighbor
    merged = previous->merge(b);

    auto const i = free_blocks.erase(next);
    free_blocks.insert(i, merged);
  } else if(merge_next) {
    // if only can merge with next neighbor
    merged = b.merge(*next);

    auto const i = free_blocks.erase(next);
    free_blocks.insert(i, merged);
  } else {
    // if can't be merge with either
    free_blocks.emplace(b);
    merged = b;
  }

  return merged;
}

template <typename Upstream>
global_arena<Upstream>::global_arena(Upstream* upstream_memory_resource, std::size_t initial_size, std::size_t maximum_size)
  : upstream_memory_resource_{upstream_memory_resource}, maximum_size_{maximum_size} {
    // assert unexpected null upstream pointer
    // assert initial arena size required to be a multiple of 256 bytes
    // assert maximum arena size required to be a multiple of 256 bytes

    if(initial_size == default_initial_size || maximum_size == default_maximum_size) {
      std::size_t free{}, total{};
      cudaMemGetInfo(&free, &total);
      if(initial_size == default_initial_size) {
        initial_size = align_up(std::min(free, total / 2));
      }
      if(maximum_size == default_maximum_size) {
        this->maximum_size_ = align_down(free) - reserverd_size;
      }
    }
    // initial size exceeds the maxium pool size
    this->free_blocks_.emplace(expand_arena(initial_size));
}

template <typename Upstream>
global_arena<Upstream>::~global_arena() {
  lock_guard lock(this->mtx_);
  for(auto const& b : this->upstream_blocks_) {
    this->upstream_memory_resource_->deallocate(b.pointer(), b.size());
  }
}

template <typename Upstream>
block global_arena<Upstream>::allocate(std::size_t size_of_bytes) {
  lock_guard lock(this->mtx_);
  return get_block(size_of_bytes);
}

template <typename Upstream>
void global_arena<Upstream>::deallocate(block const& b) {
  lock_guard lock(this->mtx_);
  coalesce_block(this->free_blocks_, b);
}

template <typename Upstream>
void global_arena<Upstream>::deallocate(std::set<block> const& free_blocks) {
  lock_guard lock(this->mtx_);
  for(auto const& b : free_blocks) {
    coalesce_block(this->free_blocks_, b);
  }
}

template <typename Upstream>
block global_arena<Upstream>::get_block(std::size_t size) {
  auto const b = first_fit(this->free_blocks_, size);
  if(b.is_valid()) return b;

  auto const upstreamblock = expand_arena(size_to_grow(size));
  coalesce_block(this->free_blocks_, upstreamblock);
  return first_fit(this->free_blocks_, size);
}

template <typename Upstream>
constexpr std::size_t global_arena<Upstream>::size_to_grow(std::size_t size) const {
  /*
  case if maximum pool size exceeded
  if(this->currentSize_ + size > this->maximum_size_) {

  }
  */
  return this->maximum_size_ - this->currentSize_;
}

template <typename Upstream>
block global_arena<Upstream>::expand_arena(std::size_t size) {
  this->upstream_blocks_.push_back({this->upstream_memory_resource_->allocate(size), size});
  this->currentSize_ += size;
  return this->upstream_blocks_.back();
}

template <typename Upstream>
arena<Upstream>::arena(global_arena<Upstream>& global_arena) : global_arena_{global_arena} {}

template <typename Upstream>
void* arena<Upstream>::allocate(std::size_t bytes) {
  lock_guard lock(this->mtx_);

  auto const b = get_block(bytes);
  this->allocated_blocks_.emplace(b.pointer(), b);

  return b.pointer();
}

template <typename Upstream>
bool arena<Upstream>::deallocate(void* p, std::size_t bytes, cuda_stream_view stream) {
  lock_guard lock(this->mtx_);

  auto const b = arena<Upstream>::free_block(p, bytes);
  if(b.is_valid()) {
    auto const merged = coalesce_block(this->free_blocks_, b);
    shrink_arena(merged, stream);
  }

  return b.is_valid();
}

template <typename Upstream>
bool arena<Upstream>::deallocate(void* p, std::size_t bytes) {
  lock_guard lock(this->mtx_);

  auto const b = arena<Upstream>::free_block(p, bytes);
  if(b.is_valid()) {
    this->global_arena_.deallocate(b);
  }

  return b.is_valid();
}

template <typename Upstream>
void arena<Upstream>::clean() {
  lock_guard lock(this->mtx_);
  this->global_arena_.deallocate(this->free_blocks_);
  this->free_blocks_.clear();
  this->allocated_blocks_.clear();
}

template <typename Upstream>
block arena<Upstream>::get_block(std::size_t size) {
  if(size < minimum_superblock_size) {
    auto const b = first_fit(this->free_blocks_, size);
    if(b.is_valid()) {
      return b;
    }
  }

  auto const superblock = expand_arena(size);
  coalesce_block(this->free_blocks_, superblock);
  return first_fit(this->free_blocks_, size);
}

template <typename Upstream>
block arena<Upstream>::expand_arena(std::size_t size) {
  auto const superblock_size = std::max(size, minimum_superblock_size);
  return this->global_arena_.allocate(superblock_size);
}

template <typename Upstream>
block arena<Upstream>::free_block(void* p, std::size_t size) noexcept {
  auto const i = this->allocated_blocks_.find(p);

  if(i == this->allocated_blocks_.end()) {
    return {};
  }

  auto const found = i->second;
  //assert if found.size == size
  this->allocated_blocks_.erase(i);

  return found;
}

template <typename Upstream>
void arena<Upstream>::shrink_arena(block const& b, cuda_stream_view stream) {
  if(!b.is_superblock()) return;

  stream.synchronize();

  this->global_arena_.deallocate(b);
  this->free_blocks_.erase(b);
}

template <typename Upstream>
arena_cleaner<Upstream>::arena_cleaner(std::shared_ptr<arena<Upstream>> const& a) : arena_(a) {}

template <typename Upstream>
arena_cleaner<Upstream>::~arena_cleaner() {
  if (!this->arena_.expired()) {
    auto arena_ptr = this->arena_.lock();
    arena_ptr->clean();
  }
}

}