/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "vecmem/memory/details_arena.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include "../utils/alignment.cpp"
#include "vecmem/memory/memory_resource.hpp"

namespace vecmem {

block::block(void* pointer, std::size_t size)
    : pointer_(static_cast<char*>(pointer)), size_(size){};

void* block::pointer() const {
    return this->pointer_;
};

std::size_t block::size() const {
    return this->size_;
};

bool block::is_valid() const {
    return this->pointer_ != nullptr;
};

bool block::is_superblock() const {
    return this->size_ >= minimum_superblock_size;
}

bool block::is_contiguous_before(block const& b) const {
    return this->pointer_ + this->size_ == b.pointer_;
}

bool block::fits(std::size_t size_of_bytes) const {
    return this->size_ >= size_of_bytes;
}

std::pair<block, block> block::split(std::size_t size) const {
    // assert condition of size_ >= size
    if (this->size_ > size) {
        return {{this->pointer_, size},
                {this->pointer_ + size, this->size_ - size}};
    } else {
        return {*this, {}};
    }
}

block block::merge(block const& b) const {
    // assert condition is_contiguous_before(b)
    return {this->pointer_, this->size_ + b.size_};
}

bool block::operator<(block const& b) const {
    return this->pointer_ < b.pointer_;
}

constexpr std::size_t align_up(std::size_t value) noexcept {
    return alignment::align_up(value, allocation_alignment);
}

constexpr std::size_t align_down(std::size_t value) noexcept {
    return alignment::align_down(value, allocation_alignment);
}

inline block first_fit(std::set<block>& free_blocks, std::size_t size) {
    auto const iter =
        std::find_if(free_blocks.cbegin(), free_blocks.cend(),
                     [size](auto const& b) { return b.fits(size); });

    if (iter == free_blocks.cend()) {
        return {};
    } else {
        // remove the block from the freeList
        auto const b = *iter;
        auto const i = free_blocks.erase(iter);

        if (b.size() > size) {
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

inline block coalesce_block(std::set<block>& free_blocks, block const& b) {
    // return the given block in case is not valid
    if (!b.is_valid())
        return b;

    // find the right place (in ascending address order) to insert the block
    auto const next = free_blocks.lower_bound(b);
    auto const previous = next == free_blocks.cend() ? next : std::prev(next);

    // coalesce with neighboring blocks
    bool const merge_prev = previous->is_contiguous_before(b);
    bool const merge_next =
        next != free_blocks.cend() && b.is_contiguous_before(*next);

    block merged{};
    if (merge_prev && merge_next) {
        // if can merge with prev and next neighbors
        merged = previous->merge(b).merge(*next);

        free_blocks.erase(previous);

        auto const i = free_blocks.erase(next);
        free_blocks.insert(i, merged);
    } else if (merge_prev) {
        // if only can merge with prev neighbor
        merged = previous->merge(b);

        auto const i = free_blocks.erase(next);
        free_blocks.insert(i, merged);
    } else if (merge_next) {
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

arena::arena(std::size_t initial_size, std::size_t maximum_size,
             memory_resource& mm)
    : maximum_size_{maximum_size}, mm_(mm) {
    // assert unexpected null upstream pointer
    // assert initial arena size required to be a multiple of 256 bytes
    // assert maximum arena size required to be a multiple of 256 bytes

    if (initial_size == default_initial_size ||
        maximum_size == default_maximum_size) {
        if (initial_size == default_initial_size) {
            initial_size = align_up(initial_size / 2);
        }
        if (maximum_size == default_maximum_size) {
            this->maximum_size_ = default_maximum_size - reserverd_size;
        }
    }
    // initial size exceeds the maxium pool size
    this->free_blocks_.emplace(expand_arena(initial_size));
}

arena::~arena() {
    for (auto b : free_blocks_) {
        void* p = b.pointer();
        std::size_t size = b.size();
        mm_.deallocate(p, size);
    }

    free_blocks_.clear();

    for (auto itr : allocated_blocks_) {
        void* p = itr.first;
        block b = itr.second;

        mm_.deallocate(p, b.size());
    }

    allocated_blocks_.clear();
}

void* arena::allocate(std::size_t bytes) {

    auto const b = get_block(bytes);
    this->allocated_blocks_.emplace(b.pointer(), b);

    return b.pointer();
}

bool arena::deallocate(void* p, std::size_t bytes) {

    auto const b = free_block(p, bytes);
    if (b.is_valid()) {
        coalesce_block(free_blocks_, b);
    }

    return b.is_valid();
}

block arena::get_block(std::size_t size) {
    if (size < minimum_superblock_size) {
        auto const b = first_fit(this->free_blocks_, size);
        if (b.is_valid()) {
            return b;
        }
    }

    auto const superblock = expand_arena(size);
    coalesce_block(this->free_blocks_, superblock);
    return first_fit(this->free_blocks_, size);
}

constexpr std::size_t arena::size_to_grow(std::size_t size) const {
    /*
    case if maximum pool size exceeded
    if(this->current_size_ + size > this->maximum_size_) {
    }
    */
    return this->maximum_size_ - this->current_size_;
}

block arena::expand_arena(std::size_t size) {
    std::pair<std::set<block>::iterator, bool> ret =
        free_blocks_.insert({mm_.allocate(size), size});
    current_size_ += size;
    return *(ret.first);
}

block arena::free_block(void* p, std::size_t size) noexcept {
    auto const i = this->allocated_blocks_.find(p);

    if (i == this->allocated_blocks_.end()) {
        return {};
    }

    auto const found = i->second;
    // assert if found.size == size
    this->allocated_blocks_.erase(i);

    return found;
}

}  // namespace vecmem