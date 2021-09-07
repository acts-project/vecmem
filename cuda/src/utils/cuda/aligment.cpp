// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// VecMem include(s).
#include "vecmem/utils/cuda/aligment.hpp"

namespace vecmem::cuda {

static constexpr std::size_t NMM_DEFAULT_HOST_ALIGNMENT{alignof(std::max_align_t)};

constexpr bool is_pow_2(std::size_t n) { return (0 == (n & (n - 1)));} 

constexpr bool is_supported_alignment(std::size_t alignment) { return is_pow_2(alignment);}

constexpr std::size_t align_up(std::size_t v, std::size_t align_bytes) noexcept {
  //if the alignment is not support, the program will end
  assert(is_supported_alignment(align_bytes));
  return (v + (align_bytes - 1)) & ~(align_bytes - 1);
}

constexpr std::size_t align_down(std::size_t v, std::size_t align_bytes) noexcept {
  //if the alignment is not support, the program will end
  assert(is_supported_alignment(align_bytes));
  return v & ~(align_bytes - 1);
}

constexpr bool is_aligned(std::size_t v, std::size_t align_bytes) noexcept {
  //if the alignment is not support, the program will end
  assert(is_supported_alignment(align_bytes));
  return v == align_down(v, align_bytes);
}

template <typename Alloc>
void *alignedAllocate(std::size_t bytes, std::size_t alignment, Alloc alloc) {
  assert(is_pow_2(alignment));

  std::size_t padded_allocation_size{bytes + alignment + sizeof(std::ptrdiff_t)};
  char *const original = static_cast<char *>(alloc(padded_allocation_size));
  void *aligned{original + sizeof(std::ptrdiff_t)};
  std::align(alignment, bytes, aligned, padded_allocation_size);
  std::ptrdiff_t offset = static_cast<char *>(aligned) - original;

  *(static_cast<std::ptrdiff_t *>(aligned) - 1) = offset;

  return aligned;
}

template <typename Dealloc>
void aligned_deallocate(void *p, std::size_t bytes, std::size_t alignment, Dealloc dealloc){
  (void) alignment;

  std::ptrdiff_t const offset = *(reinterpret_cast<std::ptrdiff_t *>(p) - 1);

  void *const original = static_cast<char *>(p) - offset;

  dealloc(original);
}

}