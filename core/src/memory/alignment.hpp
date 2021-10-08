/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "vecmem/memory/memory_resource.hpp"

// System include(s).
#include <cassert>

namespace vecmem {
namespace alignment {

inline static constexpr std::size_t NMM_DEFAULT_HOST_ALIGNMENT{
    alignof(std::max_align_t)};

inline constexpr bool is_pow_2(std::size_t n) {
    return (0 == (n & (n - 1)));
}

inline constexpr bool is_supported_alignment(std::size_t alignment) {
    return is_pow_2(alignment);
}

inline constexpr std::size_t align_up(std::size_t v,
                                      std::size_t align_bytes) noexcept {
    // if the alignment is not support, the program will end
    assert(is_supported_alignment(align_bytes));
    return (v + (align_bytes - 1)) & ~(align_bytes - 1);
}

inline constexpr std::size_t align_down(std::size_t v,
                                        std::size_t align_bytes) noexcept {
    // if the alignment is not support, the program will end
    assert(is_supported_alignment(align_bytes));
    return v & ~(align_bytes - 1);
}

inline constexpr bool is_aligned(std::size_t v,
                                 std::size_t align_bytes) noexcept {
    // if the alignment is not support, the program will end
    assert(is_supported_alignment(align_bytes));
    return v == align_down(v, align_bytes);
}

inline void *alignedAllocate(std::size_t bytes, std::size_t alignment,
                             memory_resource &mm) {
    assert(is_pow_2(alignment));

    std::size_t padded_allocation_size{bytes + alignment +
                                       sizeof(std::ptrdiff_t)};
    char *const original =
        static_cast<char *>(mm.allocate(padded_allocation_size));
    void *aligned{original + sizeof(std::ptrdiff_t)};
    std::align(alignment, bytes, aligned, padded_allocation_size);
    std::ptrdiff_t offset = static_cast<char *>(aligned) - original;

    *(static_cast<std::ptrdiff_t *>(aligned) - 1) = offset;

    return aligned;
}

inline void aligned_deallocate(void *p, std::size_t bytes,
                               std::size_t alignment, memory_resource &mm) {
    (void)alignment;

    std::ptrdiff_t const offset = *(reinterpret_cast<std::ptrdiff_t *>(p) - 1);

    void *const original = static_cast<char *>(p) - offset;

    // Mmmmmmm
    mm.deallocate(original, bytes);
}

}  // namespace alignment
}  // namespace vecmem
