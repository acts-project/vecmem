/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

namespace vecmem {
namespace details {

VECMEM_HOST_AND_DEVICE
inline void memmove(void* dest, const void* src, std::size_t bytes) {

    // Check for some trivial cases.
    if ((dest == src) || (bytes == 0)) {
        return;
    }

    // Cast the char pointers.
    char* dest_char = static_cast<char*>(dest);
    const char* src_char = static_cast<const char*>(src);

    if ((dest_char < src_char) || (dest_char >= (src_char + bytes))) {
        // Non-overlapping, or overlapping such that a forward copy does the
        // correct thing.
        for (std::size_t i = 0; i < bytes; ++i) {
            dest_char[i] = src_char[i];
        }
    } else {
        // Overlapping such that a backward copy would do the correct thing.
        for (std::size_t i = bytes; i > 0; --i) {
            dest_char[i - 1] = src_char[i - 1];
        }
    }
}

}  // namespace details
}  // namespace vecmem
