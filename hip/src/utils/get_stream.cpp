/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "get_stream.hpp"

// System include(s).
#include <cassert>

namespace vecmem::hip::details {

hipStream_t get_stream(const stream_wrapper& stream) {

    assert(stream.stream() != nullptr);
    return static_cast<hipStream_t>(stream.stream());
}

}  // namespace vecmem::hip::details
