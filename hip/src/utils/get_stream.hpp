/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/utils/hip/stream_wrapper.hpp"

// HIP include(s).
#include <hip/hip_runtime_api.h>

namespace vecmem::hip::details {

/// Get a concrete @c hipStream_t object out of our wrapper
hipStream_t get_stream(const stream_wrapper& stream);

}  // namespace vecmem::hip::details
