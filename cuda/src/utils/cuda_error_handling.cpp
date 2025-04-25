/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "cuda_error_handling.hpp"

// System include(s).
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace vecmem::cuda {

/// Specific exception for CUDA errors thrown by this library
struct runtime_error : public std::runtime_error {
    /// Inherit the base class's constructor(s)
    using std::runtime_error::runtime_error;
};

namespace details {

[[noreturn]] void throw_error(cudaError_t errorCode, const char* expression,
                              const char* file, int line) {

    // Create a nice error message.
    std::ostringstream errorMsg;
    errorMsg << file << ":" << line << " Failed to execute: " << expression
             << " (" << cudaGetErrorString(errorCode) << ")";

    // Now throw a runtime error with this message.
    throw runtime_error(errorMsg.str());
}

}  // namespace details
}  // namespace vecmem::cuda
