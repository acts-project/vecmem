/* VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021-2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "hip_error_handling.hpp"

// System include(s).
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace vecmem::hip {

/// Specific exception for HIP errors thrown by this library
struct runtime_error : public std::runtime_error {
    /// Inherit the base class's constructor(s)
    using std::runtime_error::runtime_error;
};

namespace details {

[[noreturn]] void throw_error(hipError_t errorCode, const char* expression,
                              const char* file, int line) {

    // Create a nice error message.
    std::ostringstream errorMsg;
    errorMsg << file << ":" << line << " Failed to execute: " << expression
             << " (" << hipGetErrorString(errorCode) << ")";

    // Now throw a runtime error with this message.
    throw runtime_error(errorMsg.str());
}

}  // namespace details
}  // namespace vecmem::hip
