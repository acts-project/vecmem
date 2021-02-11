/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/utils/hip_error_handling.hpp"

// CUDA include(s).
#include <hip/hip_runtime.h>

// System include(s).
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace vecmem { namespace hip { namespace details {

   void throw_error( hipError_t errorCode, const char* expression,
                     const char* file, int line ) {

      // Create a nice error message.
      std::ostringstream errorMsg;
      errorMsg << file << ":" << line << " Failed to execute: " << expression
               << " (" << hipGetErrorString( errorCode ) << ")";

      // Now throw a runtime error with this message.
      throw std::runtime_error( errorMsg.str() );
   }

} } } // namespace vecmem::hip::details
