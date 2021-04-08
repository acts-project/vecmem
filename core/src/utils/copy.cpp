/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// VecMem include(s).
#include "vecmem/utils/copy.hpp"
#include "vecmem/utils/debug.hpp"

// System include(s).
#include <cstring>

namespace vecmem {

   void copy::do_copy( std::size_t size, const void* from, void* to,
                       type::copy_type ) {

      // Perform a simple POSIX memory copy.
      memcpy( to, from, size );

      // Let the user know what happened.
      VECMEM_DEBUG_MSG( 4, "Performed POSIX memory copy of %lu bytes from %p "
                        "to %p", size, from, to );
   }

} // namespace vecmem
