/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <vector>

namespace vecmem {

   /// Vector type to use "on the host"
   ///
   /// Hiding the allocator template argument from the user. Which is necessary
   /// in the template types of this project that use a selectable vector type.
   ///
   template< typename T >
   using host_vector = std::vector< T >;

} // namespace vecmem
