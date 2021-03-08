/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "vecmem/containers/device_vector_data.hpp"

namespace vecmem {

   /// Simple struct holding data for @c vecmem::const_device_vector
   template< typename T >
   using const_device_vector_data = device_vector_data< const T >;

} // namespace vecmem
