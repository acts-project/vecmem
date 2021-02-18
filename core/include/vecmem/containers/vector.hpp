/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

#include "vecmem/memory/resources/memory_resource.hpp"

#include <vector>

namespace vecmem {
   /**
    * @brief Alias type for vectors with our polymorphic allocator
    *
    * This type serves as an alias for a common type pattern, namely a
    * host-accessible vector with a memory resource which is not known at
    * compile time, which could be host memory or shared memory.
    *
    * @warning This type should only be used with host-accessible memory
    * resources.
    */
   template<typename T>
   using vector = std::vector<T, vecmem::polymorphic_allocator<T>>;
}
