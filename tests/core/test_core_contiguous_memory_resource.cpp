/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/memory/contiguous_memory_resource.hpp"

// System include(s).
#undef NDEBUG
#include <cassert>

int main() {
   vecmem::host_memory_resource upstream;
   vecmem::contiguous_memory_resource resource(upstream, 1048576);

   vecmem::vector<int> vec1(&resource);
   vecmem::vector<char> vec2(&resource);
   vecmem::vector<double> vec3(&resource);
   vecmem::vector<float> vec4(&resource);
   vecmem::vector<int> vec5(&resource);

   vec1.reserve(100);
   vec2.reserve(100);
   vec3.reserve(100);
   vec4.reserve(100);
   vec5.reserve(100);

   assert(static_cast<void *>(vec2.data()) == static_cast<void *>(static_cast<int *>(vec1.data()) + 100));
   assert(static_cast<void *>(vec3.data()) == static_cast<void *>(static_cast<char *>(vec2.data()) + 100));
   assert(static_cast<void *>(vec4.data()) == static_cast<void *>(static_cast<double *>(vec3.data()) + 100));
   assert(static_cast<void *>(vec5.data()) == static_cast<void *>(static_cast<float *>(vec4.data()) + 100));

   return 0;
}
