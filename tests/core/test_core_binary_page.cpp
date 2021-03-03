/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/allocators/allocator.hpp"
#include "vecmem/containers/vector.hpp"
#include "vecmem/memory/host_memory_resource.hpp"
#include "vecmem/memory/binary_page_memory_resource.hpp"
#include "vecmem/memory/resources/memory_resource.hpp"

// System include(s).
#include <algorithm>
#undef NDEBUG
#include <cassert>
#include <vector>


int main() {
   vecmem::host_memory_resource upstream;
   vecmem::binary_page_memory_resource resource(upstream);

   vecmem::vector<int> vec1(&resource);
   vecmem::vector<int> vec2(&resource);
   vecmem::vector<int> vec3(&resource);

   {
      vec1.reserve(20);

      assert(vec1.size() == 0);

      for (int i = 0; i < 20; ++i) {
         vec1.push_back(i * 2);
      }

      assert(vec1.size() == 20);

      for (int i = 20; i < 200; ++i) {
         vec1.push_back(i * 2);
      }

      assert(vec1.size() == 200);
   }

   {
      vec2.resize(1000);

      assert(vec2.size() == 1000);
   }

   {
      vec3.resize(2);

      assert(vec3.size() == 2);
   }

   return 0;
}
