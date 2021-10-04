/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */


#include "vecmem/memory/arena_memory_resource.hpp"
#include "vecmem/memory/memory_resource.hpp"

#include "details_arena.cpp"

namespace vecmem {

arena_memory_resource::arena_memory_resource(memory_resource& mm, std::size_t initial_size, std::size_t maximum_size) 
  : ar(arena(initial_size, maximum_size, mm)) {}


void* arena_memory_resource::do_allocate(std::size_t bytes, std::size_t) {
  void* ptr = ar.allocate(alignment::align_up(bytes, 8));
  return ptr;
}

void arena_memory_resource::do_deallocate(void* p, std::size_t bytes, std::size_t) {
  ar.deallocate(p, alignment::align_up(bytes, 8));
}

bool arena_memory_resource::do_is_equal(const memory_resource &other) const noexcept { 
  const arena_memory_resource *c;
  c = dynamic_cast<const arena_memory_resource *>(&other);

  return c != nullptr;
}

} // namespace vecmem