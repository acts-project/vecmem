/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/memory/details_arena.hpp"

namespace vecmem {

class arena_memory_resource : public memory_resource {
public:

  arena_memory_resource(memory_resource& mm, 
                        std::size_t initial_size, 
                        std::size_t maximum_size);

private:

  virtual void* do_allocate(std::size_t bytes, std::size_t) override;

  virtual void do_deallocate(void* p, std::size_t bytes, std::size_t) override;

  virtual bool do_is_equal(const memory_resource &other) const noexcept override;

  arena ar;
};

} // namespace vecmem