// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "vecmem/memory/cuda/arena_interface_memory_resource.hpp"
#include "arena_memory_resource/arena_memory_resource.cpp"
#include "vecmem/memory/cuda/device_memory_resource.hpp"

namespace vecmem::cuda {

arena_interface_memory_resource::arena_interface_memory_resource() {
  device_memory_resource* upstream = new device_memory_resource();
  arena_ptr = new arena_details::arena_memory_resource<device_memory_resource>(upstream, 
                                                      std::numeric_limits<std::size_t>::max(), 
                                                      std::numeric_limits<std::size_t>::max());;
}

void* arena_interface_memory_resource::do_allocate(std::size_t bytes, std::size_t) {
  return (*arena_ptr).allocate(bytes);
}

void arena_interface_memory_resource::do_deallocate(void* p, std::size_t bytes, std::size_t) {
  (*arena_ptr).deallocate(p, bytes);
}

bool arena_interface_memory_resource::do_is_equal(const memory_resource &other) const noexcept {
  return (*arena_ptr).is_equal(other);
}

}