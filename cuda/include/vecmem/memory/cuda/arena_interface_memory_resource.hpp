// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/memory/cuda/device_memory_resource.hpp"
#include "arena_memory_resource/arena_memory_resource.hpp"

namespace vecmem::cuda {

// This class was made to managed the allocations from generic allocations
// to the allocations for the arena memory resource
class arena_interface_memory_resource : public memory_resource {

  public:
    arena_interface_memory_resource();

  private:
    virtual void* do_allocate(std::size_t, std::size_t) override;

    virtual void do_deallocate(void* p, std::size_t, std::size_t) override;

    virtual bool do_is_equal(const memory_resource&) const noexcept override;

    arena_details::arena_memory_resource<device_memory_resource>* arena_ptr;
};

}