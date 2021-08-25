// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "details.hpp"
#include "vecmem/memory/memory_resource.hpp"

// System include(s)
#include <cstddef>
#include <utility>
#include <map>
#include <shared_mutex>
#include <thread>

namespace vecmem::cuda {

template <typename Upstream>
class arena_memory_resource : public memory_resource {
  public:
    explicit arena_memory_resource( Upstream* upstream_memory_resource, 
                                    std::size_t initial_size = global_arena<Upstream>::default_initial_size, 
                                    std::size_t maximum_size = global_arena<Upstream>::default_maximum_size)
                                    : global_arena_{upstream_memory_resource, initial_size, maximum_size}
                                  {}

    arena_memory_resource(arena_memory_resource const&) = delete;
    arena_memory_resource& operator=(arena_memory_resource const&) = delete;

    bool supports_streams() const noexcept override;

    bool supports_get_mem_info() const noexcept override;

  private:


    void* do_allocate(std::size_t bytes, cuda_stream_view stream) override;

    void do_deallocate(void* p, std::size_t bytes, cuda_stream_view stream) override;

    void deallocate_from_other_arena(void* p, std::size_t bytes, cuda_stream_view stream);

    arena<Upstream>& get_arena(cuda_stream_view stream);

    arena<Upstream>& get_thread_arena();

    arena<Upstream>& get_stream_arena(cuda_stream_view stream);

    std::pair<std::size_t, std::size_t> do_get_mem_info(cuda_stream_view stream) const override;

    static bool use_per_thread_arena(cuda_stream_view stream) {
		  return stream.is_per_thread_default();
		}

    global_arena<Upstream> global_arena_;
    std::map<std::thread::id, std::shared_ptr<arena<Upstream>>> thread_arenas_;
    std::map<cudaStream_t, arena<Upstream>> stream_arenas_;
    mutable std::shared_timed_mutex mtx_;
};// class arena_memory_resource



}