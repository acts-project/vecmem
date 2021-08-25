// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// CUDA plugin include(s).
#include "details.cpp"
#include "vecmem/memory/cuda/arena_memory_resource/arena_memory_resource.hpp"

namespace vecmem::cuda {

struct CudaDeviceId {
    using value_type = int;

    explicit constexpr CudaDeviceId(value_type id) noexcept : id_{id} {}

    constexpr value_type value() const noexcept { return id_; }

  private:
    value_type id_;
};

template <typename Upstream>
bool arena_memory_resource<Upstream>::supports_streams() const noexcept { return true; }

template <typename Upstream>
bool arena_memory_resource<Upstream>::supports_get_mem_info() const noexcept { return false; }

template <typename Upstream>
void* arena_memory_resource<Upstream>::do_allocate(std::size_t bytes, cuda_stream_view stream) {
  if(bytes <= 0) return nullptr;

  bytes = align_up(bytes);
  return this->get_arena(stream).allocate(bytes);
}

template <typename Upstream>
void arena_memory_resource<Upstream>::do_deallocate(void* p, std::size_t bytes, cuda_stream_view stream) {
  if(p == nullptr || bytes <= 0) return;

  bytes = align_up(bytes);
  if(!this->get_arena(stream).deallocate(p, bytes, stream)) {
    deallocate_from_other_arena(p, bytes, stream);
  }
}

template <typename Upstream>
void arena_memory_resource<Upstream>::deallocate_from_other_arena(void* p, std::size_t bytes, cuda_stream_view stream) {
  stream.synchronize();

  std::shared_lock<std::shared_timed_mutex> lock(mtx_);

  if(this->use_per_thread_arena(stream)) {
    auto const id = std::this_thread::get_id();
    for(auto& kv : thread_arenas_) {
      if(kv.first != id && kv.second->deallocate(p, bytes)) return;
    }
  } else {
    for(auto& kv : stream_arenas_){
      if(stream != kv.first && kv.second.deallocate(p, bytes)) return;
    }
  }
  global_arena_.deallocate({p, bytes});
}

template <typename Upstream>
arena<Upstream>& arena_memory_resource<Upstream>::get_arena(cuda_stream_view stream) {
  if(this->use_per_thread_arena(stream)){
    return this->get_thread_arena();
  } else {
    return this->get_stream_arena(stream);
  }
}

template <typename Upstream>
arena<Upstream>& arena_memory_resource<Upstream>::get_thread_arena() {
  auto const id = std::this_thread::get_id();

  std::shared_lock<std::shared_timed_mutex> lockRead(mtx_);
  auto const it = thread_arenas_.find(id);
  if(it != thread_arenas_.end()) {
    return *it->second;
  }

  std::lock_guard<std::shared_timed_mutex> lock_write(mtx_);
  auto a = std::make_shared<arena>(global_arena_);
  thread_arenas_.emplace(id, a);
  thread_local arena_cleaner<Upstream> cleaner{a};
  return *a;
}

template <typename Upstream>
arena<Upstream>& arena_memory_resource<Upstream>::get_stream_arena(cuda_stream_view stream) {
  //if(use_per_thread_arena(stream)){

    std::shared_lock<std::shared_timed_mutex> lockRead(mtx_);
    auto const it = stream_arenas_.find(stream.value());
    if (it != stream_arenas_.end()) { return it->second; }

    std::lock_guard<std::shared_timed_mutex> lock_write(mtx_);
    stream_arenas_.emplace(stream.value(), global_arena_);
    return stream_arenas_.at(stream.value());
  //}
}

template <typename Upstream>
std::pair<std::size_t, std::size_t> arena_memory_resource<Upstream>::do_get_mem_info(cuda_stream_view stream) const {
  return std::make_pair(0, 0);
}

}