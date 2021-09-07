// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Local include(s).
#include "vecmem/memory/cuda/arena_memory_resource/cuda_stream_view.hpp"
#include "get_cuda_stream.cpp"

namespace vecmem::cuda {
namespace arena_details {

static constexpr cuda_stream_view cuda_stream_default{};

static cuda_stream_view cuda_stream_legacy{cudaStreamLegacy};

static cuda_stream_view cuda_stream_per_thread{cudaStreamPerThread};

inline bool operator==(cuda_stream_view lhs, cuda_stream_view rhs) {
  return get_cuda_stream(lhs) == get_cuda_stream(rhs);
}

inline bool operator!=(cuda_stream_view lhs, cuda_stream_view rhs) { 
  return !(lhs == rhs); 
}

inline std::ostream& operator<<(std::ostream& os, cuda_stream_view sv) {
  os << get_cuda_stream(sv);
  return os;
}

cuda_stream_view::cuda_stream_view(void* stream) : ptr_cuda_stream(stream) {}

bool cuda_stream_view::is_per_thread_default() const noexcept {
  #ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    return get_cuda_stream(*this) == get_cuda_stream(cuda_stream_per_thread) || get_cuda_stream(*this) == 0;
  #else
    return get_cuda_stream(*this) == get_cuda_stream(cuda_stream_per_thread);
  #endif
}

bool cuda_stream_view::is_default() const noexcept {
  #ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    return get_cuda_stream(*this) == get_cuda_stream(cuda_stream_legacy);
  #else
    return get_cuda_stream(*this) == get_cuda_stream(cuda_stream_legacy) || get_cuda_stream(*this) == 0;
  #endif
}

void cuda_stream_view::synchronize() const {
  //ACTS_CUDA_ERROR_CHECK(cudaStreamSynchronize(stream_));
  cudaStreamSynchronize(get_cuda_stream(*this));
}

void* cuda_stream_view::stream() {
  return ptr_cuda_stream;
}

const void* cuda_stream_view::stream() const {
  return ptr_cuda_stream;
}

} // namespace arena_details
} // namespace vecmem::cuda