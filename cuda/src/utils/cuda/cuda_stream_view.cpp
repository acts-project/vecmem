// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// VecMem include(s).
#include "vecmem/utils/cuda/cuda_stream_view.hpp"

namespace vecmem::cuda {

static constexpr cuda_stream_view cuda_stream_default{};

static cuda_stream_view cuda_stream_legacy{cuda_stream_legacy};

static cuda_stream_view cuda_stream_per_thread{cuda_stream_per_thread};

inline bool operator==(cuda_stream_view lhs, cuda_stream_view rhs) {
  return lhs.value() == rhs.value();
}

inline bool operator!=(cuda_stream_view lhs, cuda_stream_view rhs) { 
  return !(lhs == rhs); 
}

inline std::ostream& operator<<(std::ostream& os, cuda_stream_view sv) {
  os << sv.value();
  return os;
}

constexpr cuda_stream_view::cuda_stream_view(cudaStream_t stream) : stream_{stream} {}

constexpr cudaStream_t cuda_stream_view::value() const noexcept {
  return stream_;
}

/*
// Explicit conversion to cudaStream_t
explicit constexpr operator cuda_stream_view::cudaStream_t() const noexcept {
  return value();
}
*/

bool cuda_stream_view::is_per_thread_default() const noexcept {
  #ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    return value() == cuda_stream_per_thread || value() == 0;
  #else
    return value() == cuda_stream_per_thread;
  #endif
}

bool cuda_stream_view::is_default() const noexcept {
  #ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    return value() == cuda_stream_legacy;
  #else
    return value() == cuda_stream_legacy || value() == 0;
  #endif
}

void cuda_stream_view::synchronize() const {
  //ACTS_CUDA_ERROR_CHECK(cudaStreamSynchronize(stream_));
  cudaStreamSynchronize(stream_);
}

// Synchronize the viewed CUDA stream, don't throw if there is an error
/*
void synchronize_no_throw() const noexcept {
	ACTS_CUDA_ERROR_CHECK(cudaStreamSynchronize(stream_));
}
*/

}