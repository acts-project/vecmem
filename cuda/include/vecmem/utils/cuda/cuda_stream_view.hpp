// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// CUDA include(s).
#include "cuda.h"
#include "cuda_runtime.h"

// System include(s).
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace vecmem::cuda {

class cuda_stream_view {
  public:
    constexpr cuda_stream_view()                        = default;
    constexpr cuda_stream_view(cuda_stream_view const&) = default;
    constexpr cuda_stream_view(cuda_stream_view&&)      = default;
    constexpr cuda_stream_view& operator=(cuda_stream_view const&) = default;
    constexpr cuda_stream_view& operator=(cuda_stream_view&&) = default;
    ~cuda_stream_view()                                       = default;

    // Implicit conversion from cudaStream_t
    constexpr cuda_stream_view(cudaStream_t stream);

    // Returns the wrppped stream
    constexpr cudaStream_t value() const noexcept;

    // Explicit conversion to cudaStream_t
    explicit constexpr operator cudaStream_t() const noexcept {
  	  return value();
  	}

    // Return true if the wrapped stream is the CUDA per-thread default stream
    bool is_per_thread_default() const noexcept;

    // Return true if the wrapped stream is explicitly the CUDA legacy default stream
    bool is_default() const noexcept;

    // Synchronize the viewed CUDA stream
    void synchronize() const;

    // Synchronize the viewed CUDA stream, don't throw if there is an error
    /*
    void synchronize_no_throw() const noexcept {
    	ACTS_CUDA_ERROR_CHECK(cudaStreamSynchronize(stream_));
    }
    */
  private:
    cudaStream_t stream_{0};
}; // class cuda_stream_view

// Static cuda_stream_view of the default stream (stream 0), for convenience
//static constexpr cuda_stream_view cuda_stream_default{};

// Static cuda_stream_view of cudaStreamLegacy, for convenience
//static cuda_stream_view cuda_stream_legacy{cudaStreamLegacy};

// Static cuda_stream_view of cudaStreamPerThread, for convenience
//static cuda_stream_view cuda_stream_per_thread{cudaStreamPerThread};

// Equality ciomparison operator for streams
// 
// @param[in] lhs the first stream view to compare
// @param[in] rhs the second stream view to compare
// @return true if equal, false if unequal
inline bool operator==(cuda_stream_view lhs, cuda_stream_view rhs);

// Inequality comparison operator for streams
//
// @param[in] lhs the first stream view to compare
// @param[in] rhs the second stream view to compare
// @return true if unequal, false if equal
inline bool operator!=(cuda_stream_view lhs, cuda_stream_view rhs);

// Output stream operator for printing / logging streams
//
// @param[in] os the output ostream
// @param[in] sv the cuda_stream_view to output
// @return std::ostream& the output ostream
inline std::ostream& operator<<(std::ostream& os, cuda_stream_view sv);

}