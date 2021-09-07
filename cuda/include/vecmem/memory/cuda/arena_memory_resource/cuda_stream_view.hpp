// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// CUDA include(s).
//#include "cuda.h"
//#include "cuda_runtime.h"

// System include(s).
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace vecmem::cuda {
namespace arena_details {

// forward declaration of cuda_stream, to not have cuda dependency
class cuda_stream;

class cuda_stream_view {
  public:
    cuda_stream_view()                        = default;
    cuda_stream_view(cuda_stream_view const&) = default;
    cuda_stream_view(cuda_stream_view&&)      = default;
    cuda_stream_view& operator=(cuda_stream_view const&) = default;
    cuda_stream_view& operator=(cuda_stream_view&&) = default;
    ~cuda_stream_view()                                       = default;

    // Implicit conversion from cudaStream_t
    cuda_stream_view(void* stream);

    // Returns the wrppped stream
    //constexpr cuda_stream value() const noexcept;

    // Explicit conversion to cudaStream_t
    //explicit constexpr operator cuda_stream() const noexcept;

    // Return true if the wrapped stream is the CUDA per-thread default stream
    bool is_per_thread_default() const noexcept;

    // Return true if the wrapped stream is explicitly the CUDA legacy default stream
    bool is_default() const noexcept;

    // Synchronize the viewed CUDA stream
    void synchronize() const;

    // Synchronize the viewed CUDA stream, don't throw if there is an error
    /*void synchronize_no_throw() const noexcept {
    	ACTS_CUDA_ERROR_CHECK(cudaStreamSynchronize(stream_));
    }*/
    
    void* stream();

    const void* stream() const;
  private:
    void* ptr_cuda_stream;
    //std::unique_ptr<cuda_stream> stream_;
}; // class cuda_stream_view

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

} // namespace arena_details
} // namespace vecmem::cuda