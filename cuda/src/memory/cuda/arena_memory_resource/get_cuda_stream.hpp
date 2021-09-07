// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// Local include(s)
#include "vecmem/memory/cuda/arena_memory_resource/cuda_stream_view.hpp"

// Cuda include(s)
#include <cuda_runtime_api.h>

namespace vecmem::cuda {
namespace arena_details {

//functions to convert the class cuda_stream to cudaStream_t*
cudaStream_t& get_cuda_stream(cuda_stream_view& stream);

const cudaStream_t& get_cuda_stream(const cuda_stream_view& stream);

} // namespace arena_details
} // namespace vecmem::cuda