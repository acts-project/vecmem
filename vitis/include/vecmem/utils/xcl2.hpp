/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#pragma once

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define OCL_TARGET_OPENCL_VERSION 120

// OCL_CHECK doesn't work if call has templatized function call
#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

#include <CL/cl.hpp>
#include <CL/cl_ext_xilinx.h>
#include "vecmem/vecmem_vitis_export.hpp"

#include <fstream>
#include <iostream>
// When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the
// hood
// User ptr is used if and only if it is properly aligned (page aligned). When
// not
// aligned, runtime has no choice but to create its own host side buffer that
// backs
// user ptr. This in turn implies that all operations that move data to and from
// device incur an extra memcpy to move data to/from runtime's own host buffer
// from/to user pointer. So it is recommended to use this allocator if user wish
// to
// Create Buffer/Memory Object with CL_MEM_USE_HOST_PTR to align user buffer to
// the
// page boundary. It will ensure that user buffer will be used when user create
// Buffer/Mem Object with CL_MEM_USE_HOST_PTR.
template <typename T>
struct aligned_allocator {
    using value_type = T;

    aligned_allocator() {}

    aligned_allocator(const aligned_allocator&) {}

    template <typename U>
    aligned_allocator(const aligned_allocator<U>&) {}

    T* allocate(std::size_t num) {
        void* ptr = nullptr;

        {
            if (posix_memalign(&ptr, 4096, num * sizeof(T))) throw std::bad_alloc();
        }
        return reinterpret_cast<T*>(ptr);
    }
    void deallocate(T* p, std::size_t num) {
        free(p);
    }
};

namespace xcl {
VECMEM_VITIS_EXPORT
std::vector<cl::Device> get_xil_devices();

VECMEM_VITIS_EXPORT
std::vector<cl::Device> get_devices(const std::string& vendor_name);

VECMEM_VITIS_EXPORT
cl::Device find_device_bdf(const std::vector<cl::Device>& devices, const std::string& bdf);

VECMEM_VITIS_EXPORT
cl_device_id find_device_bdf_c(cl_device_id* devices, const std::string& bdf, cl_uint dev_count);

VECMEM_VITIS_EXPORT
std::string convert_size(size_t size);

VECMEM_VITIS_EXPORT
std::vector<unsigned char> read_binary_file(const std::string& xclbin_file_name);

VECMEM_VITIS_EXPORT
bool is_emulation();

VECMEM_VITIS_EXPORT
bool is_hw_emulation();
}

