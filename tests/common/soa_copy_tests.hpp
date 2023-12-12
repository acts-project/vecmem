/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Project include(s).
#include "vecmem/memory/memory_resource.hpp"
#include "vecmem/utils/copy.hpp"

// Local include(s).
#include "jagged_soa_container.hpp"
#include "simple_soa_container.hpp"

// GoogleTest include(s).
#include <gtest/gtest.h>

// System include(s).
#include <tuple>

/// Parameter for the copy tests
using soa_copy_test_parameters =
    std::tuple<vecmem::memory_resource&, vecmem::memory_resource&,
               vecmem::copy&, vecmem::copy&>;

/// Test fixture for the SoA copy tests
template <typename CONTAINER>
class soa_copy_tests_base
    : public ::testing::TestWithParam<soa_copy_test_parameters> {

protected:
    /// Test the simple/direct host->fixed device->host copy
    static void host_to_fixed_device_to_host_direct(
        const soa_copy_test_parameters& params);
    /// Test the "optimal" host->fixed device->host copy
    static void host_to_fixed_device_to_host_optimal(
        const soa_copy_test_parameters& params);
    /// Test the host->resizable device->host copy
    static void host_to_resizable_device_to_host(
        const soa_copy_test_parameters& params);
    /// Test the host->fixed device->resizable device->host copy
    static void host_to_fixed_device_to_resizable_device_to_host(
        const soa_copy_test_parameters& params);

};  // class soa_copy_tests_base

/// Parametrized copy tests for the "simple" SoA container
using soa_copy_tests_simple =
    soa_copy_tests_base<vecmem::testing::simple_soa_container>;

/// Parametrized copy tests for the "jagged" SoA container
using soa_copy_tests_jagged =
    soa_copy_tests_base<vecmem::testing::jagged_soa_container>;

// Include the implementation.
#include "soa_copy_tests.ipp"
