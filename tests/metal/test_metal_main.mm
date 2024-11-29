/*
 * VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// GoogleTest include(s).
#include <gtest/gtest.h>

// Meta import(s).
#import <Metal/Metal.h>

int main(int argc, char** argv) {

    // Initialize GoogleTest.
    testing::InitGoogleTest(&argc, argv);

    // Check if a Metal device is available.
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        std::cout << "No Metal device was found" << std::endl;
        return 0;
    }

    // Run the tests.
    return RUN_ALL_TESTS();
}
