/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// SYCL include(s).
#include <CL/sycl.hpp>

// System include(s)
#include <string>

namespace vecmem::sycl {

/// Device selector used by the VecMem SYCL library by default
class device_selector : public cl::sycl::device_selector {

    public:
    /// Constructor, with an optional device name
    device_selector(const std::string& deviceName = "");

    /// Operator used to "grade" the available devices
    int operator()(const cl::sycl::device& device) const override;

    private:
    /// Default device selector used internally
    cl::sycl::default_selector m_defaultSelector;

    /// The preferred device's name
    std::string m_deviceName;

};  // class device_selector

}  // namespace vecmem::sycl
