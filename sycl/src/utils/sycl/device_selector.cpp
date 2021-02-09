/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "vecmem/utils/sycl/device_selector.hpp"

namespace vecmem::sycl {

   device_selector::device_selector( const std::string& deviceName )
   : m_defaultSelector(), m_deviceName( deviceName ) {

   }

   int device_selector::operator()( const cl::sycl::device& device ) const {

      // Under no circumstances do we accept any NVidia OpenCL devices.
      const std::string vendor =
         device.get_info<cl::sycl::info::device::vendor>();
      const std::string version =
         device.get_info<cl::sycl::info::device::version>();
      if( ( vendor.find( "NVIDIA" ) != std::string::npos ) &&
          ( version.find( "OpenCL" ) != std::string::npos ) ) {
         return -1;
      }

      // If the user provided a substring of the device name, look for that
      // device. And give it a very high score.
      using info = cl::sycl::info::device;
      if( ( ! m_deviceName.empty() ) &&
          ( device.get_info< info::name >() == m_deviceName ) ) {
         return 1000;
      }

      // But even if a particular device was requested, but not found, still
      // return something at least...
      return m_defaultSelector( device );
   }

} // namespace vecmem::sycl
