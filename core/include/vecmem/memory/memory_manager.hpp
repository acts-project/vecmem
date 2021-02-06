/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <memory>

namespace vecmem {

   // Forward declaration(s).
   class memory_manager_interface;

   /// Singleton class pointing at the memory manager object to use
   class memory_manager {

   public:
      /// Destructor
      ~memory_manager();

      /// @name Declarations preventing any copies of the singleton object
      /// @{

      /// Disallow copy construction
      memory_manager( const memory_manager& ) = delete;
      /// Disallow move construction
      memory_manager( memory_manager&& ) = delete;

      /// Disallow copy assignment
      memory_manager& operator=( const memory_manager& ) = delete;
      /// Disallow move assignment
      memory_manager& operator=( memory_manager&& ) = delete;

      /// @}

      /// Singleton accessor
      static memory_manager& instance();

      /// @name Functions for setting and getting the global memory manager
      /// @{

      /// Set the memory manager that is to be used by the application
      void set( std::unique_ptr< memory_manager_interface > mgr );

      /// Get the memory manager that is used by the application
      memory_manager_interface& get() const;

      /// @}

   private:
      /// Constructor, hidden from the outside
      memory_manager();

      /// The memory manager instance used by the application
      std::unique_ptr< memory_manager_interface > m_mgr;

   }; // class memory_manager

} // namespace vecmem
