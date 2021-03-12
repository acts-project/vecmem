/** VecMem project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <iterator>
#include <type_traits>

namespace vecmem { namespace details {

   /// Helper trait for identifying input iterators
   ///
   /// It comes in handy in some of the functions of the custom (device)
   /// container types that use templated iterator values. Which could hide
   /// overloads of the same function with the same number of (non-templated)
   /// arguments.
   ///
   /// The implementation is *very* simplistic at the moment. It could/should
   /// be made more elaborate when the need arises.
   ///
   template< typename iterator_type, typename value_type >
   using is_iterator_of =
      std::is_convertible<
         typename std::iterator_traits< iterator_type >::value_type,
         value_type >;

} } // namespace vecmem::details
