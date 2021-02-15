#pragma once

#include <memory>
#include <cstdlib>

#include "vecmem/memory/resources/memory_resource.hpp"
#include "vecmem/memory/resources/base_resource.hpp"

namespace vecmem::memory::resources::terminal {
    class malloc_resource : public base_resource {
    public:
        virtual bool is_host_accessible() const override;

    private:
        virtual void * do_allocate(
            std::size_t,
            std::size_t
        ) override;

        virtual void do_deallocate(
            void * p,
            std::size_t,
            std::size_t
        ) override;

        virtual bool do_is_equal(
            const memory_resource &
        ) const noexcept override;
    };
}