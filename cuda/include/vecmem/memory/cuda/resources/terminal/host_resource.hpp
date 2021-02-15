#include <memory>
#include <memory_resource>

#include "vecmem/memory/resources/base_resource.hpp"

namespace vecmem::memory::resources::terminal {
    class cuda_host_resource : public base_resource {
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
            const std::pmr::memory_resource &
        ) const noexcept override;
    };
}