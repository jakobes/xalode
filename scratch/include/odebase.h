#ifndef ODEBASE_H
#define ODEBASE_H

// stl
#include <memory.h>     // Enable_shared_from_this

#include <cassert.h>


class ODEBase : public std::enable_shared_from_this< ODEBase >
{
    public:
        // TODO: Use virtual!
        template< class vector_type >
        void operator() (const vector_type &x, vector_type &dxdt, const double /* t */) {
            assert(false);
        }
};


#endif
