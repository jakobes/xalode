#ifndef ODEBASE_H
#define ODEBASE_H

// stl
#include <memory.h>     // Enable_shared_from_this


class ODEBase// : public std::enable_shared_from_this< ODEBase >
{
    public:
        template< class vector_type >
        void operator() (const vector_type &x, vector_type &dxdt, const double /* t */) { }
};


#endif
