#ifndef simpleode_H
#define simpleode

// stl
#include <cmath>
#include <iostream>
#include <memory.h>     // Enable_shared_from_this
#include <vector>

// local headers


class SimpleODE : public ODEBase
{
    public:
        typedef std::vector< double > vector_type;


        SimpleODE() {}

    std::shared_ptr< ODEBase > clone() const override
    {
        return std::make_shared< SimpleODE >(*this);
    }

    void operator() (const vector_type &x, vector_type &dxdt, const double /* t */) const override
    {
        dxdt[0] = x[1];
        dxdt[1] = x[0];
    }

    void eval(const vector_type &x, vector_type &dxdt, const double /* t */) const override
    {
        dxdt[0] = x[1];
        dxdt[1] = x[0];
    }
};

#endif
