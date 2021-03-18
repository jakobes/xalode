#ifndef NOCELLMODEL_H
#define NOCELLMODEL_H

// stl
#include <cmath>
#include <iostream>
#include <memory.h>     // Enable_shared_from_this
#include <vector>

// local headers


class NoCellModel : public ODEBase
{
    public:
        typedef std::vector< double > vector_type;

        NoCellModel() {}

    std::shared_ptr< ODEBase > clone() const override
    {
        return std::make_shared< NoCellModel >(*this);
    }

    void operator() (const vector_type &x, vector_type &dxdt, const double /* t */) const override
    {
        dxdt[0] = 0;
        dxdt[1] = -x[1];
    }

    void eval(const vector_type &x, vector_type &dxdt, const double /* t */) const override
    {
        dxdt[0] = 0;
        dxdt[1] = -x[1];
    }
};

#endif
