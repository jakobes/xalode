#ifndef FITZHUGH_H
#define FITZHUGH

// stl
#include <cmath>
#include <iostream>
#include <memory.h>     // Enable_shared_from_this
#include <vector>

// local headers
/* #include "odebase.h" */


/* class Fitzhugh : public ODEBase, public std::enable_shared_from_this< Fitzhugh > */
class Fitzhugh : public ODEBase
{
    public:
        typedef std::vector< double > vector_type;


        Fitzhugh(
                double a = 0.13,
                double b = 13.0,
                double c1 = 0.26,
                double c2 = 10.0,
                double c3 = 1.0,
                double v_rest = -70.0,
                double v_peak = 40) :
            a(a), b(b), c1(c1), c2(c2), c3(c3), v_rest(v_rest), v_peak(v_peak)
    {
        v_amp = v_peak - v_rest;
        v_th = v_rest + a*v_amp;
    }

    std::shared_ptr< ODEBase > clone() const override
    {
        return std::make_shared< Fitzhugh >(*this);
    }

    void operator() (const vector_type &x, vector_type &dxdt, const double /* t */) const override
    {
        using namespace std;

        dxdt[0] = c1/(pow(v_amp, 2))*(x[0] - v_rest)*(x[0] - v_th)*(v_peak - x[0]) - c2*x[1];
        dxdt[1] = b*(x[0] - v_rest - c3*x[1]);
    }

    void eval(const vector_type &x, vector_type &dxdt, const double /* t */) const override
    {
        using namespace std;

        dxdt[0] = c1/(pow(v_amp, 2))*(x[0] - v_rest)*(x[0] - v_th)*(v_peak - x[0]) - c2*x[1];
        dxdt[1] = b*(x[0] - v_rest - c3*x[1]);
    }

    virtual void print() const override
    {
        std::cout << "Bar" << std::endl;
    }

    private:
        double a;
        double b;
        double c1;
        double c2;
        double c3;
        double v_rest;
        double v_peak;
        double v_amp;
        double v_th;
};


#endif
