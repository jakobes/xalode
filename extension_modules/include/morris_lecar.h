#ifndef MORRISLECAR_H
#define MORRISLECAR_H

// stl
#include <cmath>
#include <iostream>
#include <memory.h>     // Enable_shared_from_this
#include <vector>

class MorrisLecar: public ODEBase
{
    public:
        typedef std::vector< double > vector_type;

        MorrisLecar(double Iext = 40.0) : Iext(Iext) {}

        std::shared_ptr< ODEBase > clone() const override
        {
            return std::make_shared< MorrisLecar >(*this);
        }

        void eval(const vector_type &x, vector_type &dxdt, const double /* t */) const override
        {
            using namespace std;
            const double Minf = (1 + tanh((x[0] - V1)/V2))/2;
            const double Ninf = (1 + tanh((x[0] - V3)/V4))/2;
            const double tau = phi*cosh((x[0] - V3)/(2*V4));

            dxdt[0] = Cm*(Iext - gl*(x[0] - Vl) - gca*Minf*(x[0] - VCa) - gk*x[1]*(x[0] - Vk));
            dxdt[0] = tau*(Ninf - x[1]);
        }

    private:
        double V1 = -1.2;
        double V2 = 18.0;
        double V3 = 10.0;
        double V4 = 17.4;
        double phi = 1.0/15;
        double gl = 2.0;
        double gca = 4.0;
        double gk = 8.0;
        double Cm = 0.2;

        double Vl = -60;
        double VCa = 120;
        double Vk = -80;

        double Iext;
};

#endif
