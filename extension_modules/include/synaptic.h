#ifndef SYNAPTIC_H
#define SYNAPTIC_H

// stl
#include <cmath>
#include <iostream>
#include <memory.h>     // Enable_shared_from_this
#include <vector>


class Synaptic : public ODEBase
{
    public:
        typedef std::vector< double > vector_type;

        Synaptic() {}

    std::shared_ptr< ODEBase > clone() const override
    {
        return std::make_shared< Synaptic >(*this);
    }

    void operator() (const vector_type &x, vector_type &dxdt, const double /* t */) const override
    {
        using namespace std;
        double g_syn = 1/(1 + exp(-0.062*x[0])/3.57);
        int spike = int(x[0] > V_threshold);

        dxdt[0] = -g_syn*x[1]*(x[0] - V_syn);
        dxdt[1] = -1/tau_s*x[1] + alpha_s*x[2]*(1 - x[1]);
        dxdt[2] = -1/tau_x*x[2] + spike;
    }

    void eval(const vector_type &x, vector_type &dxdt, const double /* t */) const override
    {
        using namespace std;
        double g_syn = 1/(1 + exp(-0.062*x[0])/3.57);
        int spike = int(x[0] > V_threshold);

        dxdt[0] = -g_syn*x[1]*(x[0] - V_syn);
        dxdt[1] = -1/tau_s*x[1] + alpha_s*x[2]*(1 - x[1]);
        dxdt[2] = -1/tau_x*x[2] + spike;
    }

    private:
        double tau_s = 100; // ms
        double tau_x = 2;   // ms
        double alpha_s = 0.5;   // kHz
        double V_threshold = 10;        // Specific value should not matter much
        double V_syn = 0;
};

#endif
