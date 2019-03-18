#ifndef COMPILE_EXT_H
#define COMPILE_EXT_H


// pybind headers
#include <pybind11/pybind11.h>

// dolfin headers
#include <dolfin/la/PETScVector.h>


namespace dolfin
{


class HarmonicOscillator
{
    /* Test case */
    public:
        HarmonicOscillator(const double gamma) : gamma(gamma) {}

        void operator() (const PETScVector &x, PETScVector &dxdt, const double /* t */) {
            VecSetValue(dxdt.vec(), 0, x[1], INSERT_VALUES);
            VecSetValue(dxdt.vec(), 1, -x[0] - gamma*x[1], INSERT_VALUES);
        }

    private:
        const double gamma;
};


PETScVector forward_euler(const PETScVector &ic, const double t0, const double t1, const double dt)
{
    auto u = ic;        // NB! should be a copy
    auto u_prev = ic;   // NB! should be a copy

    HarmonicOscillator rhs(0.1);

    auto t = t0;
    while (t < t1)
    {
        rhs(u_prev, u, t);      // u = rhs(u_prev, t)
        u *= dt;                // u = dt*rhs(u_prev, t)
        u += u_prev;            // u = u_prev + dt*rhs(u_prev, t)

        u_prev = u;
        t += dt;
    }

    return u;
}


int add(int i, int j) {
    return i + j;
}



PYBIND11_MODULE(SIGNATURE, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
    m.def("forward_euler", &forward_euler);
}


}       // namespace dolfin

#endif
