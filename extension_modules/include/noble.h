#ifndef CRESSMAN_H
#define CRESSMAN_H

// STL
#include <cmath>
#include <iostream>

// pybind headers
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>

// dolfin headers
#include <dolfin/la/PETScVector.h>

namespace py = pybind11;


class Noble
{
    public:
        Noble(const double Cm, const double gNa, const double gK1, const double gL, const double gK2) :
            Cm(Cm), gNa(gNa), gK1(gK1), gL(gL), gK2(gK2) {}

        template< class vector_type >
        void operator() (const vector_type &x, vector_type &dxdt, const double /* t */)
        {
            const double am = 0.1*(x[0] + 48)/(1 - exp(-(x[0] + 48)/15));
            const double bm = 0.12*(x[0] + 8)/(exp((x[0] + 8)/5)-1);

            const double an = 0.0001*(x[0] + 50)/(1 - exp(-(x[0] + 50)/10));
            const double bn = 0.002*exp(-(x[0] + 90)/80);

            const double ah = 0.17*exp(-(x[0] + 90)/20);
            const double bh = 1/(1 + exp(-(x[0] + 42)/10));

            const double tau_m = 1/(am + bm);
            const double m_inf = am/(am + bm);

            const double tau_n = 1/(an + bn);
            const double n_inf = an/(an + bn);

            const double tau_h = 1/(ah + bh);
            const double h_inf = ah/(ah + bh);

            const double INa = (gNa*pow(x[2], 3)*x[1] + 0.14)*(x[0] - 40);
            const double IK = (gK1*pow(x[3], 4) + gK2*exp(-(x[0] + 90)/50) + gK2/80*exp((x[0] + 90)/60))*(x[0] + 100);
            const double IL = gL*(x[0] + 60);

            dxdt[0] = -(INa + IK + IL)/Cm,
            dxdt[1] = (h_inf - x[1])/tau_h,
            dxdt[2] = (m_inf - x[2])/tau_m,
            dxdt[3] = (n_inf - x[3])/tau_n;
        }

    private:
        const double Cm;
        const double gNa;
        const double gK1;
        const double gL;
        const double gK2;
};


namespace dolfin
{


template< class vector_type, typename float_type = typename vector_type::value_type >
void axpy(vector_type &x, vector_type &y, const float_type a)
{
    std::transform(x.begin(), x.end(), y.begin(), x.begin(),
            [a](const float_type xi, const float_type yi){return a*xi + yi;});
}


/* template< class vector_type > */
template< class CallableObjectType, class vector_type >
void forward_euler(CallableObjectType &rhs, vector_type &u, vector_type &u_prev,
        const double t0, const double t1, const double dt)
{
    auto t = t0;
    while (t < t1)
    {
        rhs(u_prev, u, t);         // u = rhs(u_prev, t)
        axpy(u, u_prev, dt);       // u = dt*u + u_prev

        u_prev = u;
        t += dt;
    }
}


typedef std::vector< double > ndarray;


class ODESolverVectorised
{
    public:
        ODESolverVectorised(const ndarray &V_map, const ndarray &m_map, const ndarray &n_map,
            const ndarray &h_map):
            V_map(V_map), n_map(n_map), m_map(m_map), h_map(h_map),
            rhs(12.0, 400, 1.2, 0.1845, 1.2)
        { }

        void solve(PETScVector &state, const double t0, const double t1, const double dt)
        {
            for (size_t i = 0; i < V_map.size(); ++i)
            {
                u_prev[0] = state[V_map[i]];
                u_prev[1] = state[m_map[i]];
                u_prev[2] = state[n_map[i]];
                u_prev[3] = state[h_map[i]];

                forward_euler(rhs, u, u_prev, t0, t1, dt);

                VecSetValue(state.vec(), V_map[i], u[0], INSERT_VALUES);
                VecSetValue(state.vec(), m_map[i], u[1], INSERT_VALUES);
                VecSetValue(state.vec(), n_map[i], u[2], INSERT_VALUES);
                VecSetValue(state.vec(), h_map[i], u[3], INSERT_VALUES);
            }
        }

    private:
        Noble rhs;
        const ndarray V_map;
        const ndarray n_map;
        const ndarray m_map;
        const ndarray h_map;
        std::array< double, 4 > u;
        std::array< double, 4 > u_prev;
};


PYBIND11_MODULE(SIGNATURE, m) {
    py::class_< ODESolverVectorised >(m, "LatticeODESolver")
        .def(py::init< const ndarray &, const ndarray &, const ndarray &, const ndarray & >())
        .def("solve", &ODESolverVectorised::solve);
}


}   // namespace dolfin


#endif
