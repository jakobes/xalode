#ifndef CRESSMAN_H
#define CRESSMAN_H

// stl
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


class Cressman
{
    public:
        Cressman(const double Cm, const double GNa, const double GK, const double GAHP,
            const double GKL, const double GNaL, const double GClL, const double GCa, const double Gglia,
            const double Koinf, const double gamma1, const double tau, const double control)
            : Cm(Cm), GNa(GNa), GK(GK), GAHP(GAHP), GKL(GKL), GNaL(GNaL),
              GClL(GClL), GCa(GCa), Gglia(Gglia), Koinf(Koinf),
              gamma1(gamma1), tau(tau), control(control) { }

        template< class vector_type >
        void operator() (const vector_type &x, vector_type &dxdt, const double /* t */ ) {
            const double Ipump = rho*(1/(1 + exp((25 - x[6])/3.)))*(1/(1 + exp(5.5 - x[5])));
            const double IGlia = Gglia/(1 + exp((18.- x[5])/2.5));
            const double Idiff = eps0*(x[5] - Koinf);

            const double Ki = 140 + (18 - x[6]);
            const double Nao = 144 - beta0*(x[6] - 18);
            const double ENa = 26.64*log(Nao/x[6]);
            const double EK = 26.64*log(control*x[5]/Ki);

            const double am = (0.1*x[0] + 3.0)/(1 - exp(-0.1*x[0] - 3));
            const double bm = 4*exp(-1./18.*x[0] - 55./18.);
            const double an = (0.01*x[0] + 0.34)/(-exp(-0.1*x[0] - 3.4) + 1);
            const double bn = 0.125*exp(-0.0125*x[0] - 0.55);
            const double ah = 0.07*exp(-0.05*x[0] - 2.2);
            const double bh = 1.0/(exp(-0.1*x[0] - 1.4) + 1);

            const double taum = 1./(am + bm);
            const double minf = 1./(am + bm)*am;
            const double taun = 1./(an + bn);
            const double ninf = 1./(an + bn)*an;
            const double tauh = 1./(bh + ah);
            const double hinf = 1./(bh + ah)*ah;

            const double INa = GNa*pow(x[1], 3)*x[3]*(x[0] - ENa) + GNaL*(x[0] - ENa);
            const double IK = (GK*pow(x[2], 4) + GAHP*x[4]/(1 + x[4]) + GKL)*(x[0] - EK);
            const double ICl = GClL*(x[0] - ECl);

            dxdt[0] = -(INa + IK + ICl)/Cm;
            dxdt[1] = phi*(minf - x[1])/taum;
            dxdt[2] = phi*(ninf - x[2])/taun;
            dxdt[3] = phi*(hinf - x[3])/tauh;
            dxdt[4] = x[4]/80. - 0.002*GCa*(x[0] - ECa)/(1 + exp(-(x[0] + 25.)/2.5));
            dxdt[5] = (gamma1*beta0*IK - 2*beta0*Ipump - IGlia - Idiff)/tau;
            dxdt[6] = -(gamma1*INa + 3*Ipump)/tau;
        }

    private:
        const double rho = 1.25;
        const double eps0 = 1.2;
        const double beta0 = 7.0;
        const double ECa = 120.0;
        const double Cli = 6.0;
        const double Clo = 130.0;
        const double ECl = 26.64*log(Cli/Clo);
        const double phi = 3.0;

        const double Cm;
        const double GNa;
        const double GK;
        const double GAHP;
        const double GKL;
        const double GNaL;
        const double GClL;
        const double GCa;
        const double Gglia;
        const double Koinf;
        const double gamma1;
        const double tau;
        const double control;
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


typedef std::vector< int > ndarray;


class OdeSolverVectorised
{
    public:
        OdeSolverVectorised(const ndarray &V_map, const ndarray &m_map, const ndarray &n_map,
            const ndarray &h_map, const ndarray &Ca_map, const ndarray &K_map, const ndarray &Na_map) :
            V_map(V_map), n_map(n_map), m_map(m_map), h_map(h_map), Ca_map(Ca_map), K_map(K_map),
            Na_map(Na_map), rhs(1., 100., 40., 0.01, 0.05, 0.0175, 0.05, 0.1, 66, 4., 0.0445, 1000, 1.)
        { }

        void solve(PETScVector &state, const double t0, const double t1, const double dt)
        {
            for (size_t i = 0; i < V_map.size(); ++i)
            {
                u_prev[0] = state[V_map[i]];
                u_prev[1] = state[m_map[i]];
                u_prev[2] = state[n_map[i]];
                u_prev[3] = state[h_map[i]];
                u_prev[4] = state[Ca_map[i]];
                u_prev[5] = state[K_map[i]];
                u_prev[6] = state[Na_map[i]];

                forward_euler(rhs, u, u_prev, t0, t1, dt);

                VecSetValue(state.vec(), V_map[i], u[0], INSERT_VALUES);
                VecSetValue(state.vec(), m_map[i], u[1], INSERT_VALUES);
                VecSetValue(state.vec(), n_map[i], u[2], INSERT_VALUES);
                VecSetValue(state.vec(), h_map[i], u[3], INSERT_VALUES);
                VecSetValue(state.vec(), Ca_map[i], u[4], INSERT_VALUES);
                VecSetValue(state.vec(), K_map[i], u[5], INSERT_VALUES);
                VecSetValue(state.vec(), Na_map[i], u[6], INSERT_VALUES);
            }
        }

    private:
        Cressman rhs;
        const ndarray V_map;
        const ndarray n_map;
        const ndarray m_map;
        const ndarray h_map;
        const ndarray Ca_map;
        const ndarray K_map;
        const ndarray Na_map;
        std::array< double, 7 > u;
        std::array< double, 7 > u_prev;
};


PYBIND11_MODULE(SIGNATURE, m) {
    py::class_< OdeSolverVectorised >(m, "LatticeODESolver")
        .def(py::init< const ndarray &, const ndarray &, const ndarray &, const ndarray &, const ndarray &,
                const ndarray &, const ndarray & >())
        .def("solve", &OdeSolverVectorised::solve);
}


}   // namespace dolfin



#endif
