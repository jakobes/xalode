#ifndef FORWARD_EULER_H
#define FORWARD_EULER_H

// pybind headers
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// dolfin headers
#include <dolfin/la/PETScVector.h>

namespace py = pybind11;


namespace dolfin
{


template< class vector_type, typename float_type = typename vector_type::value_type >
void axpy(vector_type &x, vector_type &y, const float_type a)
{
    std::transform(x.begin(), x.end(), y.begin(), x.begin(),
            [a](const float_type xi, const float_type yi){return a*xi + yi;});
}


template< class CallableObjectType, class vector_type >
void forward_euler(CallableObjectType rhs, vector_type &u, vector_type &u_prev,
        const double t0, const double t1, const double dt)
{
    auto t = t0;
    while (t < t1)
    {
        rhs(u_prev, u, t);         // u = rhs(u_prev, t)
        axpy(u, u_prev, dt);       // u = dt*u + u_prev     FIXME: I don''t think this is called axpy

        /* u *= dt;                // u = dt*rhs(u_prev, t) */
        /* u += u_prev;            // u = u_prev + dt*rhs(u_prev, t) */

        u_prev = u;
        t += dt;
    }
}


/* typedef py::array_t< double > ndarray; //TODO: Add buffer protocol to constructor */
typedef std::vector< double > ndarray;


class OdeSolverVectorised
{
    public:
        OdeSolverVectorised(const ndarray &V_map, const ndarray &n_map, const ndarray &m_map,
            const ndarray &h_map, const ndarray &Ca_map, const ndarray &K_map, const ndarray &Na_map) :
            V_map(V_map), n_map(n_map), m_map(m_map), h_map(h_map), Ca_map(Ca_map), K_map(K_map),
            Na_map(Na_map)
        {
            // TODO: Why can't I do this under private?
            /* Cressman rhs(1., 100., 40., 0.01, 0.05, 0.0175, 0.05, 0.1, 66, 8., 0.0445, 1000, 1.); */
        }

        void solve(PETScVector &state, const double t0, const double t1, const double dt)
        {
            // Why can't I have this somewhere else?
            Cressman rhs(1., 100., 40., 0.01, 0.05, 0.0175, 0.05, 0.1, 66, 8., 0.0445, 1000, 1.);

            std::vector< double > u(7);
            std::vector< double > u_prev(7);

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
        const ndarray V_map;
        const ndarray n_map;
        const ndarray m_map;
        const ndarray h_map;
        const ndarray Ca_map;
        const ndarray K_map;
        const ndarray Na_map;
};


void solve_all(PETScVector &V, PETScVector &m, PETScVector &n, PETScVector &h, PETScVector &Ca,
        PETScVector &K, PETScVector &Na, const double t0, const double t1, const double dt)
{
    Cressman rhs(1., 100., 40., 0.01, 0.05, 0.0175, 0.05, 0.1, 66, 8., 0.0445, 1000, 1.);
    std::vector< double > u(7);
    std::vector< double > u_prev(7);

    for (size_t i = 0; i < V.size(); ++i)
    {
        u_prev[0] = V[i];
        u_prev[1] = m[i];
        u_prev[2] = n[i];
        u_prev[3] = h[i];
        u_prev[4] = Ca[i];
        u_prev[5] = K[i];
        u_prev[6] = Na[i];

        forward_euler(rhs, u, u_prev, t0, t1, dt);
        VecSetValue(V.vec(), i, u[0], INSERT_VALUES);
        VecSetValue(m.vec(), i, u[1], INSERT_VALUES);
        VecSetValue(n.vec(), i, u[2], INSERT_VALUES);
        VecSetValue(h.vec(), i, u[3], INSERT_VALUES);
        VecSetValue(Ca.vec(), i, u[4], INSERT_VALUES);
        VecSetValue(K.vec(), i, u[5], INSERT_VALUES);
        VecSetValue(Na.vec(), i, u[6], INSERT_VALUES);
    }
}


PYBIND11_MODULE(SIGNATURE, m) {
    m.def("ode_solve", &solve_all);

    py::class_<OdeSolverVectorised>(m, "OdeSolverVectorised")
        .def(py::init< const ndarray &, const ndarray &, const ndarray &, const ndarray &, const ndarray &,
                const ndarray &, const ndarray & >())
        .def("solve", &OdeSolverVectorised::solve);
}


}   // namespace dolfin


#endif
