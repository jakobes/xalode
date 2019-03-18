#ifndef FORWARD_EULER_H
#define FORWARD_EULER_H

// pybind headers
#include <pybind11/pybind11.h>

// dolfin headers
#include <dolfin/la/PETScVector.h>


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
}


}   // namespace dolfin


#endif
