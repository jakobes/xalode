#ifndef STEPPER_H
#define STEPPER_H

#include <iostream>

#include "cressman.h"
#include "utils.h"


template< class CallableObjectType, class vector_type >
void fe(CallableObjectType rhs, const vector_type &x, vector_type &dxdt, double t0, double t1, double dt)
{
    /* solve dx/dt = f(x, t) using forward euler. */
    auto t = t0;
    auto _x = x;
    while (t < t1)
    {
        rhs(_x, dxdt, t);
        axpy(dxdt, _x, dt);
        _x = dxdt;
        t += dt;
    }
}


template< class CallableObjectType, class vector_type >
void solve_interval_vectorised(CallableObjectType rhs, vector_type &V, vector_type &m,
        vector_type &n, vector_type &h, vector_type &Ca, vector_type &K, vector_type &Na,
        double t0, double t1, double dt)
{
    /* Solve the dx/dt = rhs(x, t) with each tuple (V[i], m[i], n[i], ...) as an initial condition. */
    vector_type x(7);
    vector_type dxdt(7);

    for (size_t i = 0; i < V.size(); ++i)
    {
        x[0] = V[i];
        x[1] = m[i];
        x[2] = n[i];
        x[3] = h[i];
        x[4] = Ca[i];
        x[5] = K[i];
        x[6] = Na[i];

        fe(rhs, x, dxdt, t0, t1, dt);
        V[i] = dxdt[0];
        m[i] = dxdt[1];
        n[i] = dxdt[2];
        h[i] = dxdt[3];
        Ca[i] = dxdt[4];
        K[i] = dxdt[5];
        Na[i] = dxdt[6];
    }
}


template< typename vector_type >
vector_type modular_forward_euler(vector_type &V, vector_type &m, vector_type &n, vector_type &h,
        vector_type &Ca, vector_type &K, vector_type &Na, double t0, double t1, double dt, size_t ode_dt_factor)
{
    Cressman rhs(1., 100., 40., 0.01, 0.05, 0.0175, 0.05, 0.1, 66, 8., 0.0445, 1000, 1.);
    size_t number_of_steps = static_cast< int >(t1 - t0)/dt + 1;
    vector_type results(number_of_steps);

    auto t = t0;
    for (size_t i = 0; i < number_of_steps; ++i)
    {
        solve_interval_vectorised(rhs, V, m, n, h, Ca, K, Na, t, t + dt, dt/ode_dt_factor);
        results[i] = V[0];
        t += dt;
    }
    return results;
}


#endif
