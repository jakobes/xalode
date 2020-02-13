#ifndef RK4_H
#define RK4_H


#include <iostream>
#include "utils.h"


using namespace std;
/* typedef vector< double > vector_type; */


template< class CallableObjectType, class vector_type >
void rk4_step(CallableObjectType rhs, vector_type &x, const double t, const double dt)
{
    vector_type x_tmp(x.size());

    vector_type k1(x.size());
    vector_type k2(x.size());
    vector_type k3(x.size());
    vector_type k4(x.size());

    // k1 = dt*rhs(x, t)
    rhs(x, k1, t);
    multiply(k1, dt);

    x_tmp = k1;
    axpy(x_tmp, x, 0.5);
    // x_tmp = x + k1/2

    // k2 = dt*rhs(x + k1/2, t + dt/2)
    rhs(x_tmp, k2, t + dt/2.);
    multiply(k2, dt);

    x_tmp = k2;
    axpy(x_tmp, x, 0.5);
    // x_tmp = x + k2/2

    // k3 = dt*rhs(x + k2/2, t + dt/2)
    rhs(x_tmp, k3, t + dt/2.);
    multiply(k3, dt);

    x_tmp = k3;
    axpy(x_tmp, x, 1.0);

    // k4 = dt*rhs(x + k3, t + dt)
    rhs(x_tmp, k4, t + dt);
    multiply(k4, dt);

    for (size_t i = 0; i < x.size(); ++i)
    {
        x[i] += 1./6.*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
    }
}


template< class CallableObjectType, class vector_type >
void runge_kutta_stepper(CallableObjectType rhs, vector_type &x, double t0, double t1, double dt)
{
    auto t = t0;
    while (t < t1)
    {
        rk4_step(rhs, x, t, dt);
        cout << x[0] << " " << x[1] << endl;
        t += dt;
    }
}





#endif
