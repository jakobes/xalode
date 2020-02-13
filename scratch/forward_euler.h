#ifndef RK4_H
#define RK4_H


#include <iostream>

#include "cressman.h"
#include "utils.h"


template< typename vector_type >
void cressman_FE(vector_type &V, vector_type &m, vector_type &n, vector_type &h, vector_type &Ca,
        vector_type &K, vector_type &Na, const double t0, const double t1, const double dt)
{
    Cressman_step< vector_type > rhs(1., 100., 40., 0.01, 0.05, 0.0175, 0.05, 0.1, 66, 8., 0.0445, 1000, 1.);

    vector_type new_V(V.size());
    vector_type new_m(V.size());
    vector_type new_n(V.size());
    vector_type new_h(V.size());
    vector_type new_Ca(V.size());
    vector_type new_K(V.size());
    vector_type new_Na(V.size());

    size_t num_steps = (t1 - t0)/dt + 1;
    for (size_t i = 0; i < num_steps; ++i)
    {
        rhs.step(V, m, n, h, Ca, K, Na, new_V, new_m, new_n, new_h, new_Ca, new_K, new_Na, dt);

        V = new_V;
        m = new_m;
        n = new_n;
        h = new_h;
        Ca = new_Ca;
        K = new_K;
        Na = new_Na;
    }
}


#endif
