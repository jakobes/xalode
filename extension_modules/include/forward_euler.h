#ifndef FORWARD_EULER_H
#define FORWARD_EULER_H

// stl
#include <vector>

// ODEINT
#include <boost/numeric/odeint.hpp>


using namespace boost::numeric::odeint;


template< class vector_type, typename float_type = typename vector_type::value_type >
void axpy(vector_type &x, vector_type &y, const float_type a)
{
    std::transform(x.begin(), x.end(), y.begin(), x.begin(),
            [a](const float_type xi, const float_type yi){return a*xi + yi;});
}


template< class CallableObjectType, class vector_type >
void forward_euler(
        CallableObjectType &rhs,
        vector_type &u,
        vector_type &u_prev,
        const double t0,
        const double t1,
        const double dt)
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


template< class controlled_stepper, class callable_object_type, class vector_type >
size_t forward_euler(
        controlled_stepper &stepper,
        callable_object_type &rhs,
        vector_type &state,
        const double t0,
        const double t1)
{
    const double dt = 1;     // time step of observer calls
    size_t num_of_steps = integrate_adaptive(stepper, rhs, state, t0, t1, dt);
    return num_of_steps;
}


template< class stepper_type, class callable_object_type, class vector_type >
void forward_euler(
        stepper_type &stepper,
        callable_object_type &rhs,
        vector_type &state,
        const double t0,
        const double t1,
        const double dt)
{
    integrate_const(stepper, rhs, state, t0, t1, dt);
}


template< class vector_type, class ode_type>
void forward_euler(
        std::shared_ptr< ode_type > rhs_ptr,
        vector_type &state,
        const double t0,
        const double t1,
        const double dt)
{
    modified_midpoint< std::vector< double > > stepper;
    /* integrate_const(stepper, *(rhs_ptr.get()), state, t0, t1, dt); */
    integrate_const(
            stepper,
            [rhs_ptr](const vector_type &x, vector_type &dxdt, const double t){ rhs_ptr.get()->eval(x, dxdt, t); },
            state, t0, t1, dt);
}

#endif
