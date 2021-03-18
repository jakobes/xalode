#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

// Timing
#include <ctime>

// ODEINT
#include <boost/numeric/odeint.hpp>

// Local
#include "cressman.h"


typedef double float_type;
typedef boost::numeric::ublas::vector< float_type > vector_type;
typedef boost::numeric::ublas::matrix< float_type > matrix_type;
/* typedef std::vector< double > vector_type; */

using namespace boost::numeric::odeint;


void take1()
{
    using namespace std;
    using namespace boost::numeric::odeint;

    // Initial condition -- initialisation list ot working
    vector_type x(7);
    x[0] = -50;
    x[1] = 0.0936;
    x[2] = 0.96859;
    x[3] = 0.08553;
    x[4] = 0.0;
    x[5] = 7.8;
    x[6] = 15.5;

    vector< vector_type > x_vec;
    vector<double> times;

    // RHS class
    Cressman rhs = Cressman();

    const double dt = 1;     // time step of observer calls
    const double T = 1e3;
    const double abs_tol = 1e-3, rel_tol=1e-2;

    typedef runge_kutta_dopri5< vector_type > error_stepper_type;
    typedef controlled_runge_kutta< error_stepper_type > controlled_stepper_type;

    controlled_stepper_type controlled_stepper(abs_tol, rel_tol);
    auto tick = clock();
    size_t num_of_steps = integrate_adaptive(controlled_stepper, rhs, x, 0.0, T, dt);
    double tock = static_cast< double >(clock() - tick);
    cout << num_of_steps << " " << tock/CLOCKS_PER_SEC << endl;
    /* cout << "Success!" << endl; */

    for (auto v: x)
        cout << v << " ";
    cout << endl;
}


template< class controlled_stepper_class, class vector_class >
size_t forward_euler(
        controlled_stepper_class &stepper,
        Cressman rhs,
        vector_class &state,
        const double t0,
        const double t1)
{
    const double dt = 1;     // time step of observer calls
    size_t num_of_steps = integrate_adaptive(stepper, rhs, state, t0, t1, dt);
    return num_of_steps;
}


int main(int argc, char** argv)
{
    take1();

    /* vector_type x(7); */
    /* x[0] = -50; */
    /* x[1] = 0.0936; */
    /* x[2] = 0.96859; */
    /* x[3] = 0.08553; */
    /* x[4] = 0.0; */
    /* x[5] = 7.8; */
    /* x[6] = 15.5; */

    /* typedef runge_kutta_dopri5< vector_type > error_stepper_type; */
    /* typedef controlled_runge_kutta< error_stepper_type > controlled_stepper_type; */
    /* const double abs_tol = 1e-3, rel_tol=1e-2; */
    /* controlled_stepper_type controlled_stepper2(abs_tol, rel_tol); */

    /* auto num_of_steps = forward_euler(controlled_stepper2, Cressman(), x, 0.0, 1e3); */

    /* std::cout << "num steps: " << num_of_steps << std::endl; */

    /* for (auto v: x) */
    /*     std::cout << v << " "; */
    /* std::cout << std::endl; */
}
