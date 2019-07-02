#ifndef VECTORISED_CRESSMAN_H
#define VECTORISED_CRESSMAN_H

// stl
#include <iostream>
#include <map>
#include <stdexcept>
#include <cstdio>
#include <algorithm>       // std::find
#include <memory>       // unique_ptr and shared_ptr

// pybind headers
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// dolfin headers
#include <dolfin/la/PETScVector.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/function/FunctionSpace.h>

// xalode headers
#include "xalode/cressman.h"
#include "xalode/forward_euler.h"
#include "xalode/utils.h"

namespace py = pybind11;

namespace dolfin
{


typedef std::vector< int > ndarray;


std::vector< int > filter_dofs(
        const GenericDofMap &dofmap,
        const MeshFunction< size_t > &cell_function,
        const std::vector< int > &cell_tags)
{
    // Consider writing a version of this function that works with shared pointers for dofmap
    std::vector< int > filtered_dofs;

    for (size_t cell_id = 0; cell_id < cell_function.size(); ++cell_id)
    {
        auto cell_dofs = dofmap.cell_dofs(cell_id);
        if (std::find(cell_tags.begin(), cell_tags.end(), cell_function[cell_id]) != cell_tags.end())
        {
            for (int i = 0; i < cell_dofs.rows(); ++i)
                filtered_dofs.emplace_back(cell_dofs.data()[i]);
        }
    }
    filtered_dofs.erase(uniquify(filtered_dofs.begin(), filtered_dofs.end()), filtered_dofs.end());
    return filtered_dofs;
}


class ODESolverVectorised
{
    public:
        ODESolverVectorised(
                const ndarray &V_map,
                const ndarray &m_map,
                const ndarray &n_map,
                const ndarray &h_map,
                const ndarray &Ca_map,
                const ndarray &K_map,
                const ndarray &Na_map) :
            V_map(V_map),
            n_map(n_map),
            m_map(m_map),
            h_map(h_map),
            Ca_map(Ca_map),
            K_map(K_map),
            Na_map(Na_map)
            {
                rhs = new Cressman();
            }

        ~ODESolverVectorised() { delete rhs; }       // Why does not unique_ptr work?

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

                forward_euler(*rhs, u, u_prev, t0, t1, dt);

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
        Cressman *rhs;
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


class ODESolverVectorisedSubDomain
{
    public:
        ODESolverVectorisedSubDomain(
                const FunctionSpace &mixed_function_space,
                const MeshFunction< size_t > &cell_function,
                const std::map< int, float > &parameter_map) :
            mixed_function_space(mixed_function_space),
            cell_function(cell_function),
            parameter_map(parameter_map)
        {
            std::vector< int > cell_tags(parameter_map.size());
            for (auto &kv : parameter_map)
            {
                rhs_map[kv.first] = Cressman(kv.second);
                cell_tags.emplace_back(kv.first);
            }

            const auto num_sub_spaces = mixed_function_space.element().get()->num_sub_elements();
            for (size_t i = 0; i < num_sub_spaces; ++i)
            {
                subdomain_maps.emplace_back(filter_dofs(
                            *(mixed_function_space.sub(i).get()->dofmap().get()),
                            cell_function,
                            cell_tags));
            }
            /*
             * I am now where I was. the next question is to map the dofs to a subdomain
             * value. This means I need an associaton between the dofs and a value. I
             * think a vector is the way to go. The only way I can think of is to make the
             * dofmaps and the map to the cell function simultaneously. I can hacky away
             * with a static offset in the dofmap when solving the ODE. I should make a
             * new filter function that takes a shared pointer dof map and the cell
             * function and returns a std::vector< pair< dof, cell function value > >.
             * That should be straight forward.
             *
             * TODO: I need a dof to cell map. How can I get this?
             * */

        } // I should have some kind of checks. Better do on python side.

    private:
        const FunctionSpace &mixed_function_space;
        const MeshFunction< size_t > &cell_function;
        const std::map< int, float > &parameter_map;
        std::map< int, Cressman > rhs_map;
        std::vector< std::vector< int > > subdomain_maps;

};


PYBIND11_MODULE(SIGNATURE, m) {
    py::class_< ODESolverVectorised >(m, "LatticeODESolver")
        .def(py::init<
                const ndarray &,
                const ndarray &,
                const ndarray &,
                const ndarray &,
                const ndarray &,
                const ndarray &,
                const ndarray & >())
        .def("solve", &ODESolverVectorised::solve);
    py::class_< ODESolverVectorisedSubDomain >(m, "LatticeODESolverSubDomain")
        .def(py::init<
                const FunctionSpace &,
                const MeshFunction< size_t > &,
                const std::map< int, float > & >());
        /* .def("solve", &ODESolverVectorisedSubDomain::solve); */

    m.def("filter_dofs", &filter_dofs);
}


}   // namespace dolfin


#endif
