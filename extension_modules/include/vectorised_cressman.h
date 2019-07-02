#ifndef VECTORISED_CRESSMAN_H
#define VECTORISED_CRESSMAN_H

// stl
#include <iostream>
#include <map>
#include <stdexcept>
#include <cstdio>
#include <algorithm>       // std::find
/* #include <memory>       // unique_ptr */

// pybind headers
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// dolfin headers
#include <dolfin/la/PETScVector.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/mesh/MeshFunction.h>

// xalode headers
#include "xalode/cressman.h"
#include "xalode/forward_euler.h"

namespace py = pybind11;

namespace dolfin
{


typedef std::vector< int > ndarray;


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
                const ndarray &V_map,
                const ndarray &m_map,
                const ndarray &n_map,
                const ndarray &h_map,
                const ndarray &Ca_map,
                const ndarray &K_map,
                const ndarray &Na_map,
                const ndarray &cell_function_map,
                std::map< int, double > &parameter_map) :
            V_map(V_map),
            n_map(n_map),
            m_map(m_map),
            h_map(h_map),
            Ca_map(Ca_map),
            K_map(K_map),
            Na_map(Na_map),
            cell_function_map(cell_function_map)
            {
                /*
                 *  The parameter map is a map {cell id: parameter value}.
                 *  The cell id for dof i is found in index i in cell_function_map.
                 *
                 */
                for (auto &key : cell_function_map)
                    if (!parameter_map.count(key))
                        throw std::invalid_argument("Missing key from parameter_map: " + std::to_string(key) + "\n");

                for (auto &p : parameter_map)
                    rhs_subdomain_map[p.first] = new Cressman(p.second);
            }

        void solve(PETScVector &state, const double t0, const double t1, const double dt)
        {
            for (size_t i = 0; i < cell_function_map.size(); ++i)
            {
                u_prev[0] = state[V_map[i]];
                u_prev[1] = state[m_map[i]];
                u_prev[2] = state[n_map[i]];
                u_prev[3] = state[h_map[i]];
                u_prev[4] = state[Ca_map[i]];
                u_prev[5] = state[K_map[i]];
                u_prev[6] = state[Na_map[i]];

                forward_euler(*rhs_subdomain_map[cell_function_map[i]], u, u_prev, t0, t1, dt);

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
        std::map< int, Cressman * > rhs_subdomain_map;
        const ndarray V_map;
        const ndarray n_map;
        const ndarray m_map;
        const ndarray h_map;
        const ndarray Ca_map;
        const ndarray K_map;
        const ndarray Na_map;
        const ndarray cell_function_map;
        std::array< double, 7 > u;
        std::array< double, 7 > u_prev;
};


std::vector< int > filter_dofs(DofMap &dofmap, MeshFunction< size_t > &cell_function, std::vector< int > &cell_tags)
{
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


PYBIND11_MODULE(SIGNATURE, m) {
    py::class_< ODESolverVectorised >(m, "LatticeODESolver")
        .def(py::init< const ndarray &, const ndarray &, const ndarray &, const ndarray &, const ndarray &,
                const ndarray &, const ndarray & >())
        .def("solve", &ODESolverVectorised::solve);
    py::class_< ODESolverVectorisedSubDomain >(m, "LatticeODESolverSubDomain")
        .def(py::init< const ndarray &, const ndarray &, const ndarray &, const ndarray &, const ndarray &,
                const ndarray &, const ndarray &, const ndarray &, std::map< int, double > & >())
        .def("solve", &ODESolverVectorisedSubDomain::solve);

    m.def("filter_dofs", &filter_dofs);
}


}   // namespace dolfin


#endif
