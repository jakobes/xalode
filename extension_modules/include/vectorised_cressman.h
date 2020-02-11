#ifndef VECTORISED_CRESSMAN_H
#define VECTORISED_CRESSMAN_H

// stl
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <stdexcept>
#include <cstdio>           // printf?
#include <algorithm>        // std::find
#include <memory>           // unique_ptr and shared_ptr

// pybind headers
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// dolfin headers
#include <dolfin/la/PETScVector.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/function/FunctionSpace.h>

#include <dolfin/log/log.h>

// ODEINT
#include <boost/numeric/odeint.hpp>


namespace py = pybind11;

namespace dolfin
{


typedef std::vector< int > ndarray;
typedef std::map< int, std::vector< double > > vertex_vector_map;
typedef std::map< int, std::set< size_t > > index_map;


class ODEMap
{
    public:
        template <typename ode_type >
        void add_ode(int key, ode_type &ode)
        {
            ode_map.emplace(std::make_pair(key, ode.clone()));
        }

        std::map< int, std::shared_ptr< ODEBase > > get_map()
        {
            return ode_map;
        }

    private:
        std::map< int, std::shared_ptr< ODEBase > > ode_map;
};



index_map new_and_improved_dof_filter(
        std::shared_ptr< const GenericDofMap > dofmap,
        const ndarray cell_function,
        const std::vector< int > &cell_tags)
{
    /*
     * Make a map my_map[cell_tag] -> dof_vector
     *
     * Only make sure that no dof is contained in a vector belonging to a tag with a higher
     * numerical value.
     *
     * NB! Cell to dof priority is the same as the order in cell_tags!!!
     */
    index_map tag_dof_map;

    // For each cell -- loop over indices
    for (size_t cell_counter = 0; cell_counter < cell_function.size(); ++cell_counter)
    {
        const auto tag = cell_function[cell_counter];

        // If cell value is contained in `cell_tags`
        if (std::find(cell_tags.begin(), cell_tags.end(), cell_function[cell_counter])
                != cell_tags.end())
        {
            auto cell_dofs = dofmap.get()->cell_dofs(cell_counter);
            for (int i = 0; i < cell_dofs.rows(); ++i)
                tag_dof_map[tag].emplace(cell_dofs.data()[i]);
        }
    }

    auto seen_dofs = tag_dof_map[cell_tags[0]];           // copy
    std::set< size_t > intersection;       // Helper for computing set intersection

    // Skip first element
    for (size_t i = 1; i < cell_tags.size(); ++i)
    {
        auto tmp_dofs = &tag_dof_map[cell_tags[i]];   //
        // Make sure no dofs in tag_dof_map[tags[i]] are in `seen_dofs`
        std::set_intersection(tmp_dofs->begin(), tmp_dofs->end(),
                              seen_dofs.begin(), seen_dofs.end(),
                              std::inserter(intersection, intersection.end()));

        // Remove the intersection with `seen_dofs` from `tag_dof_map[tags[i]]`.
        for (auto set_iterator = tmp_dofs->begin(); set_iterator != tmp_dofs->end(); )
        {
            // Do not update
            if (binary_search(begin(intersection), end(intersection), *set_iterator))
                tmp_dofs->erase(set_iterator++);
            else
                ++set_iterator;
        }

        // I hope this inserts into `seen_dofs`
        seen_dofs.insert(tmp_dofs->begin(), tmp_dofs->end());

    }
    return tag_dof_map;
}


std::vector< size_t > filter_dofs(
        std::shared_ptr< const GenericDofMap > dofmap,
        const MeshFunction< size_t > &vertex_function,
        const std::vector< int > &vertex_tags)
{
    std::vector< size_t > filtered_dofs;
    const auto dofs = dofmap.get()->dofs();

    for (auto dof: dofs)
    {
        if (std::find(vertex_tags.begin(), vertex_tags.end(), vertex_function[dof])
                != vertex_tags.end())
        {
            filtered_dofs.emplace_back(dof);
        }
    }
    return filtered_dofs;
}


MeshFunction< size_t > cell_to_vertex(const MeshFunction< size_t > &cell_function)
{
    const auto mesh = cell_function.mesh();
    const auto tdim = mesh.get()->topology().dim();
    const auto cell_vertex_count = mesh.get()->type().num_vertices(tdim);
    const auto num_cells = mesh.get()->topology().size(tdim);

    auto vertex_function = MeshFunction< size_t >(mesh, 0);
    vertex_function.set_all(0);

    for (size_t cell_number = 0; cell_number < num_cells; ++cell_number)
    {
        auto value = cell_function[cell_number];
        if (value != 0)
            for (size_t cell_vertex = 0; cell_vertex < cell_vertex_count; ++cell_vertex)
            {
                auto vf_index = mesh.get()->cells()[cell_number*cell_vertex_count + cell_vertex];
                if (value > vertex_function[vf_index])
                    vertex_function.set_value(vf_index, value);
            }
    }
    return vertex_function;
}


void assign_vector(
        PETScVector &vector,
        vertex_vector_map &initial_conditions,
        const ndarray &cell_function,
        const FunctionSpace &mixed_function_space)
{
    const auto num_sub_spaces = mixed_function_space.element().get()->num_sub_elements();

    // Store keys in separate vector
    std::vector< int > cell_tags(initial_conditions.size());
    for (const auto &kv: initial_conditions)
        cell_tags.emplace_back(kv.first);

    // For each sub space
    for (size_t sub_space_counter = 0; sub_space_counter < num_sub_spaces; ++sub_space_counter)
    {
        const auto tag_dof_map = new_and_improved_dof_filter(
                mixed_function_space.sub(sub_space_counter).get()->dofmap(),
                cell_function, cell_tags);
        // for each value of the cell tags
        for (const auto &kv: tag_dof_map)
        {
            // For each dof corresponing to a cell_tag
            for (const auto dof: kv.second)
            {
                VecSetValue(vector.vec(), dof, initial_conditions[kv.first][sub_space_counter],
                        INSERT_VALUES);
            }
        }
    }
}


class ODESolverVectorisedSubDomain
{
    public:
        ODESolverVectorisedSubDomain(
                const FunctionSpace &mixed_function_space,
                const ndarray &cell_function,
                ODEMap &ode_container)
                /* const std::map< int, float > &parameter_map) */
        {
            num_sub_spaces = mixed_function_space.element().get()->num_sub_elements();
            // vector of the cell tags in `parameter_map`.
            std::vector< int > cell_tags {};

            ode_map = ode_container.get_map();

            // Create `rhs_map` such that rhs_map[parameter value] -> rhs callable.
            /* for (auto &kv : parameter_map) */
            for (auto &kv : ode_map)
            {
                cell_tags.emplace_back(kv.first);
                /* auto ode = Cressman(kv.second); */
                /* odemap.add_ode< Cressman >(kv.first, ode); */
            }

            for (int ct: cell_tags)
            {
                tag_state_dof_map[ct] = std::vector< std::vector< size_t > >(
                        num_sub_spaces, std::vector< size_t >());
            }

            for (int sub_space_counter = 0; sub_space_counter < num_sub_spaces; ++sub_space_counter)
            {
                const auto tag_dof_map = new_and_improved_dof_filter(
                        mixed_function_space.sub(sub_space_counter).get()->dofmap(),
                        cell_function, cell_tags);
                for (const auto &kv: tag_dof_map)
                {
                    tag_state_dof_map[kv.first][sub_space_counter] =
                        std::vector< size_t >(kv.second.begin(), kv.second.end());
                }
            }

            // tag_state_dof_map will have keys == cell_tags
            // tag state_dof_map will have length 7 for all keys
            // the innermost vectors (dofs) for all 7 will have the same length

        } // I should have some kind of checks. Better do on python side.

        void solve(PETScVector &state, const double t0, const double t1, const double dt)
        {
            std::vector< double> u(num_sub_spaces);
            std::vector< double > u_prev(num_sub_spaces);

            for (const auto &kv : tag_state_dof_map)
            {
                /* auto rhs = rhs_map[kv.first]; */

                // Assume all dof vectors are of equal size. Anything else is a bug!
                for (size_t dof_counter = 0; dof_counter < kv.second[0].size(); ++dof_counter)
                {
                    // Fill values from `state` into `u_prev`.
                    for (int state_counter = 0; state_counter < num_sub_spaces; ++state_counter)
                    {
                        u_prev[state_counter] = state[kv.second[state_counter][dof_counter]];
                    }

                    /* ode_map[kv.first].get()->print(); */
                    /* for (auto v: u_prev) */
                    /*     std::cout << v << ", "; */
                    /* std::cout << std::endl; */

                    forward_euler(const_stepper, ode_map[kv.first], u_prev, t0, t1, dt);

                    /* forward_euler(const_stepper, rhs_map[kv.first], u_prev, t0, t1, dt); */
                    /* for (auto v: u_prev) */
                    /*     std::cout << v << ", "; */
                    /* std::cout << std::endl; */
                    /* std::cout << std::endl; */
                    /* std::cout << std::endl; */

                    // Fill values from `u_prev` into `State`. My custom odesolver requires u and u_prev
                    for (int state_counter = 0; state_counter < num_sub_spaces; ++state_counter)
                    {
                        VecSetValue(state.vec(), kv.second[state_counter][dof_counter],
                                u_prev[state_counter], INSERT_VALUES);
                    }
                }
            }
        }

    private:
        MeshFunction< size_t > vertex_function;
        /* std::map< int, std::shared_ptr< Cressman > > rhs_map; */
        std::map< int, std::shared_ptr< ODEBase > > rhs_map;
        int num_sub_spaces;

        // cell_tag -> state variables -> dofs
        std::map< int, std::vector< std::vector < size_t > > > tag_state_dof_map;

        // Ode stepper
        modified_midpoint< std::vector< double > > const_stepper;

        std::map< int, std::shared_ptr< ODEBase > > ode_map;
};


PYBIND11_MODULE(SIGNATURE, m) {
    py::class_< ODEMap >(m, "ODEMap")
        .def(py::init<>())
        .def("add_ode", &ODEMap::add_ode< Cressman >)
        .def("add_ode", &ODEMap::add_ode< Fitzhugh >)
        .def("add_ode", &ODEMap::add_ode< MorrisLecar >);

    py::class_< Cressman >(m, "Cressman")
        .def(py::init<
                    double, double, double, double,
                    double, double, double, double,
                    double, double, double, double>(),
               py::arg("Kbath") = 4.0,
               py::arg("Cm") = 1.0,
               py::arg("GNa") = 100.0,
               py::arg("GK") = 40.0,
               py::arg("GAHP") = 0.01,
               py::arg("GKL") = 0.05,
               py::arg("GNaL") = 0.0175,
               py::arg("GClL") = 0.05,
               py::arg("GCa") = 0.1,
               py::arg("Gglia") = 66.0,
               py::arg("gamma1") = 0.0554,
               py::arg("tau") = 1000.0);
        // There is some skullduggery with virtual functions. Trampoline classes?
        /* .def("eval", Cressman::eval); */

    py::class_< Fitzhugh >(m, "Fitzhugh")
        .def(py::init<
                    double, double, double, double,
                    double, double, double>(),
              py::arg("a") = 0.13,
              py::arg("b") = 13.0,
              py::arg("c1") = 0.26,
              py::arg("c2") = 10.0,
              py::arg("c3") = 1.0,
              py::arg("v_rest") = -70.0,
              py::arg("v_peak") = 40);
        // There is some skullduggery with virtual functions. Trampoline classes?
        /* .def("eval", Fitzhugh::eval); */

    py::class_< MorrisLecar >(m, "MorrisLecar")
        .def(py::init<double>(), py::arg("Iext") = 40);

    py::class_< ODESolverVectorisedSubDomain >(m, "LatticeODESolver")
        .def(py::init<
                const FunctionSpace &,
                const ndarray &,
                ODEMap &>())
                /* const std::map< int, float > & >()) */
        .def("solve", &ODESolverVectorisedSubDomain::solve);

    m.def("filter_dofs", &filter_dofs);
    m.def("cell_to_vertex", &cell_to_vertex);
    m.def("assign_vector", &assign_vector);
    m.def("new_and_improved_dof_filter", &new_and_improved_dof_filter);
}


}   // namespace dolfin


#endif
