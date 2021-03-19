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
#include <dolfin/common/MPI.h>

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
        void add_ode(size_t key, ode_type &ode)
        {
            ode_map.emplace(std::make_pair(key, ode.clone()));
            cell_tags.emplace_back(key);
        }

        std::map< size_t, std::shared_ptr< ODEBase > > get_map()
        {
            return ode_map;
        }

        std::vector< size_t > get_tags()
        {
            return cell_tags;
        }

    private:
        std::vector< size_t > cell_tags;
        std::map< size_t, std::shared_ptr< ODEBase > > ode_map;
};


index_map new_and_improved_dof_filter(
        std::shared_ptr< const GenericDofMap > dofmap,
        const std::vector< size_t > cell_function,
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
        const std::vector< size_t > &cell_function,
        const FunctionSpace &mixed_function_space)
{
    const size_t num_sub_spaces = mixed_function_space.element().get()->num_sub_elements();
    assert(false);

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
                ODEMap &ode_container,
                int num_sub_spaces) : num_sub_spaces(num_sub_spaces)
        {
            ode_map = ode_container.get_map();
            u_prev = std::vector< double >(num_sub_spaces);
        }

        void solve(
                /* std::vector< double > &state, */
                PETScVector &state,
                const double t0,
                const double t1,
                const double dt,
                PETScVector &indicator_function)
        {
            assert(false);
            /* std::vector< double > u(num_sub_spaces); */
            std::vector< double > local_state;
            std::vector< double > local_indicator;

            state.get_local(local_state);
            indicator_function.get_local(local_indicator);

            bool looog = true;

            size_t indicator_counter = 0;
            size_t dof_index = 0;
            while (dof_index < local_state.size())
            {
                size_t cell_tag = static_cast< size_t >(local_indicator[indicator_counter++]);
                // If map contains key
                if (ode_map.count(cell_tag) != 0)
                {


                if (looog)
                {
                    size_t jj = dof_index / 2;
                    std::cout <<
                        local_state.size() << " " <<
                        local_state[jj + 0] << " " <<
                        local_state[jj + 1] << " " <<
                        local_state[jj + 2] << " " <<
                        local_state[jj + 3] << " " <<
                        local_state[jj + 4] << " " <<
                        local_state[jj + 5] << " " <<
                        local_state[jj + 6] << " " <<
                        std::endl;
                    looog = false;
                }

                // Move variables from state to u_prev
                for (size_t sub_space_index = 0; sub_space_index < num_sub_spaces; ++sub_space_index)
                {
                    u_prev[sub_space_index] = local_state[dof_index + sub_space_index];
                }

                forward_euler(const_stepper, ode_map[cell_tag], u_prev, t0, t1, dt);

                for (size_t sub_space_index = 0; sub_space_index < num_sub_spaces; ++sub_space_index)
                    local_state[dof_index + sub_space_index] = u_prev[sub_space_index];

                }
                dof_index += num_sub_spaces;
            }
            state.set_local(local_state);
        }

    private:
        PETScVector indicator_function;
        int num_sub_spaces;

        // Ode stepper
        /* modified_midpoint< std::vector< double > > const_stepper; */
        runge_kutta4< std::vector< double > > const_stepper;

        std::map< size_t, std::shared_ptr< ODEBase > > ode_map;
        std::vector< double > u_prev;
};


void test_function_space(std::shared_ptr< const FunctionSpace > foo)
{
    std::cout << "Success! " << foo.get()->dim() << std::endl;
}

void test_function_space2(const FunctionSpace foo)
{
    std::cout << "Success! " << foo.dim() << std::endl;
}


void test_vector(std::vector< size_t > foo)
{
    std::cout << "success! \n" << std::endl;
    for (auto const f: foo) {
        std::cout << f << std::endl;
    }
}


PYBIND11_MODULE(SIGNATURE, m) {
    py::class_< ODEMap >(m, "ODEMap")
        .def(py::init<>())
        .def("get_tags", &ODEMap::get_tags)
        .def("add_ode", &ODEMap::add_ode< Cressman >)
        .def("add_ode", &ODEMap::add_ode< SimpleODE >)
        .def("add_ode", &ODEMap::add_ode< NoCellModel >)
        .def("add_ode", &ODEMap::add_ode< Fitzhugh >)
        .def("add_ode", &ODEMap::add_ode< MorrisLecar >)
        .def("add_ode", &ODEMap::add_ode< Synaptic >);

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

    py::class_< SimpleODE >(m, "SimpleODE")
        .def(py::init< >());

    py::class_< NoCellModel >(m, "NoCellModel")
        .def(py::init< >());

    py::class_< Synaptic >(m, "Synaptic")
        .def(py::init< >());

    py::class_< MorrisLecar >(m, "MorrisLecar")
        .def(py::init<double>(), py::arg("Iext") = 40);

    py::class_< ODESolverVectorisedSubDomain >(m, "LatticeODESolver")
        .def(py::init<
                ODEMap &,
                int >())
        .def("solve", &ODESolverVectorisedSubDomain::solve);

    // m.def("filter_dofs", &filter_dofs);
    // m.def("cell_to_vertex", &cell_to_vertex);
    m.def("assign_vector", &assign_vector);
    // m.def("new_and_improved_dof_filter", &new_and_improved_dof_filter);

}


}   // namespace dolfin


#endif
