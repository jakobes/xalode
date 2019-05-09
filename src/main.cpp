#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <pybind11/stl_bind.h>
/* #include <pybind11/stl.h> */
/* #include <pybind11/functional.h> */

#include "cressman.h"
#include "forward_euler.h"
#include "modular.h"

#include <vector>

#include <armadillo>


namespace py = pybind11;


PYBIND11_MAKE_OPAQUE(std::vector< double >);
PYBIND11_MAKE_OPAQUE(std::vector< int >);


template<typename T>
py::buffer_info vector_buffer(std::vector<T> &v) {
    return py::buffer_info(&v[0], sizeof(T), py::format_descriptor<T>::format(), 1, { v.size() }, { sizeof(T) });
}


PYBIND11_MODULE(bbidomain, m) {
    py::bind_vector< std::vector< double > >(m, "VectorDouble", py::buffer_protocol())
        .def_buffer(&vector_buffer< double >);

    py::bind_vector< std::vector< int > >(m, "VectorInt", py::buffer_protocol())
        .def_buffer(&vector_buffer< int >);

    m.def("cressman_FE", &cressman_FE< std::vector< double > >);
    m.def("modular_forward_euler", &modular_forward_euler< std::vector< double > >);
}
