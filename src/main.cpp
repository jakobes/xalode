#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>         // TODO: remove?
#include <pybind11/stl_bind.h>

#include <vector>


namespace py = pybind11;


PYBIND11_MAKE_OPAQUE(std::vector< double >);
PYBIND11_MAKE_OPAQUE(std::vector< int >);


template<typename T>
py::buffer_info vector_buffer(std::vector<T> &v) {
    return py::buffer_info(&v[0], sizeof(T), py::format_descriptor<T>::format(), 1,
            { v.size() }, { sizeof(T) });
}


PYBIND11_MODULE(xalode, m) {
    // TODO: I don't think I need this
    py::bind_vector< std::vector< double > >(m, "VectorDouble", py::buffer_protocol())
        .def_buffer(&vector_buffer< double >);

    py::bind_vector< std::vector< int > >(m, "VectorInt", py::buffer_protocol())
        .def_buffer(&vector_buffer< int >);
}
