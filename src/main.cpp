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


/* template<typename T> */
/* py::buffer_info vector_buffer(std::vector<T> &v) { */
/*     return py::buffer_info(&v[0], sizeof(T), py::format_descriptor<T>::format(), 1, { v.size() }, { sizeof(T) }); */
/* } */


PYBIND11_MAKE_OPAQUE(std::vector<double>);
/* PYBIND11_MAKE_OPAQUE(py::buffer); */

/* class Matrix { */
/* public: */
/*     Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) { */
/*         m_data = new float[rows*cols]; */
/*     } */
/*     float *data() { return m_data; } */
/*     size_t rows() const { return m_rows; } */
/*     size_t cols() const { return m_cols; } */
/* private: */
/*     size_t m_rows, m_cols; */
/*     float *m_data; */
/* }; */


/* py::array_t<double> add_arrays(py::buffer input1, py::buffer input2) { */

/*     py::buffer_info buf1 = input1.request(), buf2 = input2.request(); */

/*     if (buf1.ndim != 1 || buf2.ndim != 1) */
/*         throw std::runtime_error("Number of dimensions must be one"); */

/*     if (buf1.size != buf2.size) */
/*         throw std::runtime_error("Input shapes must match"); */

/*     /1* No pointer is passed, so NumPy will allocate the buffer *1/ */
/*     auto result = py::array_t<double>(buf1.size); */

/*     py::buffer_info buf3 = result.request(); */

/*     double *ptr1 = (double *) buf1.ptr, */
/*            *ptr2 = (double *) buf2.ptr, */
/*            *ptr3 = (double *) buf3.ptr; */

/*     for (size_t idx = 0; idx < buf1.shape[0]; idx++) */
/*         ptr3[idx] = ptr1[idx] + ptr2[idx]; */

/*     return result; */
/* } */

template<typename T>
py::buffer_info vector_buffer(std::vector<T> &v) {
    return py::buffer_info(&v[0], sizeof(T), py::format_descriptor<T>::format(), 1, { v.size() }, { sizeof(T) });
}


PYBIND11_MODULE(bbidomain, m) {
    py::bind_vector< std::vector< double > >(m, "VectorDouble", py::buffer_protocol())
        .def_buffer(&vector_buffer< double >);

    m.def("cressman_FE", &cressman_FE< std::vector< double > >);
    /* m.def("add_arrays", &add_arrays, "Add two NumPy arrays"); */

    /* m.def("cressman_FE", */
    /*         [](py::buffer V, py::buffer m, py::buffer &n, py::buffer &h, py::buffer &Ca, */
    /*             /1* py::buffer &K, py::buffer &Na, const double t0, const double t1, const double dt, py::buffer_protocol()) *1/ */
    /*             py::buffer &K, py::buffer &Na, const double t0, const double t1, const double dt) */
    /*         { */
    /*             auto V_buffer = V.request(); */
    /*             arma::vec V_vec(static_cast< double * >(V_buffer.ptr), V_buffer.shape[0]); */

    /*             auto m_buffer = m.request(); */
    /*             arma::vec m_vec(static_cast< double * >(m_buffer.ptr), m_buffer.shape[0]); */

    /*             auto n_buffer = n.request(); */
    /*             arma::vec n_vec(static_cast< double * >(n_buffer.ptr), n_buffer.shape[0]); */

    /*             auto h_buffer = h.request(); */
    /*             arma::vec h_vec(static_cast< double * >(h_buffer.ptr), h_buffer.shape[0]); */

    /*             auto Ca_buffer = Ca.request(); */
    /*             arma::vec Ca_vec(static_cast< double * >(Ca_buffer.ptr), Ca_buffer.shape[0]); */

    /*             auto K_buffer = K.request(); */
    /*             arma::vec K_vec(static_cast< double * >(K_buffer.ptr), K_buffer.shape[0]); */

    /*             auto Na_buffer = Na.request(); */
    /*             arma::vec Na_vec(static_cast< double * >(Na_buffer.ptr), Na_buffer.shape[0]); */

    /*             cressman_FE< arma::vec >(V_vec, m_vec, n_vec, h_vec, Ca_vec, K_vec, Na_vec, t0, t1, dt); */
    /*         }); */
    m.def("modular_forward_euler", &modular_forward_euler< std::vector< double > >);
}
