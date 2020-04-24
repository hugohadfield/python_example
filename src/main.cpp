#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "cga.h"

namespace py = pybind11;


py::array basic_op(py::array xs, py::array ys){
    /* A basic operation */
    CGA a = CGA::from_np_array(xs);
    CGA b = CGA::from_np_array(ys);
    CGA res = ((a|b)*b);
    CGA res2 = res*res;
    return res2.to_np_array();
}

PYBIND11_MODULE(cga_cpp, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cga_cpp

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("basic_op", &basic_op);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
