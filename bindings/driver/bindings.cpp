#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_TransportSolution(py::module_& m);
void register_TransportDriver(py::module_& m);

// Initialize the driver module
void init_driver(py::module_& m)
{
  m.doc() = "Driver module";

  register_TransportSolution(m);
  register_TransportDriver(m);
}
