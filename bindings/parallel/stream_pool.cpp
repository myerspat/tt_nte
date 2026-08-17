#include "ttnte/parallel/stream_pool.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_StreamPool(py::module_& m)
{
  using namespace ttnte::parallel;

  py::class_<StreamPool, StreamPool::Ptr>(m, "StreamPool")
    // =================================================================
    // Public constructors
    .def(
      py::init([](int num_streams) { return StreamPool::create(num_streams); }),
      py::arg("num_streams") = 16)
    // =================================================================
    // Public methods
    .def("try_acquire", &StreamPool::try_acquire,
      py::call_guard<py::gil_scoped_release>())
    .def("release", &StreamPool::release, py::arg("stream"),
      py::call_guard<py::gil_scoped_release>())
    .def("size", &StreamPool::size);
}
