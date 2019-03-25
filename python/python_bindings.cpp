#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "LinearSystem.hpp"

namespace py = pybind11;
using namespace linear_system;

// define some helper functions here to resolve which functions to call when there are multiple signatures

Eigen::VectorXd update(LinearSystem &sys, const Eigen::RowVectorXd &input, Time time)
{
    return sys.update(input, time);
}

void setInitialStateMatrix(LinearSystem &sys, const Eigen::MatrixXd & u_history)
{
    sys.setInitialState(u_history);
}

void setInitialStateVector(LinearSystem &sys, const Eigen::VectorXd & u_channels)
{
    sys.setInitialState(u_channels);
}

PYBIND11_MODULE(linear_system_py, m) {
    py::enum_<IntegrationMethod>(m, "IntegrationMethod")
        .value("TUSTIN", TUSTIN)
        .value("FORWARD_EULER", FORWARD_EULER)
        .value("BACKWARD_EULER", BACKWARD_EULER)
        .export_values();

    py::class_<LinearSystem>(m, "LinearSystem")
        .def(py::init<Eigen::VectorXd, Eigen::VectorXd, double, IntegrationMethod, double>(),
             py::arg("tf_num") = Eigen::VectorXd::Zero(2),
             py::arg("tf_den") = Eigen::VectorXd::Constant(2,1),
             py::arg("ts") = 0.001,
             py::arg("integration_method") = TUSTIN,
             py::arg("prewarp") = 0)
        .def_static("getTimeFromSeconds", &LinearSystem::getTimeFromSeconds)
        .def("setIntegrationMethod", &LinearSystem::setIntegrationMethod)
        .def("getIntegrationMethod", &LinearSystem::getIntegrationMethod)
        .def("setPrewarpFrequency", &LinearSystem::setPrewarpFrequency)
        .def("getPrewarpFrequency", &LinearSystem::getPrewarpFrequency)
        .def("getOrder", &LinearSystem::getOrder)
        .def("getCoefficients", &LinearSystem::getCoefficients)
        .def("useNFilters", &LinearSystem::useNFilters)
        .def("setFilter", &LinearSystem::setFilter)
        .def("setSampling", &LinearSystem::setSampling)
        .def("getSampling", &LinearSystem::getSampling)
        .def("getMaximumTimeBetweenUpdates", &LinearSystem::getMaximumTimeBetweenUpdates)
        .def("getNFilters", &LinearSystem::getNFilters)
        .def("getOutput", &LinearSystem::getOutput)
        .def("setMaximumTimeBetweenUpdates", &LinearSystem::setMaximumTimeBetweenUpdates)
        .def("discretizeSystem", &LinearSystem::discretizeSystem)
        .def("setInitialTime", &LinearSystem::setInitialTime)
        .def("update", &update)
        .def("setState", &LinearSystem::setState)
        .def("getState", &LinearSystem::getState)
        .def("setInitialStateMIMO", &setInitialStateMatrix)
        .def("setInitialStateSISO", &setInitialStateVector)
        .def("setInitialOutputDerivatives", &LinearSystem::setInitialOutputDerivatives)
    ;

}
