find_package(Python COMPONENTS Interpreter Development Development.Module REQUIRED)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(cnine_base bindings/cnine_py.cpp)
include(../cmake/FindTorchPython.cmake)
target_link_libraries(cnine_base PRIVATE cnine ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} pybind11::pybind11)

install(TARGETS cnine_base DESTINATION cnine_base COMPONENT cnine::python)
