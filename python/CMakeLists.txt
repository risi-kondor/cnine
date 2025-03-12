
find_package(Python COMPONENTS Interpreter Development Development.Module REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(cnine_base bindings/cnine_py.cpp)
target_link_libraries(cnine_base PUBLIC cnine)
if(WITH_CUDA)
  # target_link_libraries(cnine_base PUBLIC cnine_cuda)
endif(WITH_CUDA)

install(TARGETS cnine_base DESTINATION cnine_base COMPONENT cnine::python)
